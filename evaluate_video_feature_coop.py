"""
Evaluate CoOp-trained model on test set with video features and annotations.

Supports:
- Frame-level evaluation
- Video-level evaluation (aggregation)
- Per-class metrics
- Confusion matrix
- ROC-AUC curves
"""

import os
import sys
import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from datasets.video_features import VideoFeatureDataset
from trainers.video_feature_coop import load_mobileclip, VideoFeatureCLIP

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns


def load_checkpoint(model, checkpoint_path, device):
    """Load model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return False

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load prompt_learner weights
    state_dict = checkpoint.get("prompt_learner_state_dict", {})
    state_dict = {
        k: v for k, v in state_dict.items()
        if "token_prefix" not in k and "token_suffix" not in k
    }

    # Handle n_ctx mismatch between checkpoint and current model
    if "ctx" in state_dict:
        checkpoint_ctx = state_dict["ctx"]
        current_ctx_shape = model.prompt_learner.ctx.shape
        checkpoint_ctx_shape = checkpoint_ctx.shape

        if checkpoint_ctx_shape != current_ctx_shape:
            print(f"Handling n_ctx mismatch:")
            print(f"  Checkpoint ctx shape: {checkpoint_ctx_shape}")
            print(f"  Current model ctx shape: {current_ctx_shape}")

            if checkpoint_ctx_shape[0] != current_ctx_shape[0]:
                # n_cls mismatch - skip loading ctx
                print(f"  ⚠ Number of classes mismatch, skipping ctx loading")
                state_dict.pop("ctx")
            elif checkpoint_ctx_shape[1] < current_ctx_shape[1]:
                # Checkpoint has fewer context tokens - pad with random initialization
                n_cls, checkpoint_n_ctx, ctx_dim = checkpoint_ctx_shape
                current_n_ctx = current_ctx_shape[1]
                padding_n_ctx = current_n_ctx - checkpoint_n_ctx

                print(f"  Padding: {checkpoint_n_ctx} → {current_n_ctx} tokens")
                padding = torch.empty(n_cls, padding_n_ctx, ctx_dim, dtype=checkpoint_ctx.dtype)
                torch.nn.init.normal_(padding, std=0.02)
                padded_ctx = torch.cat([checkpoint_ctx, padding], dim=1)
                state_dict["ctx"] = padded_ctx
            elif checkpoint_ctx_shape[1] > current_ctx_shape[1]:
                # Checkpoint has more context tokens - truncate
                checkpoint_n_ctx = checkpoint_ctx_shape[1]
                current_n_ctx = current_ctx_shape[1]
                print(f"  Truncating: {checkpoint_n_ctx} → {current_n_ctx} tokens")
                state_dict["ctx"] = checkpoint_ctx[:, :current_ctx_shape[1], :]

    model.prompt_learner.load_state_dict(state_dict, strict=False)

    print(f"✓ Loaded checkpoint from {checkpoint_path}")
    return True


@torch.no_grad()
def evaluate(
    model, data_loader, device, classnames,
    desc="Evaluation"
):
    """Evaluate model on dataset.

    Returns:
        all_logits: [N, num_classes] logits
        all_labels: [N] ground truth labels
        all_video_ids: [N] video IDs
    """
    model.eval()

    all_logits = []
    all_labels = []
    all_video_ids = []

    pbar = tqdm(data_loader, desc=desc, leave=True)
    for features, labels, video_ids in pbar:
        features = features.to(device)
        labels = labels.to(device)

        # Forward pass
        logits = model(features)  # [B, num_classes]

        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_video_ids.extend(video_ids)

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_logits, all_labels, all_video_ids


def compute_metrics(labels, predictions, probabilities=None, classnames=None):
    """Compute evaluation metrics."""
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, predictions, average="weighted", zero_division=0),
    }

    # Per-class metrics
    if classnames:
        precision = precision_score(labels, predictions, average=None, zero_division=0)
        recall = recall_score(labels, predictions, average=None, zero_division=0)
        f1 = f1_score(labels, predictions, average=None, zero_division=0)

        metrics["per_class"] = {}
        for i, name in enumerate(classnames):
            metrics["per_class"][name] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
            }

    # ROC-AUC for multi-class (if probabilities available)
    if probabilities is not None and len(np.unique(labels)) > 2:
        try:
            # One-vs-rest AUC
            auc = roc_auc_score(labels, probabilities, multi_class="ovr", zero_division=0)
            metrics["roc_auc_ovr"] = float(auc)
        except:
            pass

    return metrics


def aggregate_predictions(all_logits, all_labels, all_video_ids, classnames, method="max"):
    """Aggregate frame-level predictions to video level.

    Args:
        method: "max", "mean", or "voting"
    """
    video_preds = {}
    video_labels = {}
    video_logits = {}

    for logits, label, video_id in zip(all_logits, all_labels, all_video_ids):
        if video_id not in video_logits:
            video_logits[video_id] = []
            video_labels[video_id] = label

        video_logits[video_id].append(logits)

    # Aggregate logits
    for video_id, logits_list in video_logits.items():
        logits_array = np.array(logits_list)  # [num_frames, num_classes]

        if method == "max":
            agg_logits = logits_array.max(axis=0)
        elif method == "mean":
            agg_logits = logits_array.mean(axis=0)
        elif method == "voting":
            # Use max logits per frame as voting
            frame_preds = logits_array.argmax(axis=1)
            agg_logits = np.zeros(logits_array.shape[1])
            for i, pred in enumerate(frame_preds):
                agg_logits[pred] += 1

        video_preds[video_id] = agg_logits.argmax()

    # Convert to arrays
    video_ids_list = sorted(video_labels.keys())
    video_labels_array = np.array([video_labels[vid] for vid in video_ids_list])
    video_preds_array = np.array([video_preds[vid] for vid in video_ids_list])

    return video_preds_array, video_labels_array, video_ids_list


def plot_confusion_matrix(labels, predictions, classnames, output_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(labels, predictions)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=classnames,
        yticklabels=classnames,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✓ Saved confusion matrix to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CoOp model on test video features"
    )
    parser.add_argument("--test-feature-dir", type=str, required=True,
                        help="Path to test feature directory")
    parser.add_argument("--test-annotation-dir", type=str, default=None,
                        help="Path to test annotation CSV directory")
    parser.add_argument("--checkpoint-path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--mobileclip-model", type=str, default="mobileclip2_s0",
                        help="MobileCLIP model variant")
    parser.add_argument("--mobileclip-path", type=str, default=None,
                        help="Path to MobileCLIP pretrained weights")
    parser.add_argument("--normal-class", type=str, default="Normal",
                        help="Name of normal class")
    parser.add_argument("--n-ctx", type=int, default=16,
                        help="Number of context tokens")
    parser.add_argument("--csc", action="store_true", default=False,
                        help="Use class-specific context")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loader workers")
    parser.add_argument("--use-video-level-pooling", action="store_true", default=False,
                        help="Use video-level mean pooling")
    parser.add_argument("--output-dir", type=str, default="./output/evaluation",
                        help="Directory to save evaluation results")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("Loading Test Dataset")
    print("=" * 80)

    test_dataset = VideoFeatureDataset(
        feature_dir=args.test_feature_dir,
        annotation_dir=args.test_annotation_dir,
        normal_class=args.normal_class,
        unit_duration=1,
        overlap_ratio=0.0,
        strict_normal_sampling=False,  # Keep all samples for evaluation
        use_video_level_pooling=args.use_video_level_pooling,
        verbose=True,
        seed=42,
    )

    classnames = test_dataset.classnames
    num_classes = len(classnames)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print("\n" + "=" * 80)
    print("Loading Model")
    print("=" * 80)

    # Load MobileCLIP
    clip_model, tokenizer = load_mobileclip(
        pretrained_path=args.mobileclip_path,
        model_name=args.mobileclip_model,
        device="cpu"
    )

    # Create model
    model = VideoFeatureCLIP(
        classnames=classnames,
        clip_model=clip_model,
        tokenizer=tokenizer,
        n_ctx=args.n_ctx,
        ctx_init="",
        csc=args.csc,
        class_token_position="end",
        temporal_agg="mean",
    )

    # Load checkpoint
    if not load_checkpoint(model, args.checkpoint_path, device):
        return

    model = model.to(device)

    print("\n" + "=" * 80)
    print("Evaluating on Test Set")
    print("=" * 80)

    # Inference
    all_logits, all_labels, all_video_ids = evaluate(
        model, test_loader, device, classnames,
        desc="Evaluating"
    )

    print(f"\nTotal samples: {len(all_labels)}")

    # Frame-level evaluation
    print("\n" + "=" * 80)
    print("Frame-Level Evaluation")
    print("=" * 80)

    frame_predictions = all_logits.argmax(axis=1)
    frame_probabilities = torch.softmax(torch.tensor(all_logits), dim=1).numpy()

    frame_metrics = compute_metrics(
        all_labels, frame_predictions,
        probabilities=frame_probabilities,
        classnames=classnames
    )

    print(f"\nFrame-level Accuracy: {frame_metrics['accuracy']:.4f}")
    print(f"Macro F1: {frame_metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {frame_metrics['weighted_f1']:.4f}")

    if "roc_auc_ovr" in frame_metrics:
        print(f"ROC-AUC (OvR): {frame_metrics['roc_auc_ovr']:.4f}")

    print("\nPer-class metrics (Frame-level):")
    if "per_class" in frame_metrics:
        for name, metrics_dict in frame_metrics["per_class"].items():
            print(f"  {name:15s}: Precision={metrics_dict['precision']:.4f}, "
                  f"Recall={metrics_dict['recall']:.4f}, F1={metrics_dict['f1']:.4f}")

    # Video-level evaluation (if not using video-level pooling in dataset)
    if not args.use_video_level_pooling:
        print("\n" + "=" * 80)
        print("Video-Level Evaluation")
        print("=" * 80)

        video_preds, video_labels, video_ids = aggregate_predictions(
            all_logits, all_labels, all_video_ids, classnames, method="max"
        )

        video_metrics = compute_metrics(
            video_labels, video_preds,
            classnames=classnames
        )

        print(f"\nVideo-level Accuracy: {video_metrics['accuracy']:.4f}")
        print(f"Macro F1: {video_metrics['macro_f1']:.4f}")
        print(f"Weighted F1: {video_metrics['weighted_f1']:.4f}")

        print("\nPer-class metrics (Video-level):")
        if "per_class" in video_metrics:
            for name, metrics_dict in video_metrics["per_class"].items():
                print(f"  {name:15s}: Precision={metrics_dict['precision']:.4f}, "
                      f"Recall={metrics_dict['recall']:.4f}, F1={metrics_dict['f1']:.4f}")

    # Save detailed results
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)

    # Save metrics as JSON
    results = {
        "frame_level": frame_metrics,
    }

    if not args.use_video_level_pooling:
        results["video_level"] = video_metrics
        # Save video-level predictions
        video_results = {
            "predictions": video_preds.tolist(),
            "labels": video_labels.tolist(),
            "video_ids": video_ids,
        }
        with open(os.path.join(args.output_dir, "video_predictions.json"), "w") as f:
            json.dump(video_results, f, indent=2)
        print(f"✓ Saved video predictions to {args.output_dir}/video_predictions.json")

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved metrics to {args.output_dir}/metrics.json")

    # Save confusion matrices
    cm_frame_path = os.path.join(args.output_dir, "confusion_matrix_frame.png")
    plot_confusion_matrix(all_labels, frame_predictions, classnames, cm_frame_path)

    if not args.use_video_level_pooling:
        cm_video_path = os.path.join(args.output_dir, "confusion_matrix_video.png")
        plot_confusion_matrix(video_labels, video_preds, classnames, cm_video_path)

    # Save classification report
    report_frame = classification_report(
        all_labels, frame_predictions,
        target_names=classnames,
        output_dict=True
    )

    with open(os.path.join(args.output_dir, "classification_report_frame.json"), "w") as f:
        json.dump(report_frame, f, indent=2)

    if not args.use_video_level_pooling:
        report_video = classification_report(
            video_labels, video_preds,
            target_names=classnames,
            output_dict=True
        )

        with open(os.path.join(args.output_dir, "classification_report_video.json"), "w") as f:
            json.dump(report_video, f, indent=2)

    print(f"\n✓ All results saved to {args.output_dir}")

    # Print summary
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    print(f"Test samples: {len(all_labels)}")
    print(f"Classes: {num_classes}")
    print(f"Frame-level accuracy: {frame_metrics['accuracy']:.4f}")
    if not args.use_video_level_pooling:
        print(f"Video-level accuracy: {video_metrics['accuracy']:.4f}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
