"""
Evaluate CoOp model using temporal ground truth annotations.

Uses Temporal_Anomaly_Annotation.txt to:
- Map frame times to anomaly events
- Generate frame-level binary labels (normal vs anomaly)
- Calculate anomaly detection metrics (AUC-ROC, AUC-PR, etc.)
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

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns


def parse_annotation_file(annotation_path):
    """Parse temporal annotation file.

    Format:
    Video_Name  Class  Event1_Start  Event1_End  Event2_Start  Event2_End
    Abuse028_x264.mp4  Abuse  165  240  -1  -1

    Returns:
        dict: {video_name_without_ext: {'class': str, 'events': [(start, end), ...]}}
    """
    annotations = {}

    with open(annotation_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            video_name = parts[0]
            class_name = parts[1]

            # Parse event times
            events = []
            for i in range(2, len(parts), 2):
                start = int(parts[i])
                end = int(parts[i + 1])

                if start != -1 and end != -1:
                    events.append((start, end))

            # Remove .mp4 extension for matching with feature files
            video_id = video_name.replace('.mp4', '')

            annotations[video_id] = {
                'class': class_name,
                'events': events,
            }

    return annotations


def frame_to_time(frame_idx, fps=25.0):
    """Convert frame index to time in seconds."""
    return frame_idx / fps


def time_to_frame(time_sec, fps=25.0):
    """Convert time in seconds to frame index."""
    return int(time_sec * fps)


def get_temporal_label(frame_idx, events, fps=25.0):
    """Get binary label for a frame based on events.

    Args:
        frame_idx: Frame index
        events: List of (start_frame, end_frame) tuples
        fps: Frames per second

    Returns:
        1 if frame is in any event, 0 otherwise
    """
    for start, end in events:
        if start <= frame_idx < end:
            return 1
    return 0


def create_frame_level_labels(all_video_ids, annotations, fps=25.0):
    """Create frame-level binary labels from temporal annotations.

    Returns:
        dict: {video_id: np.array of binary labels for each sample}
    """
    frame_labels = {}

    for video_id in all_video_ids:
        if video_id not in annotations:
            # Video not in annotations, skip
            continue

        ann = annotations[video_id]
        events = ann['events']

        # Assume each sample corresponds to 1 second (unit_duration=1)
        # We need to create labels for the window position
        # For simplicity: label the entire window as positive if
        # the center of the window overlaps with any event

        frame_labels[video_id] = {
            'class': ann['class'],
            'is_anomaly': len(events) > 0,  # Has anomaly events
            'events': events,
        }

    return frame_labels


@torch.no_grad()
def evaluate(
    model, data_loader, device, classnames,
    desc="Evaluation"
):
    """Evaluate model on dataset."""
    model.eval()

    all_logits = []
    all_labels = []
    all_video_ids = []

    pbar = tqdm(data_loader, desc=desc, leave=True)
    for batch_data in pbar:
        if len(batch_data) == 3:
            features, labels, video_ids = batch_data
        else:
            features, labels = batch_data
            video_ids = None

        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)

        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        if video_ids is not None:
            all_video_ids.extend(video_ids)

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_logits, all_labels, all_video_ids


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CoOp model using temporal ground truth"
    )
    parser.add_argument("--test-feature-dir", type=str, # required=True,
                        default="/mnt/c/JJS/UCF_Crimes/Features/MCi20-avgpooled/test",
                        help="Path to test feature directory")
    parser.add_argument("--annotation-file", type=str, # required=True,
                        default="./annotation/Temporal_Anomaly_Annotation_For_Testing_Videos/Txt_formate/Temporal_Anomaly_Annotation.txt",
                        help="Path to temporal annotation file (Temporal_Anomaly_Annotation.txt)")
    parser.add_argument("--checkpoint-path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--mobileclip-model", type=str, default="mobileclip2_s0")
    parser.add_argument("--mobileclip-path", type=str, default=None)
    parser.add_argument("--normal-class", type=str, default="Normal")
    parser.add_argument("--n-ctx", type=int, default=16)
    parser.add_argument("--csc", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="./output/evaluation_temporal")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--fps", type=float, default=25.0,
                        help="Frames per second for annotation time conversion")

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
    print("Loading Annotations")
    print("=" * 80)

    annotations = parse_annotation_file(args.annotation_file)
    print(f"Loaded annotations for {len(annotations)} videos")

    # Count anomaly vs normal videos
    anomaly_count = sum(1 for v in annotations.values() if v['events'])
    normal_count = len(annotations) - anomaly_count
    print(f"  Anomaly videos: {anomaly_count}")
    print(f"  Normal videos: {normal_count}")

    print("\n" + "=" * 80)
    print("Loading Test Dataset")
    print("=" * 80)

    test_dataset = VideoFeatureDataset(
        feature_dir=args.test_feature_dir,
        annotation_dir=None,  # Don't use CSV annotations
        normal_class=args.normal_class,
        unit_duration=1,
        overlap_ratio=0.0,
        strict_normal_sampling=False,
        use_video_level_pooling=False,
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

    clip_model, tokenizer = load_mobileclip(
        pretrained_path=args.mobileclip_path,
        model_name=args.mobileclip_model,
        device="cpu"
    )

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
    try:
        checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Trying alternative load method...")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)

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
            print(f"\nHandling n_ctx mismatch:")
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
                print(f"  Truncating: {checkpoint_n_ctx} → {current_n_ctx} tokens")
                state_dict["ctx"] = checkpoint_ctx[:, :current_ctx_shape[1], :]

    model.prompt_learner.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    print(f"✓ Loaded checkpoint from {args.checkpoint_path}")

    print("\n" + "=" * 80)
    print("Running Inference")
    print("=" * 80)

    all_logits, all_labels, all_video_ids = evaluate(
        model, test_loader, device, classnames,
        desc="Evaluating"
    )

    print(f"\nTotal samples: {len(all_labels)}")
    print(f"Sample video IDs (first 10): {all_video_ids[:10]}")
    print(f"Annotation keys (first 10): {list(annotations.keys())[:10]}")

    # Get class-level predictions
    class_preds = all_logits.argmax(axis=1)
    class_probs = torch.softmax(torch.tensor(all_logits), dim=1).numpy()

    print("\n" + "=" * 80)
    print("Multi-Class Evaluation (Standard)")
    print("=" * 80)

    # Multi-class metrics
    class_accuracy = accuracy_score(all_labels, class_preds)
    class_precision = precision_score(all_labels, class_preds, average="weighted", zero_division=0)
    class_recall = recall_score(all_labels, class_preds, average="weighted", zero_division=0)
    class_f1 = f1_score(all_labels, class_preds, average="weighted", zero_division=0)

    print(f"Multi-class Accuracy: {class_accuracy:.4f}")
    print(f"Weighted Precision: {class_precision:.4f}")
    print(f"Weighted Recall: {class_recall:.4f}")
    print(f"Weighted F1: {class_f1:.4f}")

    print("\n" + "=" * 80)
    print("Anomaly Detection Evaluation (Binary)")
    print("=" * 80)

    # Create binary labels based on ground truth
    binary_labels = []
    video_id_mapping = {}  # Debug: track which video_ids matched

    # Normalize annotation keys for matching
    annotation_keys_normalized = {}
    for key in annotations.keys():
        # Try different normalizations
        normalized = key.replace('_x264', '').replace('.mp4', '')
        annotation_keys_normalized[normalized] = key

    for video_id in all_video_ids:
        # Try to match video_id to annotation
        matched = False

        # Try exact match
        if video_id in annotations:
            has_anomaly = len(annotations[video_id]['events']) > 0
            binary_labels.append(1 if has_anomaly else 0)
            matched = True
        else:
            # Try normalized match
            normalized = video_id.replace('_x264', '').replace('.mp4', '')
            if normalized in annotation_keys_normalized:
                orig_key = annotation_keys_normalized[normalized]
                has_anomaly = len(annotations[orig_key]['events']) > 0
                binary_labels.append(1 if has_anomaly else 0)
                matched = True
                video_id_mapping[video_id] = orig_key

        if not matched:
            binary_labels.append(-1)  # Mark as unknown

    binary_labels = np.array(binary_labels)

    matched_count = sum(binary_labels != -1)
    unmatched_count = sum(binary_labels == -1)
    print(f"\n  Video ID matching: {matched_count} matched, {unmatched_count} unmatched")

    # Debug: show some unmatched video_ids
    if unmatched_count > 0:
        unmatched_ids = [all_video_ids[i] for i in range(len(all_video_ids)) if binary_labels[i] == -1]
        print(f"\n  Sample unmatched video IDs (first 5):")
        for vid in unmatched_ids[:5]:
            print(f"    - {vid}")
            # Try different normalization
            normalized = vid.replace('_x264', '').replace('.mp4', '')
            matching_annotations = [k for k in annotations.keys() if normalized.lower() in k.lower()]
            if matching_annotations:
                print(f"      Matching annotations: {matching_annotations}")
            else:
                print(f"      No matching annotations found")

    # Filter out unknown labels
    valid_mask = binary_labels != -1
    valid_binary_labels = binary_labels[valid_mask]
    valid_class_probs = class_probs[valid_mask]
    valid_video_ids = [all_video_ids[i] for i in range(len(all_video_ids)) if valid_mask[i]]

    print(f"Videos with ground truth: {np.sum(valid_mask)} / {len(binary_labels)}")
    print(f"Anomaly videos: {np.sum(valid_binary_labels)}")
    print(f"Normal videos: {len(valid_binary_labels) - np.sum(valid_binary_labels)}")

    if len(np.unique(valid_binary_labels)) > 1:
        # Get anomaly probabilities (max probability across anomaly classes)
        # Define anomaly classes (all except Normal)
        normal_idx = classnames.index(args.normal_class) if args.normal_class in classnames else 0
        anomaly_prob = 1.0 - valid_class_probs[:, normal_idx]

        # Binary predictions
        binary_preds = (anomaly_prob > 0.5).astype(int)

        # Metrics
        binary_accuracy = accuracy_score(valid_binary_labels, binary_preds)
        binary_precision = precision_score(valid_binary_labels, binary_preds, zero_division=0)
        binary_recall = recall_score(valid_binary_labels, binary_preds, zero_division=0)
        binary_f1 = f1_score(valid_binary_labels, binary_preds, zero_division=0)

        try:
            auc_roc = roc_auc_score(valid_binary_labels, anomaly_prob)
            auc_pr = average_precision_score(valid_binary_labels, anomaly_prob)
        except:
            auc_roc = None
            auc_pr = None

        print(f"\nBinary Classification Metrics:")
        print(f"  Accuracy: {binary_accuracy:.4f}")
        print(f"  Precision: {binary_precision:.4f}")
        print(f"  Recall: {binary_recall:.4f}")
        print(f"  F1: {binary_f1:.4f}")

        if auc_roc is not None:
            print(f"  AUC-ROC: {auc_roc:.4f}")
            print(f"  AUC-PR: {auc_pr:.4f}")

        # Anomaly-only AUC (multi-class AUC for anomaly classes only)
        anomaly_only_results = {}
        print(f"\n  Anomaly-only Evaluation (among anomaly classes):")
        anomaly_mask = valid_binary_labels == 1  # Only anomaly samples
        anomaly_class_labels = all_labels[valid_mask][anomaly_mask]
        anomaly_class_probs = valid_class_probs[anomaly_mask]

        if len(np.unique(anomaly_class_labels)) > 1:
            try:
                # Multi-class AUC for anomaly samples
                anomaly_auc = roc_auc_score(
                    anomaly_class_labels,
                    anomaly_class_probs,
                    multi_class="ovr"
                )
                print(f"    Multi-class AUC (among anomalies): {anomaly_auc:.4f}")
                anomaly_only_results["auc_multi_class"] = float(anomaly_auc)

                # Per-anomaly-class precision
                anomaly_preds = anomaly_class_probs.argmax(axis=1)
                anomaly_accuracy = accuracy_score(anomaly_class_labels, anomaly_preds)
                print(f"    Anomaly-only Accuracy: {anomaly_accuracy:.4f}")
                anomaly_only_results["accuracy"] = float(anomaly_accuracy)

                # Per-class metrics within anomalies
                anomaly_precision = precision_score(anomaly_class_labels, anomaly_preds, average="weighted", zero_division=0)
                anomaly_recall = recall_score(anomaly_class_labels, anomaly_preds, average="weighted", zero_division=0)
                anomaly_f1 = f1_score(anomaly_class_labels, anomaly_preds, average="weighted", zero_division=0)

                print(f"    Anomaly-only Weighted Precision: {anomaly_precision:.4f}")
                print(f"    Anomaly-only Weighted Recall: {anomaly_recall:.4f}")
                print(f"    Anomaly-only Weighted F1: {anomaly_f1:.4f}")

                anomaly_only_results["precision"] = float(anomaly_precision)
                anomaly_only_results["recall"] = float(anomaly_recall)
                anomaly_only_results["f1"] = float(anomaly_f1)

                # Count anomaly classes in predictions
                unique_anomaly_classes = np.unique(anomaly_class_labels)
                print(f"    Anomaly classes in test set: {len(unique_anomaly_classes)}")
                print(f"      Class distribution:")
                for class_idx in unique_anomaly_classes:
                    class_name = classnames[class_idx] if class_idx < len(classnames) else f"Class_{class_idx}"
                    count = np.sum(anomaly_class_labels == class_idx)
                    print(f"        {class_name}: {count} samples")

            except Exception as e:
                print(f"    Could not compute anomaly-only AUC: {e}")
        else:
            print(f"    Not enough anomaly classes for multi-class AUC")

        # Save results
        results = {
            "multi_class": {
                "accuracy": float(class_accuracy),
                "weighted_precision": float(class_precision),
                "weighted_recall": float(class_recall),
                "weighted_f1": float(class_f1),
            },
            "binary_anomaly_detection": {
                "accuracy": float(binary_accuracy),
                "precision": float(binary_precision),
                "recall": float(binary_recall),
                "f1": float(binary_f1),
            },
            "anomaly_only": anomaly_only_results,
            "statistics": {
                "total_samples": int(len(binary_labels)),
                "samples_with_gt": int(np.sum(valid_mask)),
                "anomaly_samples": int(np.sum(valid_binary_labels)),
                "normal_samples": int(len(valid_binary_labels) - np.sum(valid_binary_labels)),
            }
        }

        if auc_roc is not None:
            results["binary_anomaly_detection"]["auc_roc"] = float(auc_roc)
            results["binary_anomaly_detection"]["auc_pr"] = float(auc_pr)

        with open(os.path.join(args.output_dir, "evaluation_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(valid_binary_labels, anomaly_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', lw=2, label=f'ROC Curve (AUC={auc_roc:.3f})')
        plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Anomaly Detection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "roc_curve.png"), dpi=150)
        plt.close()
        print(f"\n✓ Saved ROC curve to {args.output_dir}/roc_curve.png")

        # Plot Precision-Recall curve
        precision, recall, _ = precision_recall_curve(valid_binary_labels, anomaly_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, 'b-', lw=2, label=f'PR Curve (AUC={auc_pr:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - Anomaly Detection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "pr_curve.png"), dpi=150)
        plt.close()
        print(f"✓ Saved PR curve to {args.output_dir}/pr_curve.png")

        # Confusion matrix
        cm = confusion_matrix(valid_binary_labels, binary_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Anomaly'],
                    yticklabels=['Normal', 'Anomaly'],
                    cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix - Anomaly Detection')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "confusion_matrix_binary.png"), dpi=150)
        plt.close()
        print(f"✓ Saved confusion matrix to {args.output_dir}/confusion_matrix_binary.png")

        print(f"\n✓ All results saved to {args.output_dir}")

    else:
        print("⚠ Not enough samples with both classes for binary evaluation")


if __name__ == "__main__":
    main()
