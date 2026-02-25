"""
Verify cache-based inference parity with evaluate_with_temporal_gt.py.

This script:
1) Runs inference on test features using cached text features (.pt/.npz)
2) Computes the same binary anomaly metrics as evaluate_with_temporal_gt.py
3) Optionally runs checkpoint-based model inference and compares predictions/logits/metrics
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

from datasets.video_features import VideoFeatureDataset
from trainers.video_feature_coop import (
    load_mobileclip,
    VideoFeatureCLIP,
    _get_text_encoder_components,
)
from evaluate_with_temporal_gt import parse_annotation_file, evaluate as reference_evaluate


def load_cache(cache_path):
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    if path.suffix.lower() == ".pt":
        obj = torch.load(path, map_location="cpu")
        if "text_features_norm" in obj:
            text_features = obj["text_features_norm"].float().cpu().numpy()
        elif "text_features" in obj:
            tf = obj["text_features"].float()
            text_features = F.normalize(tf, dim=-1).cpu().numpy()
        else:
            raise KeyError("PT cache must include 'text_features_norm' or 'text_features'")
        classnames = obj.get("classnames", None)
        meta = {
            "source": "pt",
            "checkpoint_path": obj.get("checkpoint_path", None),
            "n_cls": int(obj.get("n_cls", text_features.shape[0])),
            "embed_dim": int(obj.get("embed_dim", text_features.shape[1])),
            "classnames": classnames,
        }
        return text_features, meta

    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        if "text_features_norm" in data:
            text_features = data["text_features_norm"].astype(np.float32)
        elif "text_features" in data:
            tf = torch.from_numpy(data["text_features"]).float()
            text_features = F.normalize(tf, dim=-1).cpu().numpy()
        else:
            raise KeyError("NPZ cache must include 'text_features_norm' or 'text_features'")
        classnames = data["classnames"].tolist() if "classnames" in data else None
        meta = {
            "source": "npz",
            "checkpoint_path": None,
            "n_cls": int(text_features.shape[0]),
            "embed_dim": int(text_features.shape[1]),
            "classnames": classnames,
        }
        return text_features, meta

    raise ValueError("Unsupported cache format. Use .pt or .npz")


@torch.no_grad()
def cache_inference(data_loader, device, text_features_norm, logit_scale, temporal_agg="mean"):
    text_features = torch.from_numpy(text_features_norm).to(device)
    text_features = F.normalize(text_features, dim=-1)
    scale = logit_scale.exp()

    all_logits = []
    all_labels = []
    all_video_ids = []

    pbar = tqdm(data_loader, desc="Cache inference", leave=True)
    for batch_data in pbar:
        if len(batch_data) == 3:
            features, labels, video_ids = batch_data
        else:
            features, labels = batch_data
            video_ids = None

        features = features.to(device)
        labels = labels.to(device)

        if features.dim() == 3:
            if temporal_agg == "mean":
                image_features = features.mean(dim=1)
            elif temporal_agg == "max":
                image_features = features.max(dim=1).values
            else:
                raise ValueError(f"Unknown temporal_agg: {temporal_agg}")
        else:
            image_features = features

        image_features = F.normalize(image_features, dim=-1)
        logits = scale * image_features @ text_features.t()

        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        if video_ids is not None:
            all_video_ids.extend(video_ids)

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_logits, all_labels, all_video_ids


def compute_binary_metrics_from_logits(all_logits, all_labels, all_video_ids, annotations, classnames, normal_class):
    class_probs = torch.softmax(torch.tensor(all_logits), dim=1).numpy()

    binary_labels = []
    annotation_keys_normalized = {}
    for key in annotations.keys():
        normalized = key.replace("_x264", "").replace(".mp4", "")
        annotation_keys_normalized[normalized] = key

    for video_id in all_video_ids:
        matched = False
        if video_id in annotations:
            has_anomaly = len(annotations[video_id]["events"]) > 0
            binary_labels.append(1 if has_anomaly else 0)
            matched = True
        else:
            normalized = video_id.replace("_x264", "").replace(".mp4", "")
            if normalized in annotation_keys_normalized:
                orig_key = annotation_keys_normalized[normalized]
                has_anomaly = len(annotations[orig_key]["events"]) > 0
                binary_labels.append(1 if has_anomaly else 0)
                matched = True

        if not matched:
            binary_labels.append(-1)

    binary_labels = np.array(binary_labels)
    valid_mask = binary_labels != -1
    valid_binary_labels = binary_labels[valid_mask]
    valid_class_probs = class_probs[valid_mask]

    if len(valid_binary_labels) == 0:
        return {
            "valid_count": 0,
            "metrics_available": False,
            "reason": "No matched annotations for any sample",
        }

    normal_idx = classnames.index(normal_class) if normal_class in classnames else 0
    anomaly_prob = 1.0 - valid_class_probs[:, normal_idx]
    binary_preds = (anomaly_prob > 0.5).astype(int)

    result = {
        "valid_count": int(np.sum(valid_mask)),
        "total_count": int(len(binary_labels)),
        "anomaly_count": int(np.sum(valid_binary_labels)),
        "normal_count": int(len(valid_binary_labels) - np.sum(valid_binary_labels)),
        "metrics_available": len(np.unique(valid_binary_labels)) > 1,
        "normal_idx": int(normal_idx),
        "valid_mask": valid_mask.tolist(),
        "binary_labels": valid_binary_labels.tolist(),
        "binary_preds": binary_preds.tolist(),
        "anomaly_prob": anomaly_prob.tolist(),
    }

    if result["metrics_available"]:
        result["accuracy"] = float(accuracy_score(valid_binary_labels, binary_preds))
        result["precision"] = float(precision_score(valid_binary_labels, binary_preds, zero_division=0))
        result["recall"] = float(recall_score(valid_binary_labels, binary_preds, zero_division=0))
        result["f1"] = float(f1_score(valid_binary_labels, binary_preds, zero_division=0))
        result["auc_roc"] = float(roc_auc_score(valid_binary_labels, anomaly_prob))
        result["auc_pr"] = float(average_precision_score(valid_binary_labels, anomaly_prob))

    return result


def load_reference_model(args, classnames, device):
    clip_model, tokenizer = load_mobileclip(
        pretrained_path=args.mobileclip_path,
        model_name=args.mobileclip_model,
        device="cpu",
    )

    default_prompts = {}
    for cls in classnames:
        if cls.lower() == "normal":
            default_prompts[cls] = "a normal scene without any anomaly"
        else:
            default_prompts[cls] = f"a video with {cls.lower()} event"

    model = VideoFeatureCLIP(
        classnames=classnames,
        clip_model=clip_model,
        tokenizer=tokenizer,
        n_ctx=args.n_ctx,
        class_prompts=default_prompts,
        csc=args.csc,
        class_token_position="end",
        temporal_agg="mean",
    )

    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("prompt_learner_state_dict", {})
    state_dict = {
        k: v for k, v in state_dict.items()
        if "token_prefix" not in k and "token_suffix" not in k
    }

    if "ctx" in state_dict:
        checkpoint_ctx = state_dict["ctx"]
        current_ctx_shape = model.prompt_learner.ctx.shape
        checkpoint_ctx_shape = checkpoint_ctx.shape

        if checkpoint_ctx_shape != current_ctx_shape:
            if checkpoint_ctx_shape[0] != current_ctx_shape[0]:
                state_dict.pop("ctx")
            elif checkpoint_ctx_shape[1] < current_ctx_shape[1]:
                n_cls, checkpoint_n_ctx, ctx_dim = checkpoint_ctx_shape
                current_n_ctx = current_ctx_shape[1]
                pad_n = current_n_ctx - checkpoint_n_ctx
                padding = torch.empty(n_cls, pad_n, ctx_dim, dtype=checkpoint_ctx.dtype)
                torch.nn.init.normal_(padding, std=0.02)
                state_dict["ctx"] = torch.cat([checkpoint_ctx, padding], dim=1)
            elif checkpoint_ctx_shape[1] > current_ctx_shape[1]:
                state_dict["ctx"] = checkpoint_ctx[:, :current_ctx_shape[1], :]

    model.prompt_learner.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Verify cached prompt/text inference parity with temporal GT evaluation"
    )
    parser.add_argument("--cache-path", type=str, required=True, help="Path to prompt_text_cache.pt/.npz")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Optional checkpoint for reference comparison")
    parser.add_argument("--test-feature-dir", type=str, default="/mnt/c/JJS/UCF_Crimes/Features/MCi20-avgpooled/test")
    parser.add_argument(
        "--annotation-file",
        type=str,
        default="./annotation/Temporal_Anomaly_Annotation_For_Testing_Videos/Txt_formate/Temporal_Anomaly_Annotation.txt",
    )
    parser.add_argument("--mobileclip-model", type=str, default="mobileclip2_s0")
    parser.add_argument("--mobileclip-path", type=str, default=None)
    parser.add_argument("--normal-class", type=str, default="Normal")
    parser.add_argument("--n-ctx", type=int, default=16)
    parser.add_argument("--csc", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./output/cache_temporal_verify")
    parser.add_argument("--logit-atol", type=float, default=1e-5)
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    annotations = parse_annotation_file(args.annotation_file)
    dataset = VideoFeatureDataset(
        feature_dir=args.test_feature_dir,
        annotation_dir=None,
        normal_class=args.normal_class,
        unit_duration=1,
        overlap_ratio=0.0,
        strict_normal_sampling=False,
        use_video_level_pooling=False,
        verbose=True,
        seed=42,
    )
    classnames = dataset.classnames
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    text_features_norm, cache_meta = load_cache(args.cache_path) 
    if text_features_norm.shape[0] != len(classnames):
        raise ValueError(
            f"Cache n_cls ({text_features_norm.shape[0]}) != dataset n_cls ({len(classnames)})"
        )

    if cache_meta.get("classnames") is not None:
        if list(cache_meta["classnames"]) != list(classnames):
            print("Warning: cache classnames order differs from dataset classnames")

    clip_model, _ = load_mobileclip(
        pretrained_path=args.mobileclip_path,
        model_name=args.mobileclip_model,
        device=str(device),
    )
    logit_scale = _get_text_encoder_components(clip_model)["logit_scale"].to(device)

    cache_logits, cache_labels, cache_video_ids = cache_inference(
        loader, device, text_features_norm, logit_scale, temporal_agg="mean"
    )

    cache_binary = compute_binary_metrics_from_logits(
        cache_logits, cache_labels, cache_video_ids, annotations, classnames, args.normal_class
    )

    report = {
        "cache_path": os.path.abspath(args.cache_path),
        "cache_meta": cache_meta,
        "cache_binary": cache_binary,
        "comparison_with_reference": None,
    }

    print("\nCache Binary Metrics")
    if cache_binary.get("metrics_available", False):
        print(f"  Accuracy: {cache_binary['accuracy']:.4f}")
        print(f"  F1:       {cache_binary['f1']:.4f}")
        print(f"  AUC-ROC:  {cache_binary['auc_roc']:.4f}")
    else:
        print(f"  metrics_available: False ({cache_binary.get('reason', 'insufficient labels')})")

    if args.checkpoint_path:
        ref_model = load_reference_model(args, classnames, device)
        ref_logits, ref_labels, ref_video_ids = reference_evaluate(
            ref_model, loader, device, classnames, desc="Reference model inference"
        )
        ref_binary = compute_binary_metrics_from_logits(
            ref_logits, ref_labels, ref_video_ids, annotations, classnames, args.normal_class
        )

        same_order = (ref_video_ids == cache_video_ids) and np.array_equal(ref_labels, cache_labels)
        max_abs_logit_diff = float(np.max(np.abs(ref_logits - cache_logits)))
        mean_abs_logit_diff = float(np.mean(np.abs(ref_logits - cache_logits)))
        class_pred_equal = float(np.mean(ref_logits.argmax(axis=1) == cache_logits.argmax(axis=1)))

        comparison = {
            "checkpoint_path": os.path.abspath(args.checkpoint_path),
            "same_sample_order": bool(same_order),
            "max_abs_logit_diff": max_abs_logit_diff,
            "mean_abs_logit_diff": mean_abs_logit_diff,
            "class_pred_match_ratio": class_pred_equal,
            "logits_allclose": bool(np.allclose(ref_logits, cache_logits, atol=args.logit_atol, rtol=0.0)),
            "reference_binary": ref_binary,
            "delta_binary_metrics": {},
        }

        if cache_binary.get("metrics_available", False) and ref_binary.get("metrics_available", False):
            for k in ["accuracy", "f1", "auc_roc"]:
                comparison["delta_binary_metrics"][k] = float(cache_binary[k] - ref_binary[k])

        report["comparison_with_reference"] = comparison

        print("\nReference vs Cache")
        print(f"  same_sample_order:      {comparison['same_sample_order']}")
        print(f"  class_pred_match_ratio: {comparison['class_pred_match_ratio']:.6f}")
        print(f"  max_abs_logit_diff:     {comparison['max_abs_logit_diff']:.3e}")
        print(f"  mean_abs_logit_diff:    {comparison['mean_abs_logit_diff']:.3e}")
        print(f"  logits_allclose:        {comparison['logits_allclose']}")

        if ref_binary.get("metrics_available", False):
            print("\nReference Binary Metrics")
            print(f"  Accuracy: {ref_binary['accuracy']:.4f}")
            print(f"  F1:       {ref_binary['f1']:.4f}")
            print(f"  AUC-ROC:  {ref_binary['auc_roc']:.4f}")

    out_path = os.path.join(args.output_dir, "cache_temporal_verify_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report: {out_path}")


if __name__ == "__main__":
    main()
