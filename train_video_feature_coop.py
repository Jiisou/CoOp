#!/usr/bin/env python3
"""
Training script for CoOp prompt learning on pre-extracted video features.

Uses MobileCLIP S0 text encoder with learnable prompt context vectors.
Follows the training loop pattern from ExploreVAD/vanilla_spotting/clip/npy_training/train_resmlp.py.

Architecture:
    Input: (B, T, D) pre-extracted video features
        -> Temporal Aggregation -> (B, D)
        -> L2 Normalize
        -> Cosine Similarity with learned text prompts
        -> CrossEntropyLoss

Usage:
    python train_video_feature_coop.py \
        --feature-dir /path/to/features/train \
        --val-feature-dir /path/to/features/val \
        --mobileclip-path /path/to/mobileclip_s0.pt \
        --epochs 50
"""

import argparse
import os
import random
from collections import defaultdict
from datetime import datetime
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from datasets.video_features import VideoFeatureDataset
from trainers.video_feature_coop import load_mobileclip, VideoFeatureCLIP


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping based on validation metric."""

    def __init__(self, patience=5, mode="max"):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score
        else:
            improved = score < self.best_score

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_lr(optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


def save_checkpoint(model, optimizer, epoch, val_loss, val_acc, path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "prompt_learner_state_dict": model.prompt_learner.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
        },
        path,
    )
    print(f"Checkpoint saved: {path}")


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    # Only load prompt_learner weights (ignore fixed token buffers)
    state_dict = checkpoint["prompt_learner_state_dict"]
    state_dict = {
        k: v for k, v in state_dict.items()
        if "token_prefix" not in k and "token_suffix" not in k
    }
    model.prompt_learner.load_state_dict(state_dict, strict=False)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Loaded checkpoint from {path} (epoch {checkpoint.get('epoch', '?')})")
    return checkpoint.get("epoch", 0)


def compute_accuracy(output, target, topk=(1,)):
    """Compute top-k accuracy."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


# ---------------------------------------------------------------------------
# Training and Validation
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    scheduler=None,
    desc: str = "Training",
) -> Tuple[float, float]:
    """Train model for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    loss_meter = AverageMeter()
    correct = 0
    total = 0

    pbar = tqdm(data_loader, desc=desc, leave=True)
    for features, labels, _ in pbar:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        loss_meter.update(loss.item(), features.size(0))
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "acc": f"{100 * correct / total:.1f}%",
            "lr": f"{get_lr(optimizer):.2e}",
        })

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return loss_meter.avg, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    classnames: list = None,
    desc: str = "Validation",
) -> Tuple[float, float, dict, float]:
    """Validate model. Returns (avg_loss, accuracy, per_class_acc, auc)."""
    model.eval()
    loss_meter = AverageMeter()

    all_labels = []
    all_preds = []
    all_logits = []

    pbar = tqdm(data_loader, desc=desc, leave=True)
    for features, labels, _ in pbar:
        features = features.to(device)
        labels_dev = labels.to(device)

        logits = model(features)
        loss = F.cross_entropy(logits, labels_dev)

        _, predicted = logits.max(1)

        loss_meter.update(loss.item(), features.size(0))
        all_labels.extend(labels.numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_logits.extend(F.softmax(logits, dim=1).cpu().numpy())

        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_logits = np.array(all_logits)

    accuracy = 100.0 * (all_labels == all_preds).sum() / len(all_labels)

    # Calculate AUC (one-vs-rest for multi-class)
    try:
        auc = roc_auc_score(all_labels, all_logits, multi_class="ovr", zero_division=0)
    except Exception as e:
        auc = 0.0
        print(f"Warning: AUC calculation failed - {e}")

    # Per-class accuracy
    per_class_acc = {}
    for cls_idx in sorted(set(all_labels)):
        mask = all_labels == cls_idx
        cls_acc = 100.0 * (all_preds[mask] == cls_idx).sum() / mask.sum()
        cls_name = classnames[cls_idx] if classnames and cls_idx < len(classnames) else f"class_{cls_idx}"
        per_class_acc[cls_name] = cls_acc

    return loss_meter.avg, accuracy, per_class_acc, auc


@torch.no_grad()
def validate_video_level(
    model: nn.Module,
    dataset: VideoFeatureDataset,
    data_loader: DataLoader,
    device: torch.device,
    classnames: list = None,
) -> Tuple[float, dict]:
    """Video-level evaluation by aggregating frame logits per video."""
    model.eval()

    video_logits = defaultdict(list)
    video_labels = {}

    for features, labels, _ in tqdm(data_loader, desc="Video-level eval", leave=True):
        features = features.to(device)
        logits = model(features)

        batch_start = data_loader.dataset.samples if hasattr(data_loader.dataset, 'samples') else []
        # We need to track video IDs, so iterate over batch
        # This is done through the dataset's sample list
        pass

    # Simpler approach: iterate through entire dataset
    video_logits = defaultdict(list)
    video_labels = {}

    all_video_ids = dataset.get_video_ids()
    all_labels_list = dataset.get_labels()

    idx = 0
    for features, labels in data_loader:
        features = features.to(device)
        logits = model(features)

        batch_size = features.size(0)
        for i in range(batch_size):
            vid_id = all_video_ids[idx]
            video_logits[vid_id].append(logits[i].cpu())
            video_labels[vid_id] = all_labels_list[idx]
            idx += 1

    # Aggregate logits per video
    correct = 0
    total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    for vid_id, logit_list in video_logits.items():
        avg_logit = torch.stack(logit_list).mean(dim=0)
        pred = avg_logit.argmax().item()
        label = video_labels[vid_id]

        total += 1
        per_class_total[label] += 1
        if pred == label:
            correct += 1
            per_class_correct[label] += 1

    video_acc = 100.0 * correct / total if total > 0 else 0.0

    per_class_acc = {}
    for cls_idx in sorted(per_class_total.keys()):
        cls_name = classnames[cls_idx] if classnames and cls_idx < len(classnames) else f"class_{cls_idx}"
        per_class_acc[cls_name] = (
            100.0 * per_class_correct[cls_idx] / per_class_total[cls_idx]
            if per_class_total[cls_idx] > 0 else 0.0
        )

    return video_acc, per_class_acc


# ---------------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------------

def train(
    feature_dir: str,
    val_feature_dir: str,
    mobileclip_path: str = None,
    mobileclip_model: str = "mobileclip2_s0",
    annotation_dir: str = None,
    val_annotation_dir: str = None,
    normal_class: str = "normal",
    unit_duration: int = 1,
    overlap_ratio: float = 0.0,
    strict_normal_sampling: bool = True,
    use_video_level_pooling: bool = False,
    temporal_agg: str = "mean",
    n_ctx: int = 16,
    ctx_init: str = "",
    csc: bool = False,
    class_token_position: str = "end",
    batch_size: int = 32,
    num_workers: int = 4,
    epochs: int = 50,
    lr: float = 0.002,
    weight_decay: float = 0.0,
    warmup_epochs: int = 1,
    patience: int = 10,
    checkpoint_dir: str = "./output_ckpts/video_feature_coop",
    save_name: str = "video_feature_coop",
    save_interval: int = 10,
    log_dir: str = "./output_ckpts/video_feature_coop/tensorboard",
    seed: int = 42,
    resume_ckpt: str = None,
    eval_only: bool = False,
):
    """Main training function for CoOp prompt learning on video features."""
    set_seed(seed)
    device = get_device()

    timestamp = datetime.now().strftime("%y%m%d%H%M")

    # TensorBoard
    run_name = f"{save_name}_{timestamp}"
    writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))
    print(f"TensorBoard log dir: {writer.log_dir}")

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------
    print("\nLoading training dataset...")
    train_dataset = VideoFeatureDataset(
        feature_dir=feature_dir,
        annotation_dir=annotation_dir,
        normal_class=normal_class,
        unit_duration=unit_duration,
        overlap_ratio=overlap_ratio,
        strict_normal_sampling=strict_normal_sampling,
        use_video_level_pooling=use_video_level_pooling,
        verbose=True,
        seed=seed,
    )

    print("\nLoading validation dataset...")
    val_dataset = VideoFeatureDataset(
        feature_dir=val_feature_dir,
        annotation_dir=val_annotation_dir,
        normal_class=normal_class,
        unit_duration=unit_duration,
        overlap_ratio=0.0,  # No overlap for validation
        strict_normal_sampling=False,  # Keep all samples for validation
        use_video_level_pooling=use_video_level_pooling,
        verbose=True,
        seed=seed,
    )

    classnames = train_dataset.classnames
    num_classes = len(classnames)
    print(f"\nClasses ({num_classes}): {classnames}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    if mobileclip_path:
        print(f"\nLoading MobileCLIP ({mobileclip_model}) from: {mobileclip_path}")
    else:
        print(f"\nAuto-loading MobileCLIP ({mobileclip_model})...")
    clip_model, tokenizer = load_mobileclip(
        pretrained_path=mobileclip_path,
        model_name=mobileclip_model,
        device="cpu"
    )

    model = VideoFeatureCLIP(
        classnames=classnames,
        clip_model=clip_model,
        tokenizer=tokenizer,
        n_ctx=n_ctx,
        ctx_init=ctx_init,
        csc=csc,
        class_token_position=class_token_position,
        temporal_agg=temporal_agg,
    )

    # Freeze everything except prompt_learner
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / Total: {total_params:,}")

    model = model.to(device)

    # -----------------------------------------------------------------------
    # Optimizer and Scheduler
    # -----------------------------------------------------------------------
    optimizer = optim.SGD(
        model.prompt_learner.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader) * warmup_epochs

    warmup_scheduler = LinearLR(
        optimizer, start_factor=1e-5 / lr if lr > 0 else 0.1, total_iters=warmup_steps,
    )
    main_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps],
    )

    os.makedirs(checkpoint_dir, exist_ok=True)

    best_acc = 0.0
    early_stopping = EarlyStopping(patience=patience, mode="max")

    # Resume from checkpoint
    if resume_ckpt is not None:
        print(f"\nLoading checkpoint: {resume_ckpt}")
        load_checkpoint(resume_ckpt, model, optimizer, device=device)

    # -----------------------------------------------------------------------
    # Eval-only mode
    # -----------------------------------------------------------------------
    if eval_only:
        print("\n" + "=" * 60)
        print("Evaluation Only Mode")
        print("=" * 60)

        val_loss, val_acc, per_class, val_auc = validate(
            model, val_loader, device, classnames, desc="Test",
        )
        print(f"Frame-level accuracy: {val_acc:.2f}%")
        print(f"Frame-level AUC: {val_auc:.4f}")
        for cls_name, cls_acc in per_class.items():
            print(f"  {cls_name}: {cls_acc:.2f}%")

        video_acc, video_per_class = validate_video_level(
            model, val_dataset, val_loader, device, classnames,
        )
        print(f"\nVideo-level accuracy: {video_acc:.2f}%")
        for cls_name, cls_acc in video_per_class.items():
            print(f"  {cls_name}: {cls_acc:.2f}%")

        writer.close()
        return model

    # -----------------------------------------------------------------------
    # Training Loop
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CoOp Prompt Learning on Video Features")
    print("=" * 60)

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device,
            scheduler=scheduler,
            desc=f"Epoch {epoch + 1}/{epochs}",
        )

        val_loss, val_acc, per_class, val_auc = validate(
            model, val_loader, device, classnames, desc="Validation",
        )

        print(
            f"Epoch {epoch + 1}: "
            f"Train Loss={train_loss:.4f}, Acc={train_acc:.1f}% | "
            f"Val Loss={val_loss:.4f}, Acc={val_acc:.1f}%, AUC={val_auc:.4f}"
        )
        for cls_name, cls_acc in per_class.items():
            print(f"  {cls_name}: {cls_acc:.1f}%")

        # TensorBoard
        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)
        writer.add_scalar("AUC", val_auc, epoch)
        writer.add_scalar("LR", get_lr(optimizer), epoch)
        for cls_name, cls_acc in per_class.items():
            writer.add_scalar(f"PerClass/{cls_name}", cls_acc, epoch)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(checkpoint_dir, f"{save_name}_best.pth")
            print(f"\nSaving best model when epoch {epoch + 1}")
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, save_path)

        # Periodic checkpoint
        if (epoch + 1) % save_interval == 0:
            save_path = os.path.join(checkpoint_dir, f"{save_name}_ep{epoch + 1}.pth")
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, save_path)

        # Early stopping
        if early_stopping(val_acc):
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Save final model
    final_path = os.path.join(checkpoint_dir, f"{save_name}_final.pth")
    save_checkpoint(model, optimizer, epochs, val_loss, val_acc, final_path)

    # Video-level evaluation on best model
    print("\n" + "=" * 60)
    print("Final Video-Level Evaluation (best model)")
    print("=" * 60)
    best_path = os.path.join(checkpoint_dir, f"{save_name}_best.pth")
    if os.path.exists(best_path):
        load_checkpoint(best_path, model, device=device)
    video_acc, video_per_class = validate_video_level(
        model, val_dataset, val_loader, device, classnames,
    )
    print(f"Video-level accuracy: {video_acc:.2f}%")
    for cls_name, cls_acc in video_per_class.items():
        print(f"  {cls_name}: {cls_acc:.2f}%")

    # Log hparams
    writer.add_hparams(
        {
            "n_ctx": n_ctx,
            "ctx_init": ctx_init if ctx_init else "random",
            "csc": csc,
            "class_token_position": class_token_position,
            "temporal_agg": temporal_agg,
            "unit_duration": unit_duration,
            "batch_size": batch_size,
            "lr": lr,
            "strict_normal_sampling": strict_normal_sampling,
        },
        {
            "hparam/best_val_acc": best_acc,
            "hparam/video_acc": video_acc,
        },
    )
    writer.close()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Video-Level Accuracy: {video_acc:.2f}%")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"TensorBoard: {writer.log_dir}")
    print("=" * 60)

    return model


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="CoOp Prompt Learning on Pre-Extracted Video Features (MobileCLIP2-S0)"
    )

    # Data
    parser.add_argument("--feature-dir", type=str, required=True,
                        help="Path to training feature directory with class subdirs")
    parser.add_argument("--val-feature-dir", type=str, required=True,
                        help="Path to validation feature directory with class subdirs")
    parser.add_argument("--annotation-dir", type=str, default=None,
                        help="Path to annotation CSV directory (for strict normal filtering)")
    parser.add_argument("--val-annotation-dir", type=str, default=None,
                        help="Path to validation annotation CSV directory")
    parser.add_argument("--normal-class", type=str, default="Normal",
                        help="Name of the normal class directory (case-insensitive)")

    # Window
    parser.add_argument("--unit-duration", type=int, default=1,
                        help="Window size in seconds (frames per sample)")
    parser.add_argument("--overlap-ratio", type=float, default=0.0,
                        help="Sliding window overlap ratio for training")
    parser.add_argument("--strict-normal-sampling", action="store_true", default=True,
                        help="Apply strict normal snippet filtering")
    parser.add_argument("--no-strict-normal-sampling", dest="strict_normal_sampling",
                        action="store_false",
                        help="Disable strict normal snippet filtering")
    parser.add_argument("--use-video-level-pooling", action="store_true", default=False,
                        help="Use mean pooling to aggregate each video [T, D] -> [D] as single sample")

    # MobileCLIP
    parser.add_argument("--mobileclip-path", type=str, default=None,
                        help="Path to MobileCLIP pretrained weights (.pt). If not provided, auto-downloads.")
    parser.add_argument("--mobileclip-model", type=str, default="mobileclip2_s0",
                        choices=["mobileclip2_s0", "mobileclip_s0"],
                        help="MobileCLIP model variant")

    # CoOp
    parser.add_argument("--n-ctx", type=int, default=16,
                        help="Number of learnable context tokens")
    parser.add_argument("--ctx-init", type=str, default="",
                        help="Context initialization words (e.g., 'a video of a')")
    parser.add_argument("--csc", action="store_true", default=True,
                        help="Use class-specific context")
    parser.add_argument("--class-token-position", type=str, default="end",
                        choices=["end", "middle", "front"],
                        help="Position of class name token in prompt")
    parser.add_argument("--temporal-agg", type=str, default="mean",
                        choices=["mean", "max"],
                        help="Temporal aggregation method for video features")

    # Training
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.002,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="Weight decay")
    parser.add_argument("--warmup-epochs", type=int, default=1,
                        help="Warmup epochs")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")

    # Output
    parser.add_argument("--checkpoint-dir", type=str,
                        default="./output_ckpts/video_feature_coop",
                        help="Checkpoint save directory")
    parser.add_argument("--save-name", type=str, default="video_feature_coop",
                        help="Base name for saved models")
    parser.add_argument("--save-interval", type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--log-dir", type=str,
                        default="./output_ckpts/video_feature_coop/tensorboard",
                        help="TensorBoard log directory")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume-ckpt", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--eval-only", action="store_true", default=False,
                        help="Evaluation only mode")

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Training Start at {datetime.now()}\n")
    print("=" * 60)
    print("CoOp Prompt Learning - Video Features (MobileCLIP S0)")
    print("=" * 60)
    print(f"Feature dir:     {args.feature_dir}")
    print(f"Val feature dir: {args.val_feature_dir}")
    print(f"MobileCLIP path: {args.mobileclip_path}")
    print(f"N_CTX:           {args.n_ctx}")
    print(f"CTX_INIT:        {args.ctx_init if args.ctx_init else '(random)'}")
    print(f"CSC:             {args.csc}")
    print(f"Class position:  {args.class_token_position}")
    print(f"Temporal agg:    {args.temporal_agg}")
    print(f"Video pooling:   {args.use_video_level_pooling}")
    print(f"Unit duration:   {args.unit_duration}s")
    print(f"Overlap ratio:   {args.overlap_ratio}")
    print(f"Strict normal:   {args.strict_normal_sampling}")
    print(f"Batch size:      {args.batch_size}")
    print(f"LR:              {args.lr}")
    print(f"Epochs:          {args.epochs}")
    print(f"Seed:            {args.seed}")
    print("=" * 60)

    train(
        feature_dir=args.feature_dir,
        val_feature_dir=args.val_feature_dir,
        mobileclip_path=args.mobileclip_path,
        mobileclip_model=args.mobileclip_model,
        annotation_dir=args.annotation_dir,
        val_annotation_dir=args.val_annotation_dir,
        normal_class=args.normal_class,
        unit_duration=args.unit_duration,
        overlap_ratio=args.overlap_ratio,
        strict_normal_sampling=args.strict_normal_sampling,
        use_video_level_pooling=args.use_video_level_pooling,
        temporal_agg=args.temporal_agg,
        n_ctx=args.n_ctx,
        ctx_init=args.ctx_init,
        csc=args.csc,
        class_token_position=args.class_token_position,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
        save_name=args.save_name,
        save_interval=args.save_interval,
        log_dir=args.log_dir,
        seed=args.seed,
        resume_ckpt=args.resume_ckpt,
        eval_only=args.eval_only,
    )


if __name__ == "__main__":
    main()
