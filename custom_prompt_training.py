"""
Custom prompt training for CoOp with class-specific initial prompts.

Allows specifying custom initial prompts for each class and learning
the final prompt vectors through PromptLearner.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from datasets.video_features import VideoFeatureDataset
from trainers.video_feature_coop import load_mobileclip, VideoFeatureCLIP
from train_video_feature_coop import (
    train_one_epoch, validate, save_checkpoint, get_device
)


class PromptGenerator:
    """Generate and manage class-specific initial prompts."""

    @staticmethod
    def create_default_prompts(classnames):
        """Create default prompts based on class names.

        Args:
            classnames: List of class names

        Returns:
            dict: {classname: initial_prompt_string}
        """
        prompts = {}
        for cls in classnames:
            if cls.lower() == "normal":
                prompts[cls] = "a normal scene without any anomaly"
            else:
                prompts[cls] = f"a video with {cls.lower()} event"
        return prompts

    @staticmethod
    def save_prompts(prompts, output_path):
        """Save initial prompts to JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(prompts, f, indent=2)
        print(f"✓ Saved initial prompts to {output_path}")

    @staticmethod
    def load_prompts(input_path):
        """Load initial prompts from JSON file."""
        with open(input_path, 'r') as f:
            prompts = json.load(f)
        print(f"✓ Loaded initial prompts from {input_path}")
        return prompts


def extract_learned_prompts(model, classnames, tokenizer):
    """Extract learned prompt vectors from trained model.

    Args:
        model: VideoFeatureCLIP model with trained PromptLearner
        classnames: List of class names
        tokenizer: Tokenizer for text encoding

    Returns:
        dict: {classname: {
            'context_vector': [...],
            'prompt_string': "...",
            'embedding': [...]
        }}
    """
    prompt_learner = model.prompt_learner

    # Get learned context vectors
    ctx = prompt_learner.ctx  # [n_cls, n_ctx, ctx_dim]

    # Get tokenized prompts
    tokenized_prompts = prompt_learner.tokenized_prompts  # [n_cls, seq_len]

    # Get token embeddings
    with torch.no_grad():
        # Get the token prefix and suffix
        token_prefix = prompt_learner.token_prefix
        token_suffix = prompt_learner.token_suffix

        # Full prompt embeddings
        full_prompts = torch.cat([token_prefix, ctx, token_suffix], dim=1)

    results = {}
    for i, classname in enumerate(classnames):
        results[classname] = {
            'context_vector': ctx[i].cpu().numpy().tolist(),
            'context_shape': list(ctx[i].shape),
            'full_prompt_embedding': full_prompts[i].cpu().detach().numpy().tolist(),
            'full_prompt_shape': list(full_prompts[i].shape),
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train CoOp with custom class-specific initial prompts"
    )

    # Dataset arguments
    parser.add_argument("--feature-dir", type=str, 
                        # required=True,
                        default="/mnt/c/JJS/UCF_Crimes/Features/MCi20-avgpooled/train",
                        help="Path to training feature directory")
    parser.add_argument("--val-feature-dir", type=str, 
                        # required=True,
                        default="/mnt/c/JJS/UCF_Crimes/Features/MCi20-avgpooled/valid",
                        help="Path to validation feature directory")

    # Custom prompt arguments
    parser.add_argument("--initial-prompts-file", type=str, default=None,
                        help="Path to JSON file with custom initial prompts per class")
    parser.add_argument("--save-initial-prompts", type=str, default=None,
                        help="Save generated initial prompts to this path")
    parser.add_argument("--custom-prompts", type=str, nargs='+', default=None,
                        help="Custom prompts as: class1 'prompt1' class2 'prompt2' ...")

    # Model arguments
    parser.add_argument("--mobileclip-model", type=str, default="mobileclip2_s0")
    parser.add_argument("--mobileclip-path", type=str, default=None)
    parser.add_argument("--n-ctx", type=int, default=16)
    parser.add_argument("--csc", action="store_true", default=True)

    # Training arguments
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--num-workers", type=int, default=4)

    # Output arguments
    parser.add_argument("--output-dir", type=str, default="./output/custom_prompts_0224_0914")
    parser.add_argument("--save-prompts", action="store_true", default=True,
                        help="Save learned prompts after training")

    args = parser.parse_args()

    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("Custom Prompt Training for CoOp")
    print("=" * 80)

    # Load dataset
    print("\nLoading training dataset...")
    train_dataset = VideoFeatureDataset(
        feature_dir=args.feature_dir,
        normal_class="Normal",
        unit_duration=1,
        overlap_ratio=0.0,
        strict_normal_sampling=True,
        use_video_level_pooling=False,
        verbose=True,
        seed=42,
    )

    print("\nLoading validation dataset...")
    val_dataset = VideoFeatureDataset(
        feature_dir=args.val_feature_dir,
        normal_class="Normal",
        unit_duration=1,
        overlap_ratio=0.0,
        strict_normal_sampling=False,
        use_video_level_pooling=False,
        verbose=True,
        seed=42,
    )

    classnames = train_dataset.classnames
    num_classes = len(classnames)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Prepare initial prompts
    print("\n" + "=" * 80)
    print("Initial Prompts Setup")
    print("=" * 80)

    if args.initial_prompts_file and os.path.exists(args.initial_prompts_file):
        # Load from file
        initial_prompts = PromptGenerator.load_prompts(args.initial_prompts_file)
    elif args.custom_prompts:
        # Parse command line prompts
        initial_prompts = {}
        for i in range(0, len(args.custom_prompts), 2):
            if i + 1 < len(args.custom_prompts):
                classname = args.custom_prompts[i]
                prompt = args.custom_prompts[i + 1]
                initial_prompts[classname] = prompt
    else:
        # Generate default prompts
        initial_prompts = PromptGenerator.create_default_prompts(classnames)

    # Validate prompts cover all classes
    missing_classes = [c for c in classnames if c not in initial_prompts]
    if missing_classes:
        print(f"⚠ Warning: Missing prompts for classes: {missing_classes}")
        print("  Adding default prompts...")
        for cls in missing_classes:
            initial_prompts[cls] = f"a video with {cls.lower()}"

    # Save initial prompts
    prompts_path = args.save_initial_prompts or os.path.join(
        args.output_dir, "initial_prompts.json"
    )
    PromptGenerator.save_prompts(initial_prompts, prompts_path)

    print("\nInitial prompts per class:")
    for cls in classnames:
        prompt = initial_prompts.get(cls, "N/A")
        print(f"  {cls:15s}: {prompt}")

    # Load model with custom prompts
    print("\n" + "=" * 80)
    print("Loading Model with Custom Initial Prompts")
    print("=" * 80)

    clip_model, tokenizer = load_mobileclip(
        pretrained_path=args.mobileclip_path,
        model_name=args.mobileclip_model,
        device="cuda"
    )

    # Create model with class-specific custom prompts
    print("\nCreating VideoFeatureCLIP with class-specific custom prompts...")

    print(f"\nInitial prompts per class:")
    for cls in classnames:
        prompt = initial_prompts.get(cls, f"a video with {cls.lower()}")
        print(f"  {cls:15s}: {prompt}")

    model = VideoFeatureCLIP(
        classnames=classnames,
        clip_model=clip_model,
        tokenizer=tokenizer,
        n_ctx=args.n_ctx,
        class_prompts=initial_prompts,  # Pass class-specific prompts for initialization
        csc=args.csc,
        class_token_position="end",
        temporal_agg="mean",
    )

    print(f"\n✓ PromptLearner initialized with class-specific prompts")

    # Freeze everything except prompt_learner
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / Total: {total_params:,}")

    model = model.to(device)

    # Training setup
    optimizer = optim.SGD(
        model.prompt_learner.parameters(),
        lr=args.lr,
        weight_decay=5e-4,
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * 1

    warmup_scheduler = LinearLR(
        optimizer, start_factor=1e-5 / args.lr, total_iters=warmup_steps,
    )
    main_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps,
    )

    # Training loop
    print("\n" + "=" * 80)
    print("Training")
    print("=" * 80)

    best_acc = 0.0
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device,
            scheduler=warmup_scheduler if epoch == 0 else main_scheduler,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
        )

        val_loss, val_acc, per_class = validate(
            model, val_loader, device, classnames
        )

        print(
            f"Epoch {epoch + 1}: "
            f"Train Loss={train_loss:.4f}, Acc={train_acc:.1f}% | "
            f"Val Loss={val_loss:.4f}, Acc={val_acc:.1f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(checkpoint_dir, "best_custom_prompt_model.pth")
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, save_path)

    # Extract and save learned prompts
    print("\n" + "=" * 80)
    print("Extracting Learned Prompts")
    print("=" * 80)

    learned_prompts = extract_learned_prompts(model, classnames, tokenizer)

    learned_prompts_path = os.path.join(args.output_dir, "learned_prompts_from_custom.json")
    with open(learned_prompts_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json.dump(learned_prompts, f, indent=2)

    print(f"✓ Saved learned prompts to {learned_prompts_path}")

    # Summary
    print("\n" + "=" * 80)
    print("Training Complete")
    print("=" * 80)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - Checkpoint: {os.path.join(checkpoint_dir, 'best_custom_prompt_model.pth')}")
    print(f"  - Initial prompts: {prompts_path}")
    print(f"  - Learned prompts: {learned_prompts_path}")

    # Print summary of learned context vectors
    print(f"\nLearned Context Vectors Summary:")
    print(f"{'Class':15s} {'Context Shape':20s} {'Embedding Shape':20s}")
    print("-" * 55)
    for cls in classnames:
        ctx_shape = learned_prompts[cls]['context_shape']
        emb_shape = learned_prompts[cls]['full_prompt_shape']
        print(f"{cls:15s} {str(ctx_shape):20s} {str(emb_shape):20s}")


if __name__ == "__main__":
    main()
