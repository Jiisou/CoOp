#!/usr/bin/env python3
"""
Verification script for CoOp video feature prompt learning setup.

This script creates dummy data and tests the complete pipeline to verify:
1. Dataset loading works correctly
2. Model initialization succeeds
3. Forward pass runs without errors
4. Training loop executes properly

Usage:
    python scripts/video_feature_coop/verify_setup.py --mobileclip-path /path/to/mobileclip_s0.pt
"""

import argparse
import os
import sys
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

# Add CoOp root directory to Python path
script_dir = Path(__file__).resolve().parent
coop_root = script_dir.parent.parent  # Go up two levels: scripts/video_feature_coop -> scripts -> CoOp
sys.path.insert(0, str(coop_root))

from datasets.video_features import VideoFeatureDataset
from trainers.video_feature_coop import load_mobileclip, VideoFeatureCLIP


def create_dummy_data(base_dir, num_classes=3, videos_per_class=5, frames_per_video=10, feature_dim=512):
    """Create dummy .npy feature files for testing."""
    print(f"\nCreating dummy data in {base_dir}")

    class_names = [f"action_{i+1}" for i in range(num_classes)]

    for split in ["train", "val"]:
        split_dir = os.path.join(base_dir, split)

        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            for video_idx in range(videos_per_class):
                # Create random features [T, D]
                features = np.random.randn(frames_per_video, feature_dim).astype(np.float32)

                # Save as .npy
                filename = f"{class_name}_{video_idx:03d}_x264.npy"
                filepath = os.path.join(class_dir, filename)
                np.save(filepath, features)

        print(f"  Created {split} split: {num_classes} classes × {videos_per_class} videos × {frames_per_video} frames")

    return class_names


def test_dataset(feature_dir, unit_duration=2):
    """Test VideoFeatureDataset loading."""
    print("\n" + "="*60)
    print("TEST 1: Dataset Loading")
    print("="*60)

    try:
        dataset = VideoFeatureDataset(
            feature_dir=os.path.join(feature_dir, "train"),
            annotation_dir=None,
            unit_duration=unit_duration,
            overlap_ratio=0.0,
            strict_normal_sampling=False,
            verbose=True,
        )

        print(f"✓ Dataset loaded successfully")
        print(f"  - Total samples: {len(dataset)}")
        print(f"  - Classes: {dataset.classnames}")

        # Test __getitem__
        features, label = dataset[0]
        print(f"  - Sample shape: {features.shape}, label: {label}")

        if features.shape[0] == unit_duration:
            print(f"✓ Sample shape is correct: [{unit_duration}, D]")
        else:
            print(f"✗ Sample shape mismatch: expected [{unit_duration}, D], got {features.shape}")
            return False

        return True

    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_init(mobileclip_path, model_name, classnames, device="cpu"):
    """Test model initialization."""
    print("\n" + "="*60)
    print("TEST 2: Model Initialization")
    print("="*60)

    try:
        if mobileclip_path:
            print(f"Loading MobileCLIP ({model_name}) from: {mobileclip_path}")
        else:
            print(f"Auto-loading MobileCLIP ({model_name})...")

        clip_model, tokenizer = load_mobileclip(
            pretrained_path=mobileclip_path,
            model_name=model_name,
            device=device
        )
        print(f"✓ MobileCLIP loaded successfully")

        print("Initializing VideoFeatureCLIP...")
        model = VideoFeatureCLIP(
            classnames=classnames,
            clip_model=clip_model,
            tokenizer=tokenizer,
            n_ctx=4,
            ctx_init="",
            csc=False,
            class_token_position="end",
            temporal_agg="mean",
        )
        model = model.to(device)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"✓ Model initialized successfully")
        print(f"  - Trainable params: {trainable:,} / Total: {total:,}")

        return model

    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_forward_pass(model, batch_size=4, seq_len=2, feature_dim=512, num_classes=3, device="cpu"):
    """Test forward pass."""
    print("\n" + "="*60)
    print("TEST 3: Forward Pass")
    print("="*60)

    try:
        model.eval()

        # Create dummy input [B, T, D]
        dummy_input = torch.randn(batch_size, seq_len, feature_dim).to(device)
        print(f"Input shape: {dummy_input.shape}")

        with torch.no_grad():
            logits = model(dummy_input)

        print(f"Output shape: {logits.shape}")

        if logits.shape == (batch_size, num_classes):
            print(f"✓ Forward pass successful")
            print(f"  - Logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
            return True
        else:
            print(f"✗ Output shape mismatch: expected [{batch_size}, {num_classes}], got {logits.shape}")
            return False

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_pass(model, batch_size=4, seq_len=2, feature_dim=512, num_classes=3, device="cpu"):
    """Test backward pass and optimizer."""
    print("\n" + "="*60)
    print("TEST 4: Backward Pass & Optimization")
    print("="*60)

    try:
        model.train()

        # Setup optimizer
        optimizer = torch.optim.SGD(model.prompt_learner.parameters(), lr=0.002)

        # Create dummy batch
        dummy_input = torch.randn(batch_size, seq_len, feature_dim).to(device)
        dummy_labels = torch.randint(0, num_classes, (batch_size,)).to(device)

        # Forward
        logits = model(dummy_input)
        loss = torch.nn.functional.cross_entropy(logits, dummy_labels)

        print(f"Loss: {loss.item():.4f}")

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"✓ Backward pass and optimization successful")

        # Check gradients
        has_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                print(f"  - {name}: grad norm = {param.grad.norm().item():.4f}")

        if has_grad:
            print(f"✓ Gradients computed correctly")
            return True
        else:
            print(f"✗ No gradients found")
            return False

    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify CoOp video feature setup")
    parser.add_argument("--mobileclip-path", type=str, default=None,
                        help="Path to MobileCLIP weights (.pt file). If not provided, auto-downloads.")
    parser.add_argument("--mobileclip-model", type=str, default="mobileclip2_s0",
                        choices=["mobileclip2_s0", "mobileclip_s0"],
                        help="MobileCLIP model variant")
    parser.add_argument("--keep-dummy-data", action="store_true",
                        help="Keep dummy data directory after testing")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device to use for testing")
    args = parser.parse_args()

    print("="*60)
    print("CoOp Video Feature Setup Verification")
    print("="*60)
    print(f"MobileCLIP model: {args.mobileclip_model}")
    print(f"MobileCLIP path: {args.mobileclip_path if args.mobileclip_path else 'auto-download'}")
    print(f"Device: {args.device}")

    # Check MobileCLIP file exists if path is provided
    if args.mobileclip_path and not os.path.exists(args.mobileclip_path):
        print(f"\n✗ MobileCLIP weights not found at: {args.mobileclip_path}")
        print("Will attempt to auto-download instead...")
        args.mobileclip_path = None

    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="coop_verify_")
    print(f"\nTemporary directory: {temp_dir}")

    try:
        # Test 1: Create dummy data and test dataset
        classnames = create_dummy_data(
            temp_dir,
            num_classes=3,
            videos_per_class=5,
            frames_per_video=10,
            feature_dim=512,
        )

        if not test_dataset(temp_dir, unit_duration=2):
            print("\n✗ Dataset test failed. Stopping.")
            return

        # Test 2: Model initialization
        model = test_model_init(
            args.mobileclip_path,
            args.mobileclip_model,
            classnames,
            device=args.device
        )
        if model is None:
            print("\n✗ Model initialization test failed. Stopping.")
            return

        # Test 3: Forward pass
        if not test_forward_pass(model, batch_size=4, seq_len=2, feature_dim=512,
                                 num_classes=len(classnames), device=args.device):
            print("\n✗ Forward pass test failed. Stopping.")
            return

        # Test 4: Backward pass
        if not test_backward_pass(model, batch_size=4, seq_len=2, feature_dim=512,
                                  num_classes=len(classnames), device=args.device):
            print("\n✗ Backward pass test failed. Stopping.")
            return

        # All tests passed
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nYour setup is ready for training.")
        print("\nNext steps:")
        print("1. Prepare your .npy feature files in the required directory structure")
        print("2. (Optional) Prepare annotation CSV files for strict normal filtering")
        print("3. Run training:")
        print("   python train_video_feature_coop.py \\")
        print("       --feature-dir /path/to/features/train \\")
        print("       --val-feature-dir /path/to/features/val \\")
        if args.mobileclip_path:
            print(f"       --mobileclip-path {args.mobileclip_path} \\")
        print(f"       --mobileclip-model {args.mobileclip_model} \\")
        print("       --epochs 50")

    finally:
        if args.keep_dummy_data:
            print(f"\nDummy data kept at: {temp_dir}")
        else:
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up temporary directory")


if __name__ == "__main__":
    main()
