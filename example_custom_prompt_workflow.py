"""
Example workflow for custom prompt training with CoOp.

This script demonstrates:
1. Preparing custom prompts
2. Training with custom prompts
3. Comparing initial vs learned prompts
"""

import json
import subprocess
import argparse
from pathlib import Path
import numpy as np
from tabulate import tabulate


def prepare_custom_prompts(output_path="./custom_prompts.json"):
    """Prepare custom prompts JSON file."""

    custom_prompts = {
        "Normal": "a normal video without any anomaly, crime or unusual activity",
        "Abuse": "a video showing physical abuse, hitting, punching or violence toward a person",
        "Arrest": "a video of police officers arresting or apprehending a person",
        "Arson": "a video showing fire, flames or a building burning",
        "Assault": "a video of people attacking each other with weapons or physical force",
        "Burglary": "a video of someone breaking into a building, stealing or ransacking property",
        "Explosion": "a video showing an explosion, bomb detonation or blast",
        "Fighting": "a video of multiple people engaged in violent physical combat",
        "RoadAccidents": "a video of a car crash, traffic accident or vehicle collision",
        "Robbery": "a video of armed robbery, mugging or violent theft",
        "Shooting": "a video of a person shooting a gun or gunshot being fired",
        "Shoplifting": "a video of someone stealing items from a store or retail location",
        "Stealing": "a video of property theft, pickpocketing or stealing",
        "Vandalism": "a video showing property damage, graffiti or destruction of property"
    }

    with open(output_path, 'w') as f:
        json.dump(custom_prompts, f, indent=2)

    print(f"✓ Custom prompts saved to {output_path}")
    print("\nInitial prompts preview:")
    for cls, prompt in list(custom_prompts.items())[:3]:
        print(f"  {cls:15s}: {prompt}")
    print(f"  ... ({len(custom_prompts)} classes total)")

    return output_path


def train_with_custom_prompts(
    feature_dir,
    val_feature_dir,
    custom_prompts_path,
    output_dir="./output/custom_prompts",
    epochs=50,
    batch_size=32,
    n_ctx=16,
):
    """Train CoOp model with custom prompts."""

    cmd = [
        "python", "custom_prompt_training.py",
        "--feature-dir", str(feature_dir),
        "--val-feature-dir", str(val_feature_dir),
        "--initial-prompts-file", str(custom_prompts_path),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--n-ctx", str(n_ctx),
        "--csc",
        "--output-dir", str(output_dir),
        "--save-prompts",
    ]

    print("\n" + "="*80)
    print("Training Custom Prompt Model")
    print("="*80)
    print("\nCommand:")
    print(" ".join(cmd))
    print("\n")

    result = subprocess.run(cmd, cwd=Path.cwd())

    if result.returncode == 0:
        print("\n✓ Training completed successfully")
        return output_dir
    else:
        print("\n✗ Training failed")
        return None


def analyze_prompts(
    initial_prompts_path,
    learned_prompts_path,
):
    """Analyze and compare initial vs learned prompts."""

    print("\n" + "="*80)
    print("Prompt Analysis")
    print("="*80)

    # Load prompts
    with open(initial_prompts_path) as f:
        initial_prompts = json.load(f)

    with open(learned_prompts_path) as f:
        learned_prompts = json.load(f)

    # Prepare analysis data
    table_data = []

    for classname in sorted(initial_prompts.keys()):
        initial_text = initial_prompts[classname]

        learned_data = learned_prompts[classname]
        ctx_shape = learned_data['context_shape']
        full_shape = learned_data['full_prompt_shape']

        # Calculate context vector statistics
        ctx_vector = np.array(learned_data['context_vector'])
        ctx_norm = np.linalg.norm(ctx_vector)
        ctx_mean = np.linalg.norm(ctx_vector, axis=1).mean()

        table_data.append([
            classname,
            len(initial_text),
            f"{ctx_shape[0]}x{ctx_shape[1]}",
            f"{full_shape[0]}x{full_shape[1]}",
            f"{ctx_norm:.4f}",
            f"{ctx_mean:.4f}",
        ])

    print("\nInitial vs Learned Prompts Statistics:")
    print(tabulate(
        table_data,
        headers=[
            "Class",
            "Init Len",
            "Ctx Shape",
            "Full Shape",
            "Ctx Norm",
            "Ctx Token Mean",
        ],
        tablefmt="grid"
    ))

    # Detailed class-specific analysis
    print("\n" + "-"*80)
    print("Detailed Analysis:")
    print("-"*80)

    for classname in sorted(initial_prompts.keys())[:3]:  # Show first 3 classes
        print(f"\n{classname}:")
        print(f"  Initial prompt: \"{initial_prompts[classname]}\"")

        learned_data = learned_prompts[classname]
        ctx_vector = np.array(learned_data['context_vector'])

        print(f"  Learned context shape: {learned_data['context_shape']}")
        print(f"  Context vector statistics:")
        print(f"    - L2 norm: {np.linalg.norm(ctx_vector):.6f}")
        print(f"    - Mean token norm: {np.linalg.norm(ctx_vector, axis=1).mean():.6f}")
        print(f"    - Min token norm: {np.linalg.norm(ctx_vector, axis=1).min():.6f}")
        print(f"    - Max token norm: {np.linalg.norm(ctx_vector, axis=1).max():.6f}")
        print(f"    - Std token norm: {np.linalg.norm(ctx_vector, axis=1).std():.6f}")

    print(f"\n... (showing 3 out of {len(initial_prompts)} classes)")


def compare_with_baseline(
    baseline_dir="./output/baseline",
    custom_dir="./output/custom_prompts",
):
    """Compare custom prompts with baseline default prompts."""

    print("\n" + "="*80)
    print("Baseline vs Custom Comparison")
    print("="*80)

    baseline_learned = Path(baseline_dir) / "learned_prompts.json"
    custom_learned = Path(custom_dir) / "learned_prompts.json"

    if not baseline_learned.exists() or not custom_learned.exists():
        print("⚠ Both baseline and custom training outputs required for comparison")
        print(f"  Baseline: {baseline_learned}")
        print(f"  Custom: {custom_learned}")
        return

    with open(baseline_learned) as f:
        baseline = json.load(f)

    with open(custom_learned) as f:
        custom = json.load(f)

    table_data = []

    for classname in sorted(baseline.keys()):
        baseline_ctx = np.array(baseline[classname]['context_vector'])
        custom_ctx = np.array(custom[classname]['context_vector'])

        baseline_norm = np.linalg.norm(baseline_ctx)
        custom_norm = np.linalg.norm(custom_ctx)

        # Calculate cosine similarity
        baseline_flat = baseline_ctx.flatten()
        custom_flat = custom_ctx.flatten()
        similarity = np.dot(baseline_flat, custom_flat) / (
            np.linalg.norm(baseline_flat) * np.linalg.norm(custom_flat)
        )

        table_data.append([
            classname,
            f"{baseline_norm:.4f}",
            f"{custom_norm:.4f}",
            f"{custom_norm - baseline_norm:+.4f}",
            f"{similarity:.4f}",
        ])

    print("\nContext Vector Comparison:")
    print(tabulate(
        table_data,
        headers=[
            "Class",
            "Baseline Norm",
            "Custom Norm",
            "Norm Diff",
            "Cosine Sim",
        ],
        tablefmt="grid"
    ))


def main():
    parser = argparse.ArgumentParser(
        description="Example workflow for custom prompt training"
    )
    parser.add_argument(
        "--feature-dir",
        type=str,
        required=True,
        help="Path to training feature directory"
    )
    parser.add_argument(
        "--val-feature-dir",
        type=str,
        required=True,
        help="Path to validation feature directory"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=16,
        help="Number of context tokens"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/custom_prompts",
        help="Output directory"
    )
    parser.add_argument(
        "--custom-prompts",
        type=str,
        default="./custom_prompts.json",
        help="Custom prompts JSON file"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and only analyze"
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare with baseline training"
    )

    args = parser.parse_args()

    # Step 1: Prepare custom prompts
    print("\n" + "="*80)
    print("Step 1: Preparing Custom Prompts")
    print("="*80)

    prompts_path = Path(args.custom_prompts)
    if not prompts_path.exists():
        prompts_path = prepare_custom_prompts(str(prompts_path))
    else:
        print(f"✓ Using existing custom prompts: {prompts_path}")

    # Step 2: Train with custom prompts
    if not args.skip_training:
        print("\n" + "="*80)
        print("Step 2: Training Model with Custom Prompts")
        print("="*80)

        output_dir = train_with_custom_prompts(
            args.feature_dir,
            args.val_feature_dir,
            prompts_path,
            args.output_dir,
            args.epochs,
            args.batch_size,
            args.n_ctx,
        )

        if output_dir is None:
            print("Training failed. Exiting.")
            return
    else:
        output_dir = args.output_dir

    # Step 3: Analyze prompts
    print("\n" + "="*80)
    print("Step 3: Analyzing Learned Prompts")
    print("="*80)

    initial_path = Path(output_dir) / "initial_prompts.json"
    learned_path = Path(output_dir) / "learned_prompts.json"

    if initial_path.exists() and learned_path.exists():
        analyze_prompts(str(initial_path), str(learned_path))
    else:
        print(f"⚠ Prompt files not found:")
        print(f"  Initial: {initial_path}")
        print(f"  Learned: {learned_path}")

    # Step 4: Compare with baseline (optional)
    if args.compare_baseline:
        print("\n" + "="*80)
        print("Step 4: Comparing with Baseline")
        print("="*80)

        compare_with_baseline()

    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"\n✓ Custom prompts training completed!")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - initial_prompts.json: Input prompts")
    print(f"  - learned_prompts.json: Trained prompt vectors")
    print(f"  - checkpoints/best_model.pth: Best checkpoint")

    print("\nNext steps:")
    print("1. Review learned_prompts.json for prompt analysis")
    print("2. Evaluate model on test set:")
    print(f"   python evaluate_video_feature_coop.py \\")
    print(f"       --test-feature-dir /path/to/test \\")
    print(f"       --checkpoint-path {output_dir}/checkpoints/best_model.pth \\")
    print(f"       --n-ctx {args.n_ctx} \\")
    print(f"       --csc \\")
    print(f"       --output-dir ./output/eval_custom_prompts")
    print("3. Compare results with baseline model")


if __name__ == "__main__":
    main()
