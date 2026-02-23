"""
Debug script to verify if custom prompts are being properly applied
in custom_prompt_training.py
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from trainers.video_feature_coop import load_mobileclip, VideoFeatureCLIP, PromptLearner


def debug_ctx_init():
    """Debug how ctx_init is being processed."""

    # Simulate what custom_prompt_training.py does
    classnames = ["Normal", "Abuse", "Arrest", "Arson"]
    initial_prompts = {
        "Normal": "a normal video without any anomaly",
        "Abuse": "a video showing physical abuse",
        "Arrest": "a video of police making arrest",
        "Arson": "a video showing fire",
    }

    print("\n" + "="*80)
    print("Debug: Custom Prompt Initialization")
    print("="*80)

    # Current approach (WRONG)
    print("\n❌ CURRENT (WRONG) APPROACH:")
    ctx_init_str = " ".join(initial_prompts.get(cls, f"{cls}") for cls in classnames)
    print(f"ctx_init_str: '{ctx_init_str}'")
    print(f"Length: {len(ctx_init_str)} chars")
    print(f"Token count (approximate): {len(ctx_init_str.split())} tokens")
    print("\nProblem:")
    print("  - All class prompts are concatenated into one string")
    print("  - PromptLearner will use this as a shared context for ALL classes")
    print("  - This defeats the purpose of class-specific initial prompts!")

    # Load model and check actual behavior
    print("\n" + "-"*80)
    print("Checking actual PromptLearner behavior:")
    print("-"*80)

    clip_model, tokenizer = load_mobileclip(model_name="mc2_s0", device="cpu")

    # Create model with concatenated ctx_init (current approach)
    print(f"\nCreating VideoFeatureCLIP with ctx_init_str (length={len(ctx_init_str)})")
    model = VideoFeatureCLIP(
        classnames=classnames,
        clip_model=clip_model,
        tokenizer=tokenizer,
        n_ctx=16,
        ctx_init=ctx_init_str,
        csc=True,
        class_token_position="end",
        temporal_agg="mean",
    )

    # Check what prompts were generated
    print(f"\nPromptLearner details:")
    pl = model.prompt_learner
    print(f"  - n_ctx: {pl.n_ctx}")
    print(f"  - n_cls: {pl.n_cls}")
    print(f"  - csc: {pl.csc}")
    print(f"  - ctx shape: {pl.ctx.shape}")

    # Generate prompts and check
    with torch.no_grad():
        prompts = pl()

    print(f"  - Generated prompts shape: {prompts.shape}")

    # Check token prefix and suffix
    print(f"\nToken prefix shape: {pl.token_prefix.shape}")
    print(f"Token suffix shape: {pl.token_suffix.shape}")

    # Now test with text encoder
    print("\n" + "-"*80)
    print("Testing text feature generation:")
    print("-"*80)

    with torch.no_grad():
        prompts = pl()
        tokenized_prompts = pl.tokenized_prompts
        text_features = model.text_encoder(prompts, tokenized_prompts)

    print(f"Text features shape: {text_features.shape}")

    # Check if text features are different for different classes
    print("\nText feature differences between classes:")
    for i in range(len(classnames)):
        for j in range(i+1, len(classnames)):
            diff = (text_features[i] - text_features[j]).norm().item()
            print(f"  {classnames[i]:15s} vs {classnames[j]:15s}: {diff:.6f}")

    # Check if all features are nearly identical (bad sign)
    print(f"\nFeature statistics:")
    print(f"  - Mean norm: {text_features.norm(dim=-1).mean():.6f}")
    print(f"  - Std norm: {text_features.norm(dim=-1).std():.6f}")
    print(f"  - Min pairwise diff: {min((text_features[i] - text_features[j]).norm().item() for i in range(len(classnames)) for j in range(i+1, len(classnames))):.6f}")
    print(f"  - Max pairwise diff: {max((text_features[i] - text_features[j]).norm().item() for i in range(len(classnames)) for j in range(i+1, len(classnames))):.6f}")

    # The problem
    print("\n" + "="*80)
    print("DIAGNOSIS:")
    print("="*80)
    print("""
If the max pairwise difference is very small (< 0.01), then:
  ✗ Custom prompts are NOT being applied correctly
  ✗ All classes are getting similar text features
  ✗ This explains why loss is fixed and accuracy is ~1%

The issue is:
  1. ctx_init_str concatenates ALL class prompts into one long string
  2. PromptLearner uses this single string as SHARED context for all classes
  3. So all classes start with almost identical prompts
  4. The only difference is the class name at the end
  5. With mean pooling over tokens, differences are too small

This is WRONG because:
  - We want EACH CLASS to have ITS OWN initial prompt
  - Not all classes sharing one combined prompt string
""")


if __name__ == "__main__":
    debug_ctx_init()
