"""Debug how prompts are actually constructed in PromptLearner."""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from trainers.video_feature_coop import load_mobileclip, PromptLearner

print("=" * 80)
print("Prompt Construction Debug")
print("=" * 80)

# Load MobileCLIP
clip_model, tokenizer = load_mobileclip(model_name="mobileclip2_s0", device="cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = clip_model.to(device)

classnames = ["Abuse", "Normal", "Fighting"]

print(f"\nClasses: {classnames}")
print(f"Device: {device}")

# Create PromptLearner
prompt_learner = PromptLearner(
    classnames=classnames,
    clip_model=clip_model,
    tokenizer=tokenizer,
    n_ctx=4,
    ctx_init="",
    csc=True,
    class_token_position="end",
)
prompt_learner = prompt_learner.to(device)

print("\n" + "=" * 80)
print("Inspecting Components")
print("=" * 80)

print("\n1. Context vectors (self.ctx):")
ctx = prompt_learner.ctx
print(f"   Shape: {ctx.shape}")
print(f"   requires_grad: {ctx.requires_grad}")
print(f"   Device: {ctx.device}")
print(f"\n   ctx[0, 0, :10]:")
print(f"   {ctx[0, 0, :10]}")
print(f"   ctx[1, 0, :10]:")
print(f"   {ctx[1, 0, :10]}")
print(f"   ctx[2, 0, :10]:")
print(f"   {ctx[2, 0, :10]}")

print("\n2. Token prefix (self.token_prefix):")
prefix = prompt_learner.token_prefix
print(f"   Shape: {prefix.shape}")
print(f"   Device: {prefix.device}")
print(f"\n   prefix[0, 0, :10]:")
print(f"   {prefix[0, 0, :10]}")
print(f"   prefix[1, 0, :10]:")
print(f"   {prefix[1, 0, :10]}")
print(f"   prefix[2, 0, :10]:")
print(f"   {prefix[2, 0, :10]}")

# Check if all prefixes are identical
prefix_identical = torch.allclose(prefix[0], prefix[1]) and torch.allclose(prefix[1], prefix[2])
print(f"\n   All prefixes identical: {prefix_identical}")

print("\n3. Token suffix (self.token_suffix):")
suffix = prompt_learner.token_suffix
print(f"   Shape: {suffix.shape}")
print(f"   Device: {suffix.device}")
print(f"\n   suffix[0, 0, :10]:")
print(f"   {suffix[0, 0, :10]}")
print(f"   suffix[1, 0, :10]:")
print(f"   {suffix[1, 0, :10]}")
print(f"   suffix[2, 0, :10]:")
print(f"   {suffix[2, 0, :10]}")

# Check if suffixes are different (they should be)
suffix_01_diff = (suffix[0] - suffix[1]).abs().sum()
suffix_12_diff = (suffix[1] - suffix[2]).abs().sum()
print(f"\n   Suffix differences (sum of abs):")
print(f"      |suffix[0] - suffix[1]|: {suffix_01_diff:.6f}")
print(f"      |suffix[1] - suffix[2]|: {suffix_12_diff:.6f}")

print("\n4. class_token_position:")
print(f"   {prompt_learner.class_token_position}")

print("\n" + "=" * 80)
print("Constructing Prompts (forward pass)")
print("=" * 80)

print("\n5. Executing prompt_learner.forward()...")

# Manually reproduce the forward pass
ctx = prompt_learner.ctx
print(f"\n   ctx shape: {ctx.shape}")
print(f"   ctx.dim(): {ctx.dim()}")

if ctx.dim() == 2:
    print("   Expanding ctx from 2D to 3D")
    ctx = ctx.unsqueeze(0).expand(prompt_learner.n_cls, -1, -1)
    print(f"   After expand: {ctx.shape}")
else:
    print("   ctx is already 3D (CSC mode)")

prefix = prompt_learner.token_prefix
suffix = prompt_learner.token_suffix

print(f"\n6. Concatenating [prefix, ctx, suffix]:")
print(f"   prefix shape: {prefix.shape}")
print(f"   ctx shape: {ctx.shape}")
print(f"   suffix shape: {suffix.shape}")

if prompt_learner.class_token_position == "end":
    prompts = torch.cat([prefix, ctx, suffix], dim=1)
    print(f"   Prompts shape: {prompts.shape}")

    print(f"\n7. Inspecting concatenated prompts:")
    print(f"   prompts[0, 0, :10] (prefix part):")
    print(f"   {prompts[0, 0, :10]}")
    print(f"   prompts[0, 1, :10] (ctx part):")
    print(f"   {prompts[0, 1, :10]}")
    print(f"   prompts[0, 5, :10] (suffix part):")
    print(f"   {prompts[0, 5, :10]}")

    print(f"\n   Checking each class:")
    for i in range(3):
        print(f"\n   Class {i} ({classnames[i]}):")
        print(f"      Position 0 (prefix): {prompts[i, 0, :5]}")
        print(f"      Position 1 (ctx):    {prompts[i, 1, :5]}")
        print(f"      Position 5 (suffix): {prompts[i, 5, :5]}")

print(f"\n8. Prompt differences:")
diff_01 = (prompts[0] - prompts[1]).abs().mean()
diff_12 = (prompts[1] - prompts[2]).abs().mean()
diff_02 = (prompts[0] - prompts[2]).abs().mean()
print(f"   |prompts[0] - prompts[1]| mean: {diff_01:.6f}")
print(f"   |prompts[1] - prompts[2]| mean: {diff_12:.6f}")
print(f"   |prompts[0] - prompts[2]| mean: {diff_02:.6f}")

# Breakdown by position
print(f"\n9. Difference breakdown by position:")
for pos in [0, 1, 2, 3, 4, 5, 10, 20]:
    if pos < prompts.shape[1]:
        diff_01_pos = (prompts[0, pos] - prompts[1, pos]).abs().mean()
        diff_12_pos = (prompts[1, pos] - prompts[2, pos]).abs().mean()
        print(f"   Position {pos:2d}: diff(0,1)={diff_01_pos:.6f}, diff(1,2)={diff_12_pos:.6f}")

print(f"\n10. Using actual forward() method:")
prompts_forward = prompt_learner()
print(f"   Shape: {prompts_forward.shape}")
print(f"   Device: {prompts_forward.device}")

diff_manual_vs_forward = (prompts - prompts_forward).abs().mean()
print(f"   |manual - forward| mean: {diff_manual_vs_forward:.6f}")

if diff_manual_vs_forward < 1e-6:
    print("   ✓ Manual construction matches forward() method")
else:
    print("   ❌ Mismatch between manual and forward()!")

print("\n" + "=" * 80)
print("Debug completed")
print("=" * 80)
