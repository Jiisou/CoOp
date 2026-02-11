"""Debug script to inspect text feature generation in detail."""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from trainers.video_feature_coop import load_mobileclip, VideoFeatureCLIP

print("=" * 80)
print("Text Features Debug Script")
print("=" * 80)

# Load MobileCLIP
clip_model, tokenizer = load_mobileclip(model_name="mobileclip2_s0", device="cpu")

# Create model with CSC
classnames = ["Abuse", "Normal", "Fighting"]
print(f"\nClasses: {classnames}")

model = VideoFeatureCLIP(
    classnames=classnames,
    clip_model=clip_model,
    tokenizer=tokenizer,
    n_ctx=4,
    ctx_init="",
    csc=True,
    class_token_position="end",
    temporal_agg="mean",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"\nDevice: {device}")
print("\n" + "=" * 80)
print("Inspecting Prompt Generation")
print("=" * 80)

# Get prompt components
print("\n1. Context vectors (ctx):")
ctx = model.prompt_learner.ctx
print(f"   Shape: {ctx.shape}")
print(f"   Device: {ctx.device}")
print(f"   Stats: min={ctx.min():.4f}, max={ctx.max():.4f}, mean={ctx.mean():.4f}")
print(f"   First few values of ctx[0, 0, :5]: {ctx[0, 0, :5]}")
print(f"   First few values of ctx[1, 0, :5]: {ctx[1, 0, :5]}")
print(f"   First few values of ctx[2, 0, :5]: {ctx[2, 0, :5]}")

# Check if contexts are different
ctx_diff_01 = (ctx[0] - ctx[1]).abs().mean()
ctx_diff_12 = (ctx[1] - ctx[2]).abs().mean()
ctx_diff_02 = (ctx[0] - ctx[2]).abs().mean()
print(f"\n   Context differences:")
print(f"      |ctx[0] - ctx[1]| mean: {ctx_diff_01:.6f}")
print(f"      |ctx[1] - ctx[2]| mean: {ctx_diff_12:.6f}")
print(f"      |ctx[0] - ctx[2]| mean: {ctx_diff_02:.6f}")

if ctx_diff_01 < 1e-6 and ctx_diff_12 < 1e-6:
    print("   ❌ WARNING: All contexts are identical!")
else:
    print("   ✓ Contexts are different")

print("\n2. Token prefix (SOS):")
prefix = model.prompt_learner.token_prefix
print(f"   Shape: {prefix.shape}")
print(f"   Device: {prefix.device}")

print("\n3. Token suffix (class names + EOS):")
suffix = model.prompt_learner.token_suffix
print(f"   Shape: {suffix.shape}")
print(f"   Device: {suffix.device}")

# Check if suffixes are different (they should be, as they contain class names)
suffix_diff_01 = (suffix[0] - suffix[1]).abs().mean()
suffix_diff_12 = (suffix[1] - suffix[2]).abs().mean()
print(f"\n   Suffix differences:")
print(f"      |suffix[0] - suffix[1]| mean: {suffix_diff_01:.6f}")
print(f"      |suffix[1] - suffix[2]| mean: {suffix_diff_12:.6f}")

if suffix_diff_01 < 1e-6 and suffix_diff_12 < 1e-6:
    print("   ❌ WARNING: All suffixes are identical!")
else:
    print("   ✓ Suffixes are different (contain class names)")

print("\n4. Tokenized prompts:")
tokenized_prompts = model.prompt_learner.tokenized_prompts
print(f"   Shape: {tokenized_prompts.shape}")
print(f"   Device: {tokenized_prompts.device}")
print(f"   tokenized_prompts[0]: {tokenized_prompts[0]}")
print(f"   tokenized_prompts[1]: {tokenized_prompts[1]}")
print(f"   tokenized_prompts[2]: {tokenized_prompts[2]}")

print("\n5. Generated prompts:")
prompts = model.prompt_learner()
print(f"   Shape: {prompts.shape}")
print(f"   Device: {prompts.device}")

# Check if prompts are different
prompt_diff_01 = (prompts[0] - prompts[1]).abs().mean()
prompt_diff_12 = (prompts[1] - prompts[2]).abs().mean()
print(f"\n   Prompt differences:")
print(f"      |prompt[0] - prompt[1]| mean: {prompt_diff_01:.6f}")
print(f"      |prompt[1] - prompt[2]| mean: {prompt_diff_12:.6f}")

if prompt_diff_01 < 1e-6 and prompt_diff_12 < 1e-6:
    print("   ❌ PROBLEM: All prompts are identical!")
else:
    print("   ✓ Prompts are different")

print("\n" + "=" * 80)
print("Inspecting Text Features")
print("=" * 80)

print("\n6. Text features from prompts:")
text_features = model.text_encoder(prompts, tokenized_prompts)
print(f"   Shape: {text_features.shape}")
print(f"   Device: {text_features.device}")
print(f"   text_features[0, :5]: {text_features[0, :5]}")
print(f"   text_features[1, :5]: {text_features[1, :5]}")
print(f"   text_features[2, :5]: {text_features[2, :5]}")

# Check if text features are different
text_diff_01 = (text_features[0] - text_features[1]).abs().mean()
text_diff_12 = (text_features[1] - text_features[2]).abs().mean()
print(f"\n   Text feature differences:")
print(f"      |text[0] - text[1]| mean: {text_diff_01:.6f}")
print(f"      |text[1] - text[2]| mean: {text_diff_12:.6f}")

if text_diff_01 < 1e-6 and text_diff_12 < 1e-6:
    print("   ❌ PROBLEM: All text features are identical!")
else:
    print("   ✓ Text features are different")

# Normalize
text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)

print("\n7. Normalized text features:")
print(f"   Norms: {text_features.norm(dim=-1)}")
print(f"   After norm: {text_features_norm.norm(dim=-1)}")

print("\n" + "=" * 80)
print("Testing Full Forward Pass")
print("=" * 80)

dummy_features = torch.randn(2, 1, 512).to(device)
print(f"\n8. Dummy image features:")
print(f"   Shape: {dummy_features.shape}")

# Mean pooling
image_features = dummy_features.mean(dim=1)  # [2, 512]
print(f"   After temporal pooling: {image_features.shape}")

# Normalize
image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)

print(f"\n9. Computing similarity:")
print(f"   image_features_norm: {image_features_norm.shape}")
print(f"   text_features_norm: {text_features_norm.shape}")

# Compute similarity
similarity = image_features_norm @ text_features_norm.t()
print(f"   Similarity matrix shape: {similarity.shape}")
print(f"   Similarity matrix:")
print(similarity)

# Check logit scale
logit_scale = model.logit_scale.exp()
print(f"\n10. Logit scale:")
print(f"   Value: {logit_scale.item():.4f}")

logits = logit_scale * similarity
print(f"\n11. Final logits:")
print(logits)

# Check if logits are identical across classes
for i in range(logits.shape[0]):
    logit_std = logits[i].std()
    print(f"   Sample {i}: std={logit_std:.6f} (values: {logits[i]})")
    if logit_std < 1e-6:
        print(f"      ❌ All logits identical for sample {i}!")

print("\n" + "=" * 80)
print("Debug completed")
print("=" * 80)
