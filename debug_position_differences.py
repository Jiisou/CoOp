"""Check differences at each position after transformer."""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from trainers.video_feature_coop import load_mobileclip, VideoFeatureCLIP

print("=" * 80)
print("Position-wise Differences Debug")
print("=" * 80)

# Load model
clip_model, tokenizer = load_mobileclip(model_name="mobileclip2_s0", device="cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = clip_model.to(device)

classnames = ["Abuse", "Normal", "Fighting"]
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
model = model.to(device)

# Generate prompts
prompts = model.prompt_learner()
tokenized_prompts = model.prompt_learner.tokenized_prompts

print(f"\nTokenized prompts:")
print(f"  [0]: {tokenized_prompts[0, :10]}")
print(f"  [1]: {tokenized_prompts[1, :10]}")
print(f"  [2]: {tokenized_prompts[2, :10]}")

# Process through TextEncoder up to ln_final
text_encoder = model.text_encoder

x = prompts + text_encoder.positional_embedding.type(prompts.dtype)
x = x.permute(1, 0, 2)  # NLD -> LND

attn_mask = text_encoder.attn_mask
if attn_mask is not None:
    attn_mask = attn_mask.to(x.device)
    if attn_mask.shape[0] != x.shape[0]:
        attn_mask = attn_mask[:x.shape[0], :x.shape[0]]

x = text_encoder.transformer(x, attn_mask=attn_mask)
x = x.permute(1, 0, 2)  # LND -> NLD
x = text_encoder.ln_final(x)

print(f"\nAfter ln_final: {x.shape}")

print("\n" + "=" * 80)
print("Checking differences at each position")
print("=" * 80)

print("\nPosition | diff(0,1)  | diff(1,2)  | Status")
print("-" * 50)

for pos in range(10):
    diff_01 = (x[0, pos] - x[1, pos]).abs().mean().item()
    diff_12 = (x[1, pos] - x[2, pos]).abs().mean().item()

    status = "SAME" if (diff_01 < 1e-6 and diff_12 < 1e-6) else "DIFF"

    # Get token at this position
    token_0 = tokenized_prompts[0, pos].item()
    token_1 = tokenized_prompts[1, pos].item()
    token_2 = tokenized_prompts[2, pos].item()

    tokens_same = (token_0 == token_1 == token_2)

    print(f"{pos:8d} | {diff_01:10.6f} | {diff_12:10.6f} | {status:5s} | tokens: {token_0}{'=' if tokens_same else '≠'}{token_1}{'=' if tokens_same else '≠'}{token_2}")

print("\n" + "=" * 80)
print("EOT Token Extraction")
print("=" * 80)

eot_indices = tokenized_prompts.argmax(dim=-1)
print(f"\nEOT indices: {eot_indices}")
print(f"All EOT at same position: {(eot_indices[0] == eot_indices).all()}")

print(f"\nExtracting from position {eot_indices[0].item()}:")
x_extracted = x[torch.arange(x.shape[0], device=x.device), eot_indices]

print(f"  x_extracted[0, :5]: {x_extracted[0, :5]}")
print(f"  x_extracted[1, :5]: {x_extracted[1, :5]}")
print(f"  x_extracted[2, :5]: {x_extracted[2, :5]}")

diff_01 = (x_extracted[0] - x_extracted[1]).abs().mean()
diff_12 = (x_extracted[1] - x_extracted[2]).abs().mean()
print(f"\n  Differences: {diff_01:.6f}, {diff_12:.6f}")

# Check if different positions would give different results
print("\n" + "=" * 80)
print("What if we extract from different positions?")
print("=" * 80)

for test_pos in [1, 2, 3, 4, 5, 6, 7, 8]:
    x_test = x[:, test_pos, :]
    diff_01 = (x_test[0] - x_test[1]).abs().mean().item()
    diff_12 = (x_test[1] - x_test[2]).abs().mean().item()

    status = "DIFF" if diff_01 > 1e-6 or diff_12 > 1e-6 else "SAME"

    print(f"Position {test_pos}: diff(0,1)={diff_01:.6f}, diff(1,2)={diff_12:.6f} [{status}]")

print("\n" + "=" * 80)
print("Debug completed")
print("=" * 80)
