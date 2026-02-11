"""Check if token embeddings are actually different for different class names."""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from trainers.video_feature_coop import load_mobileclip, _get_text_encoder_components

print("=" * 80)
print("Token Embeddings Debug")
print("=" * 80)

# Load MobileCLIP
clip_model, tokenizer = load_mobileclip(model_name="mobileclip2_s0", device="cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = clip_model.to(device)

components = _get_text_encoder_components(clip_model)
token_embedding = components['token_embedding']

print(f"\nDevice: {device}")
print(f"Token embedding device: {token_embedding.weight.device}")

# Test class names
classnames = ["Abuse", "Normal", "Fighting"]

print(f"\n1. Tokenizing class names:")
for name in classnames:
    tokens = tokenizer([f"X X X X {name} ."])
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.tensor(tokens)
    print(f"   '{name}': {tokens[0, :10]}")

print(f"\n2. Getting embeddings for class names:")
embeddings_list = []
for name in classnames:
    tokens = tokenizer([f"X X X X {name} ."])
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.tensor(tokens)
    tokens = tokens.to(device)

    with torch.no_grad():
        emb = token_embedding(tokens)
    embeddings_list.append(emb)

    print(f"\n   '{name}':")
    print(f"      Tokens: {tokens[0, :10]}")
    print(f"      Embedding shape: {emb.shape}")
    print(f"      Position 5 (class name) embedding[:5]: {emb[0, 5, :5]}")

print(f"\n3. Checking embedding differences:")
# Compare position 5 (where class name is)
emb_0_5 = embeddings_list[0][0, 5, :]
emb_1_5 = embeddings_list[1][0, 5, :]
emb_2_5 = embeddings_list[2][0, 5, :]

diff_01 = (emb_0_5 - emb_1_5).abs().mean()
diff_12 = (emb_1_5 - emb_2_5).abs().mean()
diff_02 = (emb_0_5 - emb_2_5).abs().mean()

print(f"   Position 5 (class name) differences:")
print(f"      |emb[0,5] - emb[1,5]| mean: {diff_01:.6f}")
print(f"      |emb[1,5] - emb[2,5]| mean: {diff_12:.6f}")
print(f"      |emb[0,5] - emb[2,5]| mean: {diff_02:.6f}")

if diff_01 < 1e-6 and diff_12 < 1e-6:
    print("\n   ❌ PROBLEM: Class name embeddings are identical!")
else:
    print("\n   ✓ Class name embeddings are different")

# Check ALL positions
print(f"\n4. Checking all positions:")
for pos in range(10):
    emb_0_pos = embeddings_list[0][0, pos, :]
    emb_1_pos = embeddings_list[1][0, pos, :]
    emb_2_pos = embeddings_list[2][0, pos, :]

    diff_01_pos = (emb_0_pos - emb_1_pos).abs().mean()
    diff_12_pos = (emb_1_pos - emb_2_pos).abs().mean()

    status = "SAME" if (diff_01_pos < 1e-6 and diff_12_pos < 1e-6) else "DIFF"
    print(f"   Position {pos}: diff(0,1)={diff_01_pos:.6f}, diff(1,2)={diff_12_pos:.6f} [{status}]")

print("\n" + "=" * 80)
print("Checking what happens with modified prompts")
print("=" * 80)

# Now create prompts with learnable contexts (random)
print(f"\n5. Creating prompts with random contexts:")
n_cls = 3
n_ctx = 4
ctx_dim = 512

# Random context vectors (simulating learned contexts)
torch.manual_seed(42)
ctx = torch.randn(n_cls, n_ctx, ctx_dim).to(device) * 0.02

print(f"   ctx shape: {ctx.shape}")
print(f"   ctx[0, 0, :5]: {ctx[0, 0, :5]}")
print(f"   ctx[1, 0, :5]: {ctx[1, 0, :5]}")
print(f"   ctx[2, 0, :5]: {ctx[2, 0, :5]}")

# Get token embeddings for full prompts
print(f"\n6. Getting token embeddings for full prompts:")
full_embeddings = []
for i, name in enumerate(classnames):
    tokens = tokenizer([f"X X X X {name} ."])
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.tensor(tokens)
    tokens = tokens.to(device)

    with torch.no_grad():
        emb = token_embedding(tokens)
    full_embeddings.append(emb)

# Manually replace context positions with random contexts
print(f"\n7. Replacing context positions (1-4) with random contexts:")
modified_embeddings = []
for i in range(n_cls):
    emb = full_embeddings[i].clone()
    # Replace positions 1-4 with random context
    emb[0, 1:5, :] = ctx[i, :, :]
    modified_embeddings.append(emb)

    print(f"   Class {i} ({classnames[i]}):")
    print(f"      Position 0 (SOS): {emb[0, 0, :5]}")
    print(f"      Position 1 (ctx): {emb[0, 1, :5]}")
    print(f"      Position 5 (name): {emb[0, 5, :5]}")

# Check differences in modified embeddings
print(f"\n8. Modified embedding differences:")
for pos in [0, 1, 2, 3, 4, 5]:
    diff_01 = (modified_embeddings[0][0, pos] - modified_embeddings[1][0, pos]).abs().mean()
    diff_12 = (modified_embeddings[1][0, pos] - modified_embeddings[2][0, pos]).abs().mean()
    status = "SAME" if (diff_01 < 1e-6 and diff_12 < 1e-6) else "DIFF"
    print(f"   Position {pos}: diff(0,1)={diff_01:.6f}, diff(1,2)={diff_12:.6f} [{status}]")

print("\n" + "=" * 80)
print("Debug completed")
print("=" * 80)
