"""Debug TextEncoder forward pass step by step."""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from trainers.video_feature_coop import load_mobileclip, _get_text_encoder_components

print("=" * 80)
print("TextEncoder Forward Pass Debug")
print("=" * 80)

# Load MobileCLIP
clip_model, tokenizer = load_mobileclip(model_name="mobileclip2_s0", device="cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = clip_model.to(device)

# Extract components
components = _get_text_encoder_components(clip_model)

print("\n1. Extracted components:")
for key, val in components.items():
    if isinstance(val, torch.Tensor):
        print(f"   {key}: shape={val.shape}, dtype={val.dtype}, device={val.device}")
    elif hasattr(val, '__class__'):
        print(f"   {key}: {val.__class__.__name__}")
    else:
        print(f"   {key}: {type(val)}")

# Create dummy prompts (simulating 3 different class prompts)
n_cls = 3
seq_len = 77
embed_dim = components['ln_final'].weight.shape[0]

print(f"\n2. Creating dummy prompts:")
print(f"   n_cls={n_cls}, seq_len={seq_len}, embed_dim={embed_dim}")

# Create different prompts for each class
torch.manual_seed(42)
prompts = torch.randn(n_cls, seq_len, embed_dim).to(device)
# Make them clearly different
prompts[0] *= 0.1
prompts[1] *= 0.2
prompts[2] *= 0.3

print(f"   prompts shape: {prompts.shape}")
print(f"   prompts[0, 0, :5]: {prompts[0, 0, :5]}")
print(f"   prompts[1, 0, :5]: {prompts[1, 0, :5]}")
print(f"   prompts[2, 0, :5]: {prompts[2, 0, :5]}")

prompt_diff_01 = (prompts[0] - prompts[1]).abs().mean()
prompt_diff_12 = (prompts[1] - prompts[2]).abs().mean()
print(f"   Input differences: {prompt_diff_01:.6f}, {prompt_diff_12:.6f}")

# Create tokenized prompts (simulating the actual structure)
# [SOS, X, X, X, X, CLASS_NAME, ., EOS, PAD, ...]
tokenized_prompts = torch.zeros(n_cls, seq_len, dtype=torch.long).to(device)
tokenized_prompts[:, 0] = 49406  # SOS
tokenized_prompts[:, 1:5] = 343  # X tokens (context placeholders)
tokenized_prompts[0, 5] = 7678   # "Abuse"
tokenized_prompts[1, 5] = 5967   # "Normal"
tokenized_prompts[2, 5] = 4652   # "Fighting"
tokenized_prompts[:, 6] = 269    # "."
tokenized_prompts[:, 7] = 49407  # EOS

print(f"\n3. Tokenized prompts:")
print(f"   tokenized_prompts[0, :10]: {tokenized_prompts[0, :10]}")
print(f"   EOT positions: {tokenized_prompts.argmax(dim=-1)}")

# Step-by-step forward pass
print("\n" + "=" * 80)
print("Step-by-step Forward Pass")
print("=" * 80)

positional_embedding = components['positional_embedding']
transformer = components['transformer']
ln_final = components['ln_final']
text_projection = components['text_projection']
attn_mask = components.get('attn_mask', None)
dtype = components['dtype']

print(f"\n4. Positional embedding:")
print(f"   Shape: {positional_embedding.shape}")
print(f"   Device: {positional_embedding.device}")
print(f"   Dtype: {positional_embedding.dtype}")

# Step 1: Add positional embedding
print(f"\n5. Step 1: Add positional embedding")
print(f"   prompts shape: {prompts.shape}")
print(f"   positional_embedding shape: {positional_embedding.shape}")

x = prompts + positional_embedding.type(prompts.dtype)
print(f"   After addition: {x.shape}")
print(f"   x[0, 0, :5]: {x[0, 0, :5]}")
print(f"   x[1, 0, :5]: {x[1, 0, :5]}")
print(f"   x[2, 0, :5]: {x[2, 0, :5]}")

x_diff_01 = (x[0] - x[1]).abs().mean()
x_diff_12 = (x[1] - x[2]).abs().mean()
print(f"   Differences after pos_emb: {x_diff_01:.6f}, {x_diff_12:.6f}")

# Step 2: Permute NLD -> LND
print(f"\n6. Step 2: Permute NLD -> LND")
x = x.permute(1, 0, 2)
print(f"   After permute: {x.shape}")

# Step 3: Apply transformer
print(f"\n7. Step 3: Apply transformer")
print(f"   Attention mask: {attn_mask.shape if attn_mask is not None else None}")

if attn_mask is not None:
    attn_mask = attn_mask.to(x.device)
    if attn_mask.shape[0] != x.shape[0]:
        print(f"   Resizing attn_mask from {attn_mask.shape} to match seq_len={x.shape[0]}")
        attn_mask = attn_mask[:x.shape[0], :x.shape[0]]
    print(f"   Final attn_mask shape: {attn_mask.shape}")

x = transformer(x, attn_mask=attn_mask)
print(f"   After transformer: {x.shape}")

# Step 4: Permute back LND -> NLD
print(f"\n8. Step 4: Permute LND -> NLD")
x = x.permute(1, 0, 2)
print(f"   After permute: {x.shape}")
print(f"   x[0, 0, :5]: {x[0, 0, :5]}")
print(f"   x[1, 0, :5]: {x[1, 0, :5]}")
print(f"   x[2, 0, :5]: {x[2, 0, :5]}")

x_diff_01 = (x[0] - x[1]).abs().mean()
x_diff_12 = (x[1] - x[2]).abs().mean()
print(f"   Differences after transformer: {x_diff_01:.6f}, {x_diff_12:.6f}")

# Step 5: Layer norm
print(f"\n9. Step 5: Layer norm")
x = ln_final(x)
print(f"   After ln_final: {x.shape}")
print(f"   x[0, 0, :5]: {x[0, 0, :5]}")
print(f"   x[1, 0, :5]: {x[1, 0, :5]}")
print(f"   x[2, 0, :5]: {x[2, 0, :5]}")

x_diff_01 = (x[0] - x[1]).abs().mean()
x_diff_12 = (x[1] - x[2]).abs().mean()
print(f"   Differences after ln_final: {x_diff_01:.6f}, {x_diff_12:.6f}")

# Step 6: Extract EOT token features
print(f"\n10. Step 6: Extract EOT token features")
eot_indices = tokenized_prompts.argmax(dim=-1)
print(f"   EOT indices: {eot_indices}")

x_extracted = x[torch.arange(x.shape[0], device=x.device), eot_indices]
print(f"   Extracted features shape: {x_extracted.shape}")
print(f"   x_extracted[0, :5]: {x_extracted[0, :5]}")
print(f"   x_extracted[1, :5]: {x_extracted[1, :5]}")
print(f"   x_extracted[2, :5]: {x_extracted[2, :5]}")

x_diff_01 = (x_extracted[0] - x_extracted[1]).abs().mean()
x_diff_12 = (x_extracted[1] - x_extracted[2]).abs().mean()
print(f"   Differences after extraction: {x_diff_01:.6f}, {x_diff_12:.6f}")

# Step 7: Text projection
print(f"\n11. Step 7: Text projection")
print(f"   text_projection type: {type(text_projection)}")
if hasattr(text_projection, 'shape'):
    print(f"   text_projection shape: {text_projection.shape}")

if isinstance(text_projection, torch.nn.Linear):
    x_final = text_projection(x_extracted)
else:
    x_final = x_extracted @ text_projection

print(f"   Final features shape: {x_final.shape}")
print(f"   x_final[0, :5]: {x_final[0, :5]}")
print(f"   x_final[1, :5]: {x_final[1, :5]}")
print(f"   x_final[2, :5]: {x_final[2, :5]}")

x_diff_01 = (x_final[0] - x_final[1]).abs().mean()
x_diff_12 = (x_final[1] - x_final[2]).abs().mean()
print(f"   Final differences: {x_diff_01:.6f}, {x_diff_12:.6f}")

if x_diff_01 < 1e-6 and x_diff_12 < 1e-6:
    print("\n   ❌ PROBLEM: Final features are identical!")
else:
    print("\n   ✓ Final features are different")

print("\n" + "=" * 80)
print("Debug completed")
print("=" * 80)
