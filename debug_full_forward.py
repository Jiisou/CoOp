"""Debug the full forward pass with actual model components."""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from trainers.video_feature_coop import load_mobileclip, VideoFeatureCLIP

print("=" * 80)
print("Full Forward Pass Debug")
print("=" * 80)

# Load MobileCLIP
clip_model, tokenizer = load_mobileclip(model_name="mobileclip2_s0", device="cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = clip_model.to(device)

classnames = ["Abuse", "Normal", "Fighting"]

print(f"\nClasses: {classnames}")
print(f"Device: {device}")

# Create model
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

print("\n" + "=" * 80)
print("Step 1: Generate Prompts")
print("=" * 80)

prompts = model.prompt_learner()
print(f"\nPrompts shape: {prompts.shape}")
print(f"Prompts device: {prompts.device}")

# Check prompt differences
diff_01 = (prompts[0] - prompts[1]).abs().mean()
diff_12 = (prompts[1] - prompts[2]).abs().mean()
print(f"\nPrompt differences:")
print(f"  |prompts[0] - prompts[1]| mean: {diff_01:.6f}")
print(f"  |prompts[1] - prompts[2]| mean: {diff_12:.6f}")

# Show a few values
print(f"\nPrompt values at position 1 (context):")
print(f"  prompts[0, 1, :5]: {prompts[0, 1, :5]}")
print(f"  prompts[1, 1, :5]: {prompts[1, 1, :5]}")
print(f"  prompts[2, 1, :5]: {prompts[2, 1, :5]}")

print("\n" + "=" * 80)
print("Step 2: Get Tokenized Prompts")
print("=" * 80)

tokenized_prompts = model.prompt_learner.tokenized_prompts
print(f"\nTokenized prompts shape: {tokenized_prompts.shape}")
print(f"Tokenized prompts device: {tokenized_prompts.device}")
print(f"\nTokenized prompts:")
print(f"  [0]: {tokenized_prompts[0, :10]}")
print(f"  [1]: {tokenized_prompts[1, :10]}")
print(f"  [2]: {tokenized_prompts[2, :10]}")

print("\n" + "=" * 80)
print("Step 3: Encode Text Features")
print("=" * 80)

print(f"\nCalling text_encoder...")
print(f"  Input prompts shape: {prompts.shape}")
print(f"  Input tokenized_prompts shape: {tokenized_prompts.shape}")

text_features = model.text_encoder(prompts, tokenized_prompts)

print(f"\nText features shape: {text_features.shape}")
print(f"Text features device: {text_features.device}")

print(f"\nText features:")
print(f"  [0, :10]: {text_features[0, :10]}")
print(f"  [1, :10]: {text_features[1, :10]}")
print(f"  [2, :10]: {text_features[2, :10]}")

# Check differences
diff_01 = (text_features[0] - text_features[1]).abs().mean()
diff_12 = (text_features[1] - text_features[2]).abs().mean()
diff_02 = (text_features[0] - text_features[2]).abs().mean()

print(f"\nText feature differences:")
print(f"  |text[0] - text[1]| mean: {diff_01:.6f}")
print(f"  |text[1] - text[2]| mean: {diff_12:.6f}")
print(f"  |text[0] - text[2]| mean: {diff_02:.6f}")

if diff_01 < 1e-6 and diff_12 < 1e-6:
    print("\n  ❌ PROBLEM: Text features are identical!")

    # Let's debug the text_encoder internal steps
    print("\n" + "=" * 80)
    print("Debugging TextEncoder Internals")
    print("=" * 80)

    # Manually run through text_encoder steps
    text_encoder = model.text_encoder

    print(f"\n4.1. Adding positional embedding:")
    print(f"  positional_embedding shape: {text_encoder.positional_embedding.shape}")
    print(f"  positional_embedding device: {text_encoder.positional_embedding.device}")

    x = prompts + text_encoder.positional_embedding.type(prompts.dtype)
    print(f"  After addition: {x.shape}")

    diff_01_x = (x[0] - x[1]).abs().mean()
    print(f"  Differences after pos_emb: {diff_01_x:.6f}")

    print(f"\n4.2. Permuting NLD -> LND:")
    x = x.permute(1, 0, 2)
    print(f"  After permute: {x.shape}")

    print(f"\n4.3. Applying transformer:")
    attn_mask = text_encoder.attn_mask
    if attn_mask is not None:
        attn_mask = attn_mask.to(x.device)
        if attn_mask.shape[0] != x.shape[0]:
            attn_mask = attn_mask[:x.shape[0], :x.shape[0]]
        print(f"  attn_mask shape: {attn_mask.shape}")
    else:
        print(f"  attn_mask: None")

    x = text_encoder.transformer(x, attn_mask=attn_mask)
    print(f"  After transformer: {x.shape}")

    print(f"\n4.4. Permuting LND -> NLD:")
    x = x.permute(1, 0, 2)
    print(f"  After permute: {x.shape}")

    diff_01_x = (x[0] - x[1]).abs().mean()
    print(f"  Differences after transformer: {diff_01_x:.6f}")

    print(f"\n4.5. Layer norm:")
    x = text_encoder.ln_final(x)
    print(f"  After ln_final: {x.shape}")

    diff_01_x = (x[0] - x[1]).abs().mean()
    print(f"  Differences after ln_final: {diff_01_x:.6f}")

    print(f"\n4.6. Extracting EOT features:")
    eot_indices = tokenized_prompts.argmax(dim=-1)
    print(f"  EOT indices: {eot_indices}")
    print(f"  Are all EOT indices the same? {(eot_indices[0] == eot_indices).all()}")

    x_extracted = x[torch.arange(x.shape[0], device=x.device), eot_indices]
    print(f"  Extracted shape: {x_extracted.shape}")
    print(f"  x_extracted[0, :5]: {x_extracted[0, :5]}")
    print(f"  x_extracted[1, :5]: {x_extracted[1, :5]}")
    print(f"  x_extracted[2, :5]: {x_extracted[2, :5]}")

    diff_01_x = (x_extracted[0] - x_extracted[1]).abs().mean()
    print(f"  Differences after extraction: {diff_01_x:.6f}")

    print(f"\n4.7. Text projection:")
    if isinstance(text_encoder.text_projection, torch.nn.Linear):
        x_final = text_encoder.text_projection(x_extracted)
    else:
        x_final = x_extracted @ text_encoder.text_projection

    print(f"  Final shape: {x_final.shape}")
    print(f"  x_final[0, :5]: {x_final[0, :5]}")
    print(f"  x_final[1, :5]: {x_final[1, :5]}")
    print(f"  x_final[2, :5]: {x_final[2, :5]}")

    diff_01_x = (x_final[0] - x_final[1]).abs().mean()
    print(f"  Final differences: {diff_01_x:.6f}")

    # Compare with actual text_encoder output
    diff_manual_vs_actual = (x_final - text_features).abs().mean()
    print(f"\n  |manual - actual| mean: {diff_manual_vs_actual:.6f}")

else:
    print("\n  ✓ Text features are different!")

print("\n" + "=" * 80)
print("Debug completed")
print("=" * 80)
