"""Debug script to check gradient flow in CoOp training."""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Add CoOp root to path
sys.path.insert(0, str(Path(__file__).parent))

from trainers.video_feature_coop import load_mobileclip, VideoFeatureCLIP

print("=" * 60)
print("Gradient Flow Debug Script")
print("=" * 60)

# Load MobileCLIP
print("\n1. Loading MobileCLIP...")
clip_model, tokenizer = load_mobileclip(
    model_name="mobileclip2_s0",
    device="cpu"
)
print("✓ MobileCLIP loaded")

# Create model with CSC
classnames = ["Abuse", "Normal", "Fighting"]
print(f"\n2. Creating VideoFeatureCLIP with CSC=True...")
print(f"   Classes: {classnames}")

model = VideoFeatureCLIP(
    classnames=classnames,
    clip_model=clip_model,
    tokenizer=tokenizer,
    n_ctx=4,  # Small for testing
    ctx_init="",
    csc=True,  # Enable CSC
    class_token_position="end",
    temporal_agg="mean",
)

# Freeze everything except prompt_learner
for name, param in model.named_parameters():
    if "prompt_learner" not in name:
        param.requires_grad_(False)

print(f"✓ Model created")
print(f"   Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"   Total params: {sum(p.numel() for p in model.parameters()):,}")

# Check parameters
print("\n3. Trainable parameters:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"   {name}: shape={list(param.shape)}, requires_grad={param.requires_grad}")

# Test forward pass
print("\n4. Testing forward pass...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Device: {device}")

model = model.to(device)
dummy_features = torch.randn(4, 1, 512).to(device)  # [B=4, T=1, D=512]
dummy_labels = torch.tensor([0, 1, 2, 0]).to(device)  # Some labels

print(f"   Input shape: {dummy_features.shape}")
print(f"   Labels: {dummy_labels}")

# Forward
logits = model(dummy_features)
print(f"   Output shape: {logits.shape}")
print(f"   Logits:\n{logits}")

# Compute loss
loss = F.cross_entropy(logits, dummy_labels)
print(f"\n5. Loss computation:")
print(f"   Loss value: {loss.item():.4f}")

# Backward
print("\n6. Computing gradients...")
model.zero_grad()
loss.backward()

print("\n7. Gradient check:")
has_gradient = False
for name, param in model.named_parameters():
    if param.requires_grad:
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.mean().item()
            grad_max = param.grad.abs().max().item()
            print(f"   {name}:")
            print(f"      grad_norm={grad_norm:.6f}, grad_mean={grad_mean:.6f}, grad_max={grad_max:.6f}")
            has_gradient = True
        else:
            print(f"   {name}: ❌ NO GRADIENT")

if has_gradient:
    print("\n✓ Gradients are flowing!")
else:
    print("\n❌ NO GRADIENTS - something is wrong!")

# Test optimizer step
print("\n8. Testing optimizer step...")
optimizer = torch.optim.SGD(model.prompt_learner.parameters(), lr=0.01)

print("   Before step:")
ctx_before = model.prompt_learner.ctx.data.clone()
print(f"      ctx mean: {ctx_before.mean().item():.6f}")

optimizer.step()

print("   After step:")
ctx_after = model.prompt_learner.ctx.data
print(f"      ctx mean: {ctx_after.mean().item():.6f}")
print(f"      Change: {(ctx_after - ctx_before).abs().mean().item():.6f}")

if (ctx_after - ctx_before).abs().mean().item() > 1e-8:
    print("   ✓ Parameters updated!")
else:
    print("   ❌ Parameters NOT updated!")

# Test multiple iterations
print("\n9. Testing multiple iterations...")
losses = []
for i in range(5):
    dummy_features = torch.randn(4, 1, 512).to(device)
    dummy_labels = torch.randint(0, 3, (4,)).to(device)

    optimizer.zero_grad()
    logits = model(dummy_features)
    loss = F.cross_entropy(logits, dummy_labels)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    print(f"   Iter {i+1}: loss={loss.item():.4f}")

print(f"\n   Loss trend: {losses[0]:.4f} → {losses[-1]:.4f}")
if losses[-1] < losses[0]:
    print("   ✓ Loss is decreasing!")
else:
    print("   ⚠ Loss is not decreasing (might need more iterations)")

print("\n" + "=" * 60)
print("Debug script completed")
print("=" * 60)
