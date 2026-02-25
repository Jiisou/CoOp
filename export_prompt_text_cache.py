"""
Export and validate prompt embeddings/text features from a trained CoOp checkpoint.

Usage example:
python export_prompt_text_cache.py \
  --checkpoint-path ./output/checkpoints/best_custom_prompt_model.pth \
  --mobileclip-model mobileclip2_s0 \
  --output-dir ./output/prompt_cache
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from trainers.video_feature_coop import load_mobileclip, TextEncoder


def _load_checkpoint_state(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("prompt_learner_state_dict", checkpoint)
    return checkpoint, state


def _build_prompts_from_state(state, device):
    required = ["ctx", "token_prefix", "token_suffix"]
    missing = [k for k in required if k not in state]
    if missing:
        raise KeyError(f"Missing required keys in checkpoint state: {missing}")

    ctx = state["ctx"].to(device)
    token_prefix = state["token_prefix"].to(device)
    token_suffix = state["token_suffix"].to(device)

    if ctx.dim() == 2:
        ctx = ctx.unsqueeze(0).expand(token_prefix.shape[0], -1, -1)
    elif ctx.dim() != 3:
        raise ValueError(f"Unexpected ctx dim={ctx.dim()} (expected 2 or 3)")

    frozen_keys = sorted(
        [k for k in state.keys() if k.startswith("frozen_prompt_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    use_class_specific = len(frozen_keys) > 0

    if use_class_specific:
        prompts_list = []
        for i in range(token_prefix.shape[0]):
            key = f"frozen_prompt_{i}"
            if key not in state:
                raise KeyError(f"Class-specific checkpoint missing key: {key}")
            frozen = state[key].to(device).unsqueeze(0)
            prompts_i = torch.cat(
                [token_prefix[i : i + 1], frozen, ctx[i : i + 1], token_suffix[i : i + 1]],
                dim=1,
            )
            prompts_list.append(prompts_i)
        prompts = torch.cat(prompts_list, dim=0)
    else:
        prompts = torch.cat([token_prefix, ctx, token_suffix], dim=1)

    tokenized_prompts = state.get("tokenized_prompts", None)
    if tokenized_prompts is None:
        raise KeyError("Missing 'tokenized_prompts' in checkpoint state")
    tokenized_prompts = tokenized_prompts.to(device)

    eot_indices = state.get("eot_indices", None)
    if eot_indices is None:
        eot_indices = tokenized_prompts.argmax(dim=-1)
    eot_indices = eot_indices.to(device)

    return prompts, tokenized_prompts, eot_indices, use_class_specific


def _load_classnames(classnames_file):
    if classnames_file is None:
        return None

    path = Path(classnames_file)
    if not path.exists():
        raise FileNotFoundError(f"Classnames file not found: {path}")

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            classnames = list(data.keys())
        elif isinstance(data, list):
            classnames = [str(x) for x in data]
        else:
            raise ValueError("JSON classnames file must be list or dict")
    else:
        with open(path, "r", encoding="utf-8") as f:
            classnames = [line.strip() for line in f if line.strip()]

    return classnames


def main():
    parser = argparse.ArgumentParser(
        description="Export prompt embeddings and text features from CoOp checkpoint"
    )
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--mobileclip-model", type=str, default="mobileclip2_s0")
    parser.add_argument("--mobileclip-path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, help="cuda/cpu")
    parser.add_argument("--output-dir", type=str, default="./output/prompt_cache")
    parser.add_argument(
        "--classnames-file",
        type=str,
        default=None,
        help="Optional .json/.txt file for class names (for metadata)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for deterministic text-feature validation",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-7,
        help="Absolute tolerance for deterministic text-feature validation",
    )
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Export Prompt/Text Cache")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint_path}")

    checkpoint, state = _load_checkpoint_state(args.checkpoint_path, device)
    prompts, tokenized_prompts, eot_indices, use_class_specific = _build_prompts_from_state(state, device)

    clip_model, _ = load_mobileclip(
        pretrained_path=args.mobileclip_path,
        model_name=args.mobileclip_model,
        device=str(device),
    )
    text_encoder = TextEncoder(clip_model).to(device).eval()

    with torch.no_grad():
        text_features_1 = text_encoder(prompts, tokenized_prompts, eot_indices=eot_indices)
        text_features_2 = text_encoder(prompts, tokenized_prompts, eot_indices=eot_indices)
        text_features_norm = F.normalize(text_features_1, dim=-1)

    max_abs_diff = (text_features_1 - text_features_2).abs().max().item()
    deterministic_ok = torch.allclose(
        text_features_1, text_features_2, rtol=args.rtol, atol=args.atol
    )

    classnames = _load_classnames(args.classnames_file)
    n_cls = int(prompts.shape[0])
    if classnames is not None and len(classnames) != n_cls:
        print(
            f"Warning: classnames length ({len(classnames)}) != n_cls ({n_cls}). "
            f"Class names will still be saved as provided."
        )

    cache_pt_path = os.path.join(args.output_dir, "prompt_text_cache.pt")
    cache_npz_path = os.path.join(args.output_dir, "prompt_text_cache.npz")
    meta_json_path = os.path.join(args.output_dir, "prompt_text_cache_meta.json")

    cache_obj = {
        "checkpoint_path": os.path.abspath(args.checkpoint_path),
        "epoch": checkpoint.get("epoch", None),
        "n_cls": n_cls,
        "n_ctx": int(state["ctx"].shape[1]) if state["ctx"].dim() == 3 else int(state["ctx"].shape[0]),
        "ctx_dim": int(prompts.shape[-1]),
        "prompt_seq_len": int(prompts.shape[1]),
        "embed_dim": int(text_features_1.shape[-1]),
        "use_class_specific_init": use_class_specific,
        "deterministic_ok": bool(deterministic_ok),
        "max_abs_diff_text_features": float(max_abs_diff),
        "classnames": classnames,
        "prompt_embeddings": prompts.detach().cpu(),
        "text_features": text_features_1.detach().cpu(),
        "text_features_norm": text_features_norm.detach().cpu(),
        "tokenized_prompts": tokenized_prompts.detach().cpu(),
        "eot_indices": eot_indices.detach().cpu(),
    }
    torch.save(cache_obj, cache_pt_path)

    np.savez_compressed(
        cache_npz_path,
        prompt_embeddings=prompts.detach().cpu().numpy(),
        text_features=text_features_1.detach().cpu().numpy(),
        text_features_norm=text_features_norm.detach().cpu().numpy(),
        tokenized_prompts=tokenized_prompts.detach().cpu().numpy(),
        eot_indices=eot_indices.detach().cpu().numpy(),
    )

    meta = {
        "checkpoint_path": os.path.abspath(args.checkpoint_path),
        "epoch": checkpoint.get("epoch", None),
        "n_cls": n_cls,
        "n_ctx": cache_obj["n_ctx"],
        "ctx_dim": cache_obj["ctx_dim"],
        "prompt_seq_len": cache_obj["prompt_seq_len"],
        "embed_dim": cache_obj["embed_dim"],
        "use_class_specific_init": use_class_specific,
        "deterministic_ok": bool(deterministic_ok),
        "max_abs_diff_text_features": float(max_abs_diff),
        "mobileclip_model": args.mobileclip_model,
        "mobileclip_path": args.mobileclip_path,
        "device": str(device),
        "classnames": classnames,
        "files": {
            "cache_pt": cache_pt_path,
            "cache_npz": cache_npz_path,
        },
    }
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nValidation")
    print(f"  deterministic_ok: {deterministic_ok}")
    print(f"  max_abs_diff_text_features: {max_abs_diff:.3e}")

    print("\nSaved")
    print(f"  {cache_pt_path}")
    print(f"  {cache_npz_path}")
    print(f"  {meta_json_path}")

    print("\nQuick retrieval example:")
    print("  cache = torch.load('.../prompt_text_cache.pt', map_location='cpu')")
    print("  text_features = cache['text_features_norm']  # [n_cls, embed_dim]")
    print("  # retrieval logits: query_feature @ text_features.T")


if __name__ == "__main__":
    main()
