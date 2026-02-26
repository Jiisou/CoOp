"""
t-SNE visualization from trained CoOp checkpoint (without cache file).

Flow:
1) Build VideoFeatureCLIP with classnames from initial prompts JSON
2) Load trained prompt_learner state_dict from checkpoint
3) Compute trained text features via prompt_learner + text_encoder
4) Run t-SNE (2D/3D) and save plots/json
"""

import os
import json
import argparse
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from trainers.video_feature_coop import load_mobileclip, VideoFeatureCLIP


def load_initial_prompts(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f, object_pairs_hook=OrderedDict)
    if not isinstance(data, dict):
        raise ValueError("initial prompts JSON must be {classname: prompt}")
    classnames = list(data.keys())
    return classnames, data


def resolve_normal_index(classnames, normal_class_name="Normal"):
    target = normal_class_name.lower()
    for i, name in enumerate(classnames):
        if str(name).lower() == target:
            return i
    for i, name in enumerate(classnames):
        if "normal" in str(name).lower():
            return i
    return None


def load_prompt_learner_state(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("prompt_learner_state_dict", {})
    if not state_dict:
        raise KeyError("checkpoint does not include 'prompt_learner_state_dict'")

    # Keep compatibility with prior eval scripts.
    state_dict = {
        k: v for k, v in state_dict.items()
        if "token_prefix" not in k and "token_suffix" not in k
    }

    if "ctx" in state_dict:
        checkpoint_ctx = state_dict["ctx"]
        current_ctx_shape = model.prompt_learner.ctx.shape
        checkpoint_ctx_shape = checkpoint_ctx.shape
        if checkpoint_ctx_shape != current_ctx_shape:
            if checkpoint_ctx_shape[0] != current_ctx_shape[0]:
                raise ValueError(
                    f"n_cls mismatch: checkpoint {checkpoint_ctx_shape[0]} vs model {current_ctx_shape[0]}"
                )
            if checkpoint_ctx_shape[1] < current_ctx_shape[1]:
                n_cls, checkpoint_n_ctx, ctx_dim = checkpoint_ctx_shape
                current_n_ctx = current_ctx_shape[1]
                pad_n = current_n_ctx - checkpoint_n_ctx
                padding = torch.empty(n_cls, pad_n, ctx_dim, dtype=checkpoint_ctx.dtype)
                torch.nn.init.normal_(padding, std=0.02)
                state_dict["ctx"] = torch.cat([checkpoint_ctx, padding], dim=1)
            else:
                state_dict["ctx"] = checkpoint_ctx[:, :current_ctx_shape[1], :]

    model.prompt_learner.load_state_dict(state_dict, strict=False)
    return checkpoint


@torch.no_grad()
def extract_trained_text_features(model):
    model.eval()
    prompts = model.prompt_learner()
    tokenized_prompts = model.prompt_learner.tokenized_prompts
    eot_indices = model.prompt_learner.eot_indices if hasattr(model.prompt_learner, "eot_indices") else None
    text_features = model.text_encoder(prompts, tokenized_prompts, eot_indices=eot_indices)
    text_features = F.normalize(text_features, dim=-1)
    return text_features.cpu().numpy().astype(np.float32)


def plot_tsne(features, classnames, output_dir, n_components, random_state, perplexity, binary_label, normal_class_name):
    tsne = TSNE(
        n_components=n_components,
        random_state=random_state,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    )
    emb = tsne.fit_transform(features)

    n_cls = len(classnames)
    cmap = plt.cm.get_cmap("tab20", n_cls)
    normal_idx = resolve_normal_index(classnames, normal_class_name)

    if n_components == 2:
        fig, ax = plt.subplots(figsize=(11, 9))
        for i in range(n_cls):
            if binary_label:
                color = "#1f77b4" if normal_idx is not None and i == normal_idx else "#d62728"
            else:
                color = cmap(i)
            ax.scatter(
                emb[i, 0], emb[i, 1],
                c=[color],
                s=95,
                marker="o",
                edgecolors="black",
                linewidths=0.4,
                alpha=0.95,
            )
            ax.text(emb[i, 0], emb[i, 1], classnames[i], fontsize=8, alpha=0.9)
        ax.set_title(f"Trained Prompt Text Feature t-SNE ({n_components}D)")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(alpha=0.2)
        if binary_label:
            legend = [
                Line2D([0], [0], marker="o", color="w", label=normal_class_name, markerfacecolor="#1f77b4",
                       markeredgecolor="black", markersize=8),
                Line2D([0], [0], marker="o", color="w", label=f"Not {normal_class_name}", markerfacecolor="#d62728",
                       markeredgecolor="black", markersize=8),
            ]
            ax.legend(handles=legend, loc="best")
        fig.tight_layout()
    elif n_components == 3:
        fig = plt.figure(figsize=(11, 9))
        ax = fig.add_subplot(111, projection="3d")
        for i in range(n_cls):
            if binary_label:
                color = "#1f77b4" if normal_idx is not None and i == normal_idx else "#d62728"
            else:
                color = cmap(i)
            ax.scatter(
                emb[i, 0], emb[i, 1], emb[i, 2],
                c=[color], s=75, marker="o",
                edgecolors="black", linewidths=0.3, alpha=0.95,
            )
        ax.set_title(f"Trained Prompt Text Feature t-SNE ({n_components}D)")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_zlabel("t-SNE 3")
    else:
        raise ValueError("n_components must be 2 or 3")

    plot_path = os.path.join(output_dir, f"trained_prompt_tsne_{n_components}d.png")
    json_path = os.path.join(output_dir, f"trained_prompt_tsne_{n_components}d.json")
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    html_path = None
    html_error = None
    if n_components == 3:
        try:
            import plotly.graph_objects as go

            colors = []
            for i in range(n_cls):
                if binary_label:
                    colors.append("#1f77b4" if normal_idx is not None and i == normal_idx else "#d62728")
                else:
                    colors.append(f"hsl({(i * 360.0 / max(1, n_cls)):.1f},70%,50%)")

            fig3d = go.Figure(
                data=[
                    go.Scatter3d(
                        x=emb[:, 0],
                        y=emb[:, 1],
                        z=emb[:, 2],
                        mode="markers+text",
                        text=classnames,
                        textposition="top center",
                        marker=dict(size=6, color=colors, line=dict(width=1, color="black")),
                    )
                ]
            )
            fig3d.update_layout(
                title="Trained Prompt Text Feature t-SNE (3D, interactive)",
                scene=dict(
                    xaxis=dict(title="", showgrid=True, showticklabels=False),
                    yaxis=dict(title="", showgrid=True, showticklabels=False),
                    zaxis=dict(title="", showgrid=True, showticklabels=False),
                ),
                margin=dict(l=0, r=0, b=0, t=40),
            )
            html_path = os.path.join(output_dir, "trained_prompt_tsne_3d.html")
            fig3d.write_html(html_path, include_plotlyjs="cdn")
        except Exception as e:
            html_error = str(e)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_components": n_components,
                "classnames": classnames,
                "embedding": emb.tolist(),
                "binary_label": binary_label,
                "normal_class_name": normal_class_name,
                "resolved_normal_index": normal_idx,
                "plot_path": plot_path,
                "html_path": html_path,
                "html_error": html_error,
            },
            f,
            indent=2,
        )
    return plot_path, json_path, html_path, html_error


def main():
    parser = argparse.ArgumentParser(description="Visualize trained prompt text features from checkpoint")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--initial-prompts-path", type=str, default="./annotation/initial_prompts.json")
    parser.add_argument("--mobileclip-model", type=str, default="mobileclip2_s0")
    parser.add_argument("--mobileclip-path", type=str, default=None)
    parser.add_argument("--n-ctx", type=int, default=16)
    parser.add_argument("--csc", action="store_true", default=True)
    parser.add_argument("--components", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--perplexity", type=float, default=10.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--binary-label", action="store_true", default=False)
    parser.add_argument("--normal-class-name", type=str, default="Normal")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./output/prompt_tsne_ckpt")
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    classnames, class_prompts = load_initial_prompts(args.initial_prompts_path)
    clip_model, tokenizer = load_mobileclip(
        pretrained_path=args.mobileclip_path,
        model_name=args.mobileclip_model,
        device="cpu",
    )
    model = VideoFeatureCLIP(
        classnames=classnames,
        clip_model=clip_model,
        tokenizer=tokenizer,
        n_ctx=args.n_ctx,
        class_prompts=class_prompts,
        csc=args.csc,
        class_token_position="end",
        temporal_agg="mean",
    ).to(device)

    checkpoint = load_prompt_learner_state(model, args.checkpoint_path, device)
    trained_features = extract_trained_text_features(model)

    np.savez_compressed(
        os.path.join(args.output_dir, "trained_prompt_text_features.npz"),
        classnames=np.array(classnames),
        trained_prompt_text_feature=trained_features,
    )

    generated = []
    for c in args.components:
        if c not in (2, 3):
            raise ValueError(f"Unsupported component count: {c}")
        generated.append(
            plot_tsne(
                features=trained_features,
                classnames=classnames,
                output_dir=args.output_dir,
                n_components=c,
                random_state=args.random_state,
                perplexity=args.perplexity,
                binary_label=args.binary_label,
                normal_class_name=args.normal_class_name,
            )
        )

    meta = {
        "checkpoint_path": os.path.abspath(args.checkpoint_path),
        "epoch": checkpoint.get("epoch", None),
        "initial_prompts_path": os.path.abspath(args.initial_prompts_path),
        "mobileclip_model": args.mobileclip_model,
        "mobileclip_path": args.mobileclip_path,
        "n_ctx": args.n_ctx,
        "csc": args.csc,
        "perplexity": args.perplexity,
        "random_state": args.random_state,
        "binary_label": args.binary_label,
        "normal_class_name": args.normal_class_name,
        "resolved_normal_index": resolve_normal_index(classnames, args.normal_class_name),
        "classnames": classnames,
        "generated": [{"plot": p, "json": j, "html": h, "html_error": e} for p, j, h, e in generated],
    }
    with open(os.path.join(args.output_dir, "trained_prompt_tsne_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved trained-ckpt t-SNE outputs:")
    for p, j, h, e in generated:
        print(f"  plot: {p}")
        print(f"  data: {j}")
        if h is not None:
            print(f"  html: {h}")
        if e is not None:
            print(f"  html_error: {e}")
    print(f"  features: {os.path.join(args.output_dir, 'trained_prompt_text_features.npz')}")
    print(f"  meta: {os.path.join(args.output_dir, 'trained_prompt_tsne_meta.json')}")


if __name__ == "__main__":
    main()

