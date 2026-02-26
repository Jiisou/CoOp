"""
Visualize prompt feature evolution with t-SNE.

Compares three sets of class-wise text features:
1) Initial custom prompt text features (prompt strings only)
2) Initial custom prompt + untrained n_ctx context tokens (PromptLearner init)
3) Learned prompt text features loaded from prompt_text_cache (.pt/.npz)
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

from trainers.video_feature_coop import (
    load_mobileclip,
    TextEncoder,
    PromptLearner,
    _get_text_encoder_components,
)


def resolve_normal_index(classnames, normal_class_name="Normal"):
    target = normal_class_name.lower()
    for i, name in enumerate(classnames):
        if str(name).lower() == target:
            return i
    for i, name in enumerate(classnames):
        if "normal" in str(name).lower():
            return i
    return None


def class_color(class_idx, normal_idx, binary_label):
    if not binary_label:
        return None
    if normal_idx is None:
        return "#7f7f7f"
    return "#1f77b4" if class_idx == normal_idx else "#d62728"


def load_initial_prompts(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f, object_pairs_hook=OrderedDict)
    if not isinstance(data, dict):
        raise ValueError("initial_prompts file must be a JSON object: {classname: prompt}")
    classnames = list(data.keys())
    prompts = [data[c] for c in classnames]
    return classnames, prompts, data


def load_cache_features(cache_path):
    ext = os.path.splitext(cache_path)[1].lower()
    if ext == ".pt":
        obj = torch.load(cache_path, map_location="cpu")
        if "text_features_norm" in obj:
            feats = obj["text_features_norm"].float().cpu().numpy()
        elif "text_features" in obj:
            feats = F.normalize(obj["text_features"].float(), dim=-1).cpu().numpy()
        else:
            raise KeyError("Cache .pt must contain 'text_features_norm' or 'text_features'")
        classnames = obj.get("classnames", None)
        if classnames is not None:
            classnames = list(classnames)
        return feats.astype(np.float32), classnames

    if ext == ".npz":
        data = np.load(cache_path, allow_pickle=True)
        if "text_features_norm" in data:
            feats = data["text_features_norm"].astype(np.float32)
        elif "text_features" in data:
            feats = F.normalize(torch.from_numpy(data["text_features"]).float(), dim=-1).cpu().numpy()
        else:
            raise KeyError("Cache .npz must contain 'text_features_norm' or 'text_features'")
        classnames = data["classnames"].tolist() if "classnames" in data else None
        return feats.astype(np.float32), classnames

    raise ValueError("Unsupported cache format. Use .pt or .npz")


@torch.no_grad()
def encode_text_from_strings(clip_model, tokenizer, prompts, device):
    tokenized = tokenizer(prompts)
    if not isinstance(tokenized, torch.Tensor):
        tokenized = torch.tensor(tokenized)
    tokenized = tokenized.to(device)

    if hasattr(clip_model, "encode_text"):
        feats = clip_model.encode_text(tokenized)
    else:
        components = _get_text_encoder_components(clip_model)
        token_embedding = components["token_embedding"]
        embedding = token_embedding(tokenized).type(components["dtype"])
        text_encoder = TextEncoder(clip_model).to(device).eval()
        eot_indices = tokenized.argmax(dim=-1)
        feats = text_encoder(embedding, tokenized, eot_indices=eot_indices)

    feats = F.normalize(feats, dim=-1)
    return feats.cpu().numpy().astype(np.float32)


@torch.no_grad()
def encode_text_with_untrained_ctx(clip_model, tokenizer, classnames, class_prompts, n_ctx, csc, seed, device):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    prompt_learner = PromptLearner(
        classnames=classnames,
        clip_model=clip_model,
        tokenizer=tokenizer,
        n_ctx=n_ctx,
        csc=csc,
        class_token_position="end",
        class_prompts=class_prompts,
    ).to(device).eval()

    text_encoder = TextEncoder(clip_model).to(device).eval()
    prompt_embeddings = prompt_learner()
    tokenized_prompts = prompt_learner.tokenized_prompts
    eot_indices = prompt_learner.eot_indices if hasattr(prompt_learner, "eot_indices") else tokenized_prompts.argmax(dim=-1)
    feats = text_encoder(prompt_embeddings, tokenized_prompts, eot_indices=eot_indices)
    feats = F.normalize(feats, dim=-1)
    return feats.cpu().numpy().astype(np.float32)


def reorder_by_classnames(features, src_names, dst_names):
    if src_names is None:
        if features.shape[0] != len(dst_names):
            raise ValueError(
                f"Cannot align unnamed features: n_features={features.shape[0]}, n_classes={len(dst_names)}"
            )
        return features

    src_to_idx = {n: i for i, n in enumerate(src_names)}
    missing = [n for n in dst_names if n not in src_to_idx]
    if missing:
        raise ValueError(f"Cache classnames missing entries: {missing}")
    idx = [src_to_idx[n] for n in dst_names]
    return features[idx]


def run_tsne_and_plot(
    features_by_phase,
    classnames,
    n_components,
    random_state,
    perplexity,
    output_dir,
    binary_label=False,
    normal_class_name="Normal",
):
    phase_names = list(features_by_phase.keys())
    n_cls = len(classnames)

    X = np.concatenate([features_by_phase[p] for p in phase_names], axis=0)
    phase_ids = np.concatenate([np.full(n_cls, i) for i in range(len(phase_names))], axis=0)
    class_ids = np.tile(np.arange(n_cls), len(phase_names))

    tsne = TSNE(
        n_components=n_components,
        random_state=random_state,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    )
    Y = tsne.fit_transform(X)

    cmap = plt.cm.get_cmap("tab20", n_cls)
    markers = ["o", "^", "s", "D", "P", "X"]
    normal_idx = resolve_normal_index(classnames, normal_class_name)

    if n_components == 2:
        fig, ax = plt.subplots(figsize=(12, 10))
        for i in range(Y.shape[0]):
            color = class_color(class_ids[i], normal_idx, binary_label)
            ax.scatter(
                Y[i, 0],
                Y[i, 1],
                c=[color] if color is not None else [cmap(class_ids[i])],
                marker=markers[phase_ids[i] % len(markers)],
                s=90,
                alpha=0.9,
                edgecolors="black",
                linewidths=0.4,
            )

        for i, cname in enumerate(classnames):
            anchor = Y[i]
            ax.text(anchor[0], anchor[1], cname, fontsize=8, alpha=0.9)

        ax.set_title(f"Prompt Feature t-SNE ({n_components}D)")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(alpha=0.2)

        phase_legend = [
            Line2D([0], [0], marker=markers[i % len(markers)], color="w", label=phase_names[i],
                   markerfacecolor="gray", markeredgecolor="black", markersize=8)
            for i in range(len(phase_names))
        ]
        if binary_label:
            class_legend = [
                Line2D([0], [0], marker="o", color="w", label=f"{normal_class_name}", markerfacecolor="#1f77b4",
                       markeredgecolor="black", markersize=8),
                Line2D([0], [0], marker="o", color="w", label=f"Not {normal_class_name}", markerfacecolor="#d62728",
                       markeredgecolor="black", markersize=8),
            ]
            ax.legend(handles=phase_legend + class_legend, loc="best")
        else:
            ax.legend(handles=phase_legend, loc="best")
        fig.tight_layout()

    elif n_components == 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
        for i in range(Y.shape[0]):
            color = class_color(class_ids[i], normal_idx, binary_label)
            ax.scatter(
                Y[i, 0], Y[i, 1], Y[i, 2],
                c=[color] if color is not None else [cmap(class_ids[i])],
                marker=markers[phase_ids[i] % len(markers)],
                s=70,
                alpha=0.9,
                edgecolors="black",
                linewidths=0.3,
            )
        ax.set_title(f"Prompt Feature t-SNE ({n_components}D)")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_zlabel("t-SNE 3")
    else:
        raise ValueError("n_components must be 2 or 3")

    plot_path = os.path.join(output_dir, f"prompt_tsne_{n_components}d.png")
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    out = {
        "n_components": n_components,
        "phase_names": phase_names,
        "classnames": classnames,
        "embedding": Y.tolist(),
        "phase_ids": phase_ids.tolist(),
        "class_ids": class_ids.tolist(),
        "plot_path": plot_path,
    }
    json_path = os.path.join(output_dir, f"prompt_tsne_{n_components}d.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return plot_path, json_path


def run_tsne_single_phase(
    phase_name,
    features,
    classnames,
    n_components,
    random_state,
    perplexity,
    output_dir,
    binary_label=False,
    normal_class_name="Normal",
):
    tsne = TSNE(
        n_components=n_components,
        random_state=random_state,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    )
    Y = tsne.fit_transform(features)

    n_cls = len(classnames)
    cmap = plt.cm.get_cmap("tab20", n_cls)
    normal_idx = resolve_normal_index(classnames, normal_class_name)

    if n_components == 2:
        fig, ax = plt.subplots(figsize=(11, 9))
        for i in range(n_cls):
            color = class_color(i, normal_idx, binary_label)
            ax.scatter(
                Y[i, 0], Y[i, 1],
                c=[color] if color is not None else [cmap(i)],
                marker="o",
                s=95,
                alpha=0.95,
                edgecolors="black",
                linewidths=0.4,
            )
            ax.text(Y[i, 0], Y[i, 1], classnames[i], fontsize=8, alpha=0.9)
        ax.set_title(f"{phase_name} t-SNE ({n_components}D)")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(alpha=0.2)
        if binary_label:
            legend = [
                Line2D([0], [0], marker="o", color="w", label=f"{normal_class_name}", markerfacecolor="#1f77b4",
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
            color = class_color(i, normal_idx, binary_label)
            ax.scatter(
                Y[i, 0], Y[i, 1], Y[i, 2],
                c=[color] if color is not None else [cmap(i)],
                marker="o",
                s=75,
                alpha=0.95,
                edgecolors="black",
                linewidths=0.3,
            )
        ax.set_title(f"{phase_name} t-SNE ({n_components}D)")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_zlabel("t-SNE 3")
    else:
        raise ValueError("n_components must be 2 or 3")

    safe_name = phase_name.replace(" ", "_")
    plot_path = os.path.join(output_dir, f"prompt_tsne_{safe_name}_{n_components}d.png")
    json_path = os.path.join(output_dir, f"prompt_tsne_{safe_name}_{n_components}d.json")

    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    html_path = None
    html_error = None
    if n_components == 3:
        try:
            import plotly.graph_objects as go
            colors = []
            for i in range(n_cls):
                c = class_color(i, normal_idx, binary_label)
                if c is None:
                    c = f"hsl({(i * 360.0 / max(1, n_cls)):.1f},70%,50%)"
                colors.append(c)

            fig3d = go.Figure(
                data=[
                    go.Scatter3d(
                        x=Y[:, 0],
                        y=Y[:, 1],
                        z=Y[:, 2],
                        mode="markers+text",
                        text=classnames,
                        textposition="top center",
                        marker=dict(size=6, color=colors, line=dict(width=1, color="black")),
                    )
                ]
            )
            fig3d.update_layout(
                title=f"{phase_name} t-SNE (3D, interactive)",
                scene=dict(
                    xaxis=dict(title="", showgrid=True, showticklabels=False),
                    yaxis=dict(title="", showgrid=True, showticklabels=False),
                    zaxis=dict(title="", showgrid=True, showticklabels=False),
                ),
                margin=dict(l=0, r=0, b=0, t=40),
            )
            html_path = os.path.join(output_dir, f"prompt_tsne_{safe_name}_3d.html")
            fig3d.write_html(html_path, include_plotlyjs="cdn")
        except Exception as e:
            html_error = str(e)

    out = {
        "phase_name": phase_name,
        "n_components": n_components,
        "classnames": classnames,
        "embedding": Y.tolist(),
        "plot_path": plot_path,
        "html_path": html_path,
        "html_error": html_error,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return plot_path, json_path, html_path


def main():
    parser = argparse.ArgumentParser(description="t-SNE visualization for prompt feature evolution")
    parser.add_argument(
        "--initial-prompts-path",
        type=str,
        default="/mnt/c/Users/USER/Desktop/CoOp/annotation/initial_prompts.json",
    )
    parser.add_argument(
        "--learned-cache-path",
        type=str,
        default="/mnt/c/Users/USER/Desktop/CoOp/output/prompt_cache_test/prompt_text_cache.pt",
    )
    parser.add_argument("--mobileclip-model", type=str, default="mobileclip2_s0")
    parser.add_argument("--mobileclip-path", type=str, default=None)
    parser.add_argument("--n-ctx", type=int, default=16)
    parser.add_argument("--csc", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=10.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--components",
        type=int,
        nargs="+",
        default=[2, 3],
        help="t-SNE components to generate. Example: --components 2 3",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./output/prompt_tsne")
    parser.add_argument("--binary-label", action="store_true", default=False,
                        help="If set, color Normal vs Not-Normal in two colors")
    parser.add_argument("--normal-class-name", type=str, default="Normal",
                        help="Class name treated as normal for --binary-label")
    parser.add_argument(
        "--combined-plot",
        action="store_true",
        default=False,
        help="Also generate a combined 1/2/3 plot in a shared t-SNE space",
    )
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    classnames, prompt_texts, class_prompts = load_initial_prompts(args.initial_prompts_path)
    learned_features, learned_classnames = load_cache_features(args.learned_cache_path)
    learned_features = reorder_by_classnames(learned_features, learned_classnames, classnames)

    clip_model, tokenizer = load_mobileclip(
        pretrained_path=args.mobileclip_path,
        model_name=args.mobileclip_model,
        device=str(device),
    )

    initial_text_features = encode_text_from_strings(
        clip_model=clip_model,
        tokenizer=tokenizer,
        prompts=prompt_texts,
        device=device,
    )

    initial_plus_ctx_features = encode_text_with_untrained_ctx(
        clip_model=clip_model,
        tokenizer=tokenizer,
        classnames=classnames,
        class_prompts=class_prompts,
        n_ctx=args.n_ctx,
        csc=args.csc,
        seed=args.seed,
        device=device,
    )

    features_by_phase = OrderedDict(
        [
            ("initial_custom_prompt_text", initial_text_features),
            ("initial_custom_prompt_plus_untrained_ctx", initial_plus_ctx_features),
            ("learned_prompt_text_feature", learned_features),
        ]
    )

    np.savez_compressed(
        os.path.join(args.output_dir, "prompt_feature_sets.npz"),
        classnames=np.array(classnames),
        initial_custom_prompt_text=initial_text_features,
        initial_custom_prompt_plus_untrained_ctx=initial_plus_ctx_features,
        learned_prompt_text_feature=learned_features,
    )

    generated = []
    for c in args.components:
        if c not in (2, 3):
            raise ValueError(f"Unsupported component count: {c} (use 2 or 3)")
        for phase_name, phase_features in features_by_phase.items():
            plot_path, json_path, html_path = run_tsne_single_phase(
                phase_name=phase_name,
                features=phase_features,
                classnames=classnames,
                n_components=c,
                random_state=args.random_state,
                perplexity=args.perplexity,
                output_dir=args.output_dir,
                binary_label=args.binary_label,
                normal_class_name=args.normal_class_name,
            )
            generated.append((plot_path, json_path, html_path))

        if args.combined_plot:
            plot_path, json_path = run_tsne_and_plot(
                features_by_phase=features_by_phase,
                classnames=classnames,
                n_components=c,
                random_state=args.random_state,
                perplexity=args.perplexity,
                output_dir=args.output_dir,
                binary_label=args.binary_label,
                normal_class_name=args.normal_class_name,
            )
            generated.append((plot_path, json_path, None))

    meta = {
        "initial_prompts_path": os.path.abspath(args.initial_prompts_path),
        "learned_cache_path": os.path.abspath(args.learned_cache_path),
        "mobileclip_model": args.mobileclip_model,
        "mobileclip_path": args.mobileclip_path,
        "n_ctx": args.n_ctx,
        "csc": args.csc,
        "seed": args.seed,
        "perplexity": args.perplexity,
        "random_state": args.random_state,
        "binary_label": args.binary_label,
        "normal_class_name": args.normal_class_name,
        "resolved_normal_index": resolve_normal_index(classnames, args.normal_class_name),
        "classnames": classnames,
        "generated": [{"plot": p, "json": j, "html": h} for p, j, h in generated],
    }
    with open(os.path.join(args.output_dir, "tsne_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved t-SNE outputs:")
    for p, j, h in generated:
        print(f"  plot: {p}")
        print(f"  data: {j}")
        if h is not None:
            print(f"  html: {h}")
    print(f"  features: {os.path.join(args.output_dir, 'prompt_feature_sets.npz')}")
    print(f"  meta: {os.path.join(args.output_dir, 'tsne_meta.json')}")


if __name__ == "__main__":
    main()
