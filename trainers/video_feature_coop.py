"""
CoOp-style prompt learning for pre-extracted video features with MobileCLIP S0.

Adapts trainers/coop.py to:
1. Use MobileCLIP S0 text encoder (via open_clip) instead of OpenAI CLIP
2. Accept pre-extracted features directly (no image encoder)
3. Support temporal aggregation for video features [B, T, D] -> [B, D]
"""

import torch
import torch.nn as nn

import open_clip


def load_mobileclip(pretrained_path: str, device: str = "cpu"):
    """Load MobileCLIP S0 model and tokenizer via open_clip.

    Args:
        pretrained_path: Path to pretrained weights (.pt file).
        device: Device to load model on.

    Returns:
        model: MobileCLIP model (eval mode, on CPU for component extraction).
        tokenizer: open_clip tokenizer for MobileCLIP-S0.
    """
    model, _, _ = open_clip.create_model_and_transforms(
        "MobileCLIP-S0", pretrained=pretrained_path,
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("MobileCLIP-S0")
    return model, tokenizer


class TextEncoder(nn.Module):
    """Text encoder wrapping MobileCLIP's text transformer.

    Follows the same pattern as trainers/coop.py:37-57 but adapted
    for open_clip's MobileCLIP architecture.
    """

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.attn_mask = clip_model.attn_mask

    def forward(self, prompts, tokenized_prompts):
        """Encode prompt embeddings through text transformer.

        Args:
            prompts: [n_cls, seq_len, ctx_dim] prompt embeddings.
            tokenized_prompts: [n_cls, seq_len] token indices (for EOT position).

        Returns:
            text_features: [n_cls, embed_dim] text feature vectors.
        """
        x = prompts + self.positional_embedding.type(prompts.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        # Apply attention mask if available
        attn_mask = self.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask.to(x.device)
            if attn_mask.shape[0] != x.shape[0]:
                attn_mask = attn_mask[:x.shape[0], :x.shape[0]]

        x = self.transformer(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # Extract EOT token features (highest token index per sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]

        # Project to embedding space
        if isinstance(self.text_projection, nn.Linear):
            x = self.text_projection(x)
        else:
            x = x @ self.text_projection

        return x


class PromptLearner(nn.Module):
    """Learnable prompt context for MobileCLIP-based CoOp.

    Follows trainers/coop.py:60-182 but adapted for open_clip tokenizer.

    Args:
        classnames: List of class name strings.
        clip_model: MobileCLIP model (for token embeddings).
        tokenizer: open_clip tokenizer.
        n_ctx: Number of learnable context tokens.
        ctx_init: Optional initialization string (e.g., "a video of a").
        csc: If True, use class-specific context vectors.
        class_token_position: Where to place class token - "end", "middle", or "front".
    """

    def __init__(
        self,
        classnames,
        clip_model,
        tokenizer,
        n_ctx=16,
        ctx_init="",
        csc=False,
        class_token_position="end",
    ):
        super().__init__()
        n_cls = len(classnames)
        ctx_dim = clip_model.ln_final.weight.shape[0]
        dtype = clip_model.token_embedding.weight.dtype

        if ctx_init:
            # Initialize context vectors from given words
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt_tokens = tokenizer(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt_tokens).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # Random initialization
            if csc:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(tokenizer(name)[0]) - 2 for name in classnames]  # exclude SOT/EOT
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([tokenizer(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # Register static token embeddings as buffers
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = class_token_position

    def forward(self):
        """Generate prompt embeddings for all classes.

        Returns:
            prompts: [n_cls, seq_len, ctx_dim] prompt embeddings.
        """
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [prefix_i, class_i, ctx_i, suffix_i], dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError(f"Unknown class_token_position: {self.class_token_position}")

        return prompts


class VideoFeatureCLIP(nn.Module):
    """CLIP model adapted for pre-extracted video features with CoOp prompt learning.

    No image encoder — features are passed directly.
    Temporal aggregation converts [B, T, D] -> [B, D] before similarity computation.

    Args:
        classnames: List of class name strings.
        clip_model: MobileCLIP model.
        tokenizer: open_clip tokenizer.
        n_ctx: Number of learnable context tokens.
        ctx_init: Context initialization string.
        csc: Class-specific context.
        class_token_position: Position of class token in prompt.
        temporal_agg: Temporal aggregation method ("mean", "max").
    """

    def __init__(
        self,
        classnames,
        clip_model,
        tokenizer,
        n_ctx=16,
        ctx_init="",
        csc=False,
        class_token_position="end",
        temporal_agg="mean",
    ):
        super().__init__()
        self.prompt_learner = PromptLearner(
            classnames, clip_model, tokenizer,
            n_ctx=n_ctx, ctx_init=ctx_init, csc=csc,
            class_token_position=class_token_position,
        )
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.temporal_agg = temporal_agg

    def forward(self, features):
        """Forward pass with pre-extracted features.

        Args:
            features: [B, T, D] video features (T frames, D dimensions)
                      or [B, D] if already aggregated.

        Returns:
            logits: [B, n_cls] classification logits.
        """
        # Temporal aggregation: [B, T, D] -> [B, D]
        if features.dim() == 3:
            if self.temporal_agg == "mean":
                image_features = features.mean(dim=1)
            elif self.temporal_agg == "max":
                image_features = features.max(dim=1).values
            else:
                raise ValueError(f"Unknown temporal_agg: {self.temporal_agg}")
        else:
            image_features = features

        # Generate text features from learned prompts
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        # L2 normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Cosine similarity with temperature scaling
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits
