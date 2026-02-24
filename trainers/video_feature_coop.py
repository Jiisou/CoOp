"""
CoOp-style prompt learning for pre-extracted video features with MobileCLIP S0.

Adapts trainers/coop.py to:
1. Use MobileCLIP S0 text encoder (via open_clip) instead of OpenAI CLIP
2. Accept pre-extracted features directly (no image encoder)
3. Support temporal aggregation for video features [B, T, D] -> [B, D]
"""

import os
import torch
import torch.nn as nn


def load_mobileclip(pretrained_path: str = None, model_name: str = "mc2_s0", device: str = "cpu"):
    """Load MobileCLIP model and tokenizer (supports v1 and v2).

    Tries multiple loading strategies:
    1. open_clip package (v2: MobileCLIP2-*) - preferred for v2
    2. Apple mobileclip package (v1: mobileclip_*) - for v1
    3. Auto-download if path not provided

    Args:
        pretrained_path: Path to pretrained weights (.pt file) or checkpoint name.
                        If None, attempts to auto-download.
        model_name: Model variant name (mobileclip_s0, mobileclip_s1, mobileclip_s2).
        device: Device to load model on.

    Returns:
        model: MobileCLIP model (eval mode, on CPU for component extraction).
        tokenizer: Tokenizer for the model.
    """
    print(f"Loading MobileCLIP model: {model_name}")

    # Strategy 1: Try open_clip package (v2)
    try:
        import open_clip
        print("  → Attempting to load via open_clip package (MobileCLIP2-*)...")

        # Map user-friendly names to open_clip v2 names
        v2_name_map = {
            "mc2_s0": "MobileCLIP2-S0",
            "mc1_s0": "MobileCLIP-S0",  # v1 in mobileclip
            # "mobileclip_s2": "MobileCLIP2-S2",
            # "mobileclip_s3": "MobileCLIP2-S3",
            # "mobileclip_s4": "MobileCLIP2-S4",
            # "mobileclip_b": "MobileCLIP2-B",
            # "mobileclip_l": "MobileCLIP2-L-14",
        }
        openclip_name = v2_name_map.get(model_name.lower(), "MobileCLIP2-S0")

        if pretrained_path and os.path.exists(pretrained_path):
            # Load from local checkpoint
            model, _, _ = open_clip.create_model_and_transforms(
                openclip_name, pretrained=pretrained_path
            )
            print(f"  ✓ Loaded from local checkpoint: {pretrained_path}")
        else:
            # Try auto-download via open_clip
            available_models = open_clip.list_pretrained()

            # Find matching pretrained checkpoints
            matching = [m for m in available_models if openclip_name in m[0]]

            if matching:
                model_str, checkpoint = matching[0]
                print(f"  → Auto-downloading {model_str} with checkpoint {checkpoint}")
                model, _, _ = open_clip.create_model_and_transforms(
                    model_str, pretrained=checkpoint
                )
                print(f"  ✓ Auto-downloaded successfully")
            else:
                # No pretrained available, try loading architecture
                print(f"  ⚠ No pretrained checkpoint found for {openclip_name}, trying v1 package...")
                raise ImportError(f"Model {openclip_name} not available in open_clip")

        tokenizer = open_clip.get_tokenizer(openclip_name)
        model = model.to(device).eval()
        print(f"  ✓ MobileCLIP v2 (open_clip) loaded successfully")
        return model, tokenizer

    except ImportError as e:
        print(f"  ℹ open_clip v2 not available ({e}), trying mobileclip package (v1)...")
    except Exception as e:
        print(f"  ℹ open_clip loading failed: {e}, trying mobileclip package (v1)...")

    # Strategy 2: Try Apple mobileclip package (version 1)
    try:
        import mobileclip
        print("  → Attempting to load via mobileclip package (v1)...")

        if pretrained_path and os.path.exists(pretrained_path):
            # Load from local checkpoint
            model, _, preprocess = mobileclip.create_model_and_transforms(
                model_name, pretrained=pretrained_path
            )
            print(f"  ✓ Loaded from local checkpoint: {pretrained_path}")
        else:
            # Auto-download
            model, _, preprocess = mobileclip.create_model_and_transforms(
                model_name, pretrained=f'{model_name}.pt'
            )
            print(f"  ✓ Auto-downloaded {model_name}")

        tokenizer = mobileclip.get_tokenizer(model_name)
        model = model.to(device).eval()
        print(f"  ✓ MobileCLIP v1 (mobileclip package) loaded successfully")
        return model, tokenizer

    except Exception as e:
        print(f"  ✗ mobileclip loading failed: {e}")
        raise RuntimeError(
            f"Failed to load MobileCLIP. Please install either:\n"
            f"  - open_clip: pip install open_clip_torch  (for MobileCLIP v2)\n"
            f"  - mobileclip: pip install git+https://github.com/apple/ml-mobileclip.git  (for MobileCLIP v1)\n"
            f"Error: {e}"
        )


def _get_text_encoder_components(clip_model):
    """Safely extract text encoder components from various CLIP model structures.

    Handles:
    - Standard CLIP: model.transformer, model.ln_final, etc.
    - CustomTextCLIP: model.text.transformer, model.text.ln_final, etc.
    - Other variants

    Returns:
        dict with keys: transformer, positional_embedding, ln_final, text_projection,
                       token_embedding, attn_mask, logit_scale, dtype
    """
    components = {}

    # Helper to try multiple attribute paths
    def get_attr(names):
        for name_path in names:
            obj = clip_model
            try:
                for attr in name_path.split('.'):
                    obj = getattr(obj, attr)
                return obj
            except AttributeError:
                continue
        raise AttributeError(f"Could not find any of {names} in model type {type(clip_model)}")

    # Extract components
    try:
        components['transformer'] = get_attr(['transformer', 'text.transformer'])
        components['positional_embedding'] = get_attr(['positional_embedding', 'text.positional_embedding'])
        components['ln_final'] = get_attr(['ln_final', 'text.ln_final'])
        components['text_projection'] = get_attr(['text_projection', 'text.text_projection'])
        components['token_embedding'] = get_attr(['token_embedding', 'text.token_embedding'])

        # Optional components
        try:
            components['attn_mask'] = get_attr(['attn_mask', 'text.attn_mask'])
        except AttributeError:
            components['attn_mask'] = None

        try:
            components['logit_scale'] = get_attr(['logit_scale', 'text.logit_scale'])
        except AttributeError:
            components['logit_scale'] = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

        # Get dtype from token_embedding
        components['dtype'] = components['token_embedding'].weight.dtype

        print(f"  ✓ Successfully extracted text encoder components from {type(clip_model).__name__}")
        return components

    except AttributeError as e:
        print(f"  ✗ Failed to extract text encoder components: {e}")
        print(f"  Model attributes: {dir(clip_model)}")
        if hasattr(clip_model, 'text'):
            print(f"  Model.text attributes: {dir(clip_model.text)}")
        raise


class TextEncoder(nn.Module):
    """Text encoder wrapping MobileCLIP's text transformer.

    Follows the same pattern as trainers/coop.py:37-57 but adapted
    for open_clip's MobileCLIP architecture.
    """

    def __init__(self, clip_model):
        super().__init__()
        components = _get_text_encoder_components(clip_model)
        self.transformer = components['transformer']
        self.positional_embedding = components['positional_embedding']
        self.ln_final = components['ln_final']
        self.text_projection = components['text_projection']
        self.attn_mask = components['attn_mask']

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

        # Extract features
        # NOTE: For CoOp with many identical tokens, EOT position may have identical
        # representations. Instead, we use mean pooling over non-padding tokens.
        # Find the first padding token (token_id = 0) for each sequence
        eot_indices = tokenized_prompts.argmax(dim=-1)  # Position of EOT token

        # Use mean pooling over tokens up to (and including) EOT position
        # This aggregates information from all meaningful tokens
        batch_size = x.shape[0]
        pooled_features = []
        for i in range(batch_size):
            eot_pos = eot_indices[i].item()
            # Mean pool from SOS (pos 0) to EOT (inclusive)
            pooled = x[i, :eot_pos+1, :].mean(dim=0)
            pooled_features.append(pooled)
        x = torch.stack(pooled_features, dim=0)

        # Project to embedding space
        if isinstance(self.text_projection, nn.Linear):
            x = self.text_projection(x)
        else:
            x = x @ self.text_projection

        return x


class PromptLearner(nn.Module):
    """Learnable prompt context for MobileCLIP-based CoOp.

    Follows trainers/coop.py:60-182 but adapted for open_clip tokenizer.
    Supports class-specific initial prompts.

    Args:
        classnames: List of class name strings.
        clip_model: MobileCLIP model (for token embeddings).
        tokenizer: open_clip tokenizer.
        n_ctx: Number of learnable context tokens.
        ctx_init: Optional initialization string (e.g., "a video of a").
                 Can also be a dict {classname: prompt_str} for class-specific init.
        csc: If True, use class-specific context vectors.
        class_token_position: Where to place class token - "end", "middle", or "front".
        class_prompts: Dict of {classname: initial_prompt_str} for class-specific initialization.
                      Alternative to ctx_init dict format.
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
        class_prompts=None,
    ):
        super().__init__()
        n_cls = len(classnames)

        # Extract components safely
        components = _get_text_encoder_components(clip_model)
        ctx_dim = components['ln_final'].weight.shape[0]
        dtype = components['dtype']
        token_embedding = components['token_embedding']

        # Check if class_prompts is provided (dict of {classname: prompt})
        use_class_specific_init = class_prompts is not None and isinstance(class_prompts, dict)

        if use_class_specific_init:
            # Initialize context vectors from class-specific prompts
            print("Initializing class-specific contexts from custom prompts")
            ctx_vectors_list = []
            prompt_prefixes = []

            classnames_normalized = [name.replace("_", " ") for name in classnames]

            for cls_name in classnames_normalized:
                # Get class-specific prompt
                prompt_str = class_prompts.get(cls_name, f"a video with {cls_name.lower()}")
                prompt_str = prompt_str.replace("_", " ")

                # Tokenize the prompt
                prompt_tokens = tokenizer([prompt_str])
                if not isinstance(prompt_tokens, torch.Tensor):
                    prompt_tokens = torch.tensor(prompt_tokens)

                with torch.no_grad():
                    prompt_tokens = prompt_tokens.to(token_embedding.weight.device)
                    embedding = token_embedding(prompt_tokens).type(dtype)

                # Extract middle tokens (skip SOS and EOS)
                # embedding shape: [1, seq_len, ctx_dim]
                # Extract first n_ctx tokens after SOS
                embedded = embedding[0, 1:, :]  # Skip SOS
                if embedded.shape[0] >= n_ctx:
                    ctx_vec = embedded[:n_ctx, :]
                else:
                    # If prompt has fewer tokens than n_ctx, pad with learnable tokens
                    padding = torch.empty(n_ctx - embedded.shape[0], ctx_dim, dtype=dtype)
                    nn.init.normal_(padding, std=0.02)
                    ctx_vec = torch.cat([embedded, padding], dim=0)

                ctx_vectors_list.append(ctx_vec)
                prompt_prefixes.append(prompt_str)

            ctx_vectors = torch.stack(ctx_vectors_list, dim=0)  # [n_cls, n_ctx, ctx_dim]
            prompt_prefix = "class-specific"

        elif ctx_init:
            # Initialize context vectors from given words
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))

            # Tokenize with proper handling for both mobileclip and open_clip
            prompt_tokens = tokenizer([ctx_init])
            if not isinstance(prompt_tokens, torch.Tensor):
                prompt_tokens = torch.tensor(prompt_tokens)

            with torch.no_grad():
                # Move tokens to same device as token_embedding
                prompt_tokens = prompt_tokens.to(token_embedding.weight.device)
                embedding = token_embedding(prompt_tokens).type(dtype)
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

        # Calculate name lengths with robust tokenization
        name_lens = []
        for name in classnames:
            tokens = tokenizer([name])
            if not isinstance(tokens, torch.Tensor):
                tokens = torch.tensor(tokens)
            # Subtract 2 for SOS/EOS tokens
            name_lens.append(tokens.shape[1] - 2)

        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        # Tokenize all prompts
        tokenized_list = []
        for p in prompts:
            tokens = tokenizer([p])
            if not isinstance(tokens, torch.Tensor):
                tokens = torch.tensor(tokens)
            tokenized_list.append(tokens)
        tokenized_prompts = torch.cat(tokenized_list, dim=0)

        with torch.no_grad():
            # Move tokenized_prompts to same device as token_embedding
            tokenized_prompts = tokenized_prompts.to(token_embedding.weight.device)
            embedding = token_embedding(tokenized_prompts).type(dtype)

        # Register static token embeddings as buffers
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        self.register_buffer("tokenized_prompts", tokenized_prompts)  # 모델 이동시 토큰화된 프롬프트도 항상 같은 디바이스에 있도록 버퍼에 등록

        self.n_cls = n_cls
        self.n_ctx = n_ctx
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
    Supports class-specific custom initial prompts.

    Args:
        classnames: List of class name strings.
        clip_model: MobileCLIP model.
        tokenizer: open_clip tokenizer.
        n_ctx: Number of learnable context tokens.
        ctx_init: Context initialization string.
        csc: Class-specific context.
        class_token_position: Position of class token in prompt.
        temporal_agg: Temporal aggregation method ("mean", "max").
        class_prompts: Dict of {classname: initial_prompt_str} for class-specific initialization.
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
        class_prompts=None,
    ):
        super().__init__()
        self.prompt_learner = PromptLearner(
            classnames, clip_model, tokenizer,
            n_ctx=n_ctx, ctx_init=ctx_init, csc=csc,
            class_token_position=class_token_position,
            class_prompts=class_prompts,
        )
        self.text_encoder = TextEncoder(clip_model)

        # Extract logit_scale safely
        components = _get_text_encoder_components(clip_model)
        self.logit_scale = components['logit_scale']
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
        tokenized_prompts = self.prompt_learner.tokenized_prompts # 버퍼에 직접 접근해 텐서를 가져옴 (올바른 DEVICE로부터)
        text_features = self.text_encoder(prompts, tokenized_prompts)

        # L2 normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Cosine similarity with temperature scaling
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits
