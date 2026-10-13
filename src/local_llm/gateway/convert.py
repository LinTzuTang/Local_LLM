"""Convert Hugging Face (transformers) Whisper weights to the MLX layout mlx-whisper loads.

Used for Breeze-ASR-25: MediaTek publishes transformers safetensors, and converting them here
keeps the weights traceable to the official repo instead of a community re-upload. The network
is OpenAI Whisper, so converting is renaming tensors, plus transposing the two convolution
kernels (MLX stores them channels-last). Mirrors mlx-examples/whisper/convert.py, which starts
from OpenAI's .pt checkpoints instead.
"""

import json
import re
from pathlib import Path

# transformers name -> mlx-whisper name, applied in order.
RENAMES = [
    (r"^model\.encoder\.layers\.", "encoder.blocks."),
    (r"^model\.decoder\.layers\.", "decoder.blocks."),
    (r"^model\.encoder\.layer_norm\.", "encoder.ln_post."),
    (r"^model\.decoder\.layer_norm\.", "decoder.ln."),
    (r"^model\.decoder\.embed_tokens\.", "decoder.token_embedding."),
    (r"^model\.decoder\.embed_positions\.weight$", "decoder.positional_embedding"),
    (r"^model\.encoder\.", "encoder."),
    (r"\.self_attn_layer_norm\.", ".attn_ln."),
    (r"\.encoder_attn_layer_norm\.", ".cross_attn_ln."),
    (r"\.final_layer_norm\.", ".mlp_ln."),
    (r"\.self_attn\.", ".attn."),
    (r"\.encoder_attn\.", ".cross_attn."),
    (r"\.q_proj\.", ".query."),
    (r"\.k_proj\.", ".key."),
    (r"\.v_proj\.", ".value."),
    (r"\.out_proj\.", ".out."),
    (r"\.fc1\.", ".mlp1."),
    (r"\.fc2\.", ".mlp2."),
]
# The encoder's sinusoidal positions are recomputed by MLX; the output projection is tied to
# the token embedding.
DROP = {"model.encoder.embed_positions.weight", "proj_out.weight"}


def mlx_name(hf_name: str) -> str:
    for pattern, replacement in RENAMES:
        hf_name = re.sub(pattern, replacement, hf_name)
    return hf_name


def model_dims(config: dict) -> dict:
    """mlx-whisper ModelDimensions from a transformers WhisperConfig."""
    return {
        "n_mels": config["num_mel_bins"],
        "n_audio_ctx": config["max_source_positions"],
        "n_audio_state": config["d_model"],
        "n_audio_head": config["encoder_attention_heads"],
        "n_audio_layer": config["encoder_layers"],
        "n_vocab": config["vocab_size"],
        "n_text_ctx": config["max_target_positions"],
        "n_text_state": config["d_model"],
        "n_text_head": config["decoder_attention_heads"],
        "n_text_layer": config["decoder_layers"],
    }


def hf_whisper_to_mlx(source: Path, out: Path) -> Path:
    """source: config.json + model.safetensors -> out: config.json + weights.safetensors (fp16)."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_whisper import whisper

    dims = model_dims(json.loads((source / "config.json").read_text()))
    weights = {}
    for name, value in mx.load(str(source / "model.safetensors")).items():
        if name in DROP:
            continue
        key = mlx_name(name)
        if key.endswith(("conv1.weight", "conv2.weight")):
            value = value.swapaxes(1, 2)
        weights[key] = value.astype(mx.float16)

    # Check names and shapes against a freshly built model before writing anything.
    expected = dict(tree_flatten(whisper.Whisper(whisper.ModelDimensions(**dims)).parameters()))
    expected.pop("alignment_heads")  # not a weight; mlx-whisper derives a default
    missing, unexpected = expected.keys() - weights.keys(), weights.keys() - expected.keys()
    if missing or unexpected:
        raise ValueError(f"unexpected Whisper layout: missing {sorted(missing)[:5]}, "
                         f"unknown {sorted(unexpected)[:5]}")  # fmt: skip
    wrong = [k for k, v in weights.items() if v.shape != expected[k].shape]
    if wrong:
        raise ValueError(f"shape mismatch in {wrong[:5]}")

    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "weights.safetensors"), weights)
    (out / "config.json").write_text(json.dumps({**dims, "model_type": "whisper"}, indent=2))
    return out


CONVERTERS = {"hf-whisper-to-mlx": hf_whisper_to_mlx}
