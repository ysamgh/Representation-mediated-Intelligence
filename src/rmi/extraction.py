"""Optional Hugging Face activation extraction utilities.

Install optional dependencies with: pip install -e ".[huggingface]"
"""
from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

Pooling = Literal["mean", "cls", "last_token"]


def pool_token_activations(
    hidden_states: NDArray[np.floating],
    attention_mask: NDArray[np.integer],
    *,
    pooling: Pooling = "mean",
) -> NDArray[np.float64]:
    """Pool (batch, sequence, hidden) activations into one vector per sequence."""
    states = np.asarray(hidden_states, dtype=np.float64)
    mask = np.asarray(attention_mask)
    if states.ndim != 3:
        raise ValueError("hidden_states must have shape (batch, sequence, hidden).")
    if mask.shape != states.shape[:2]:
        raise ValueError("attention_mask shape must match hidden_states batch and sequence axes.")
    if not np.isfinite(states).all():
        raise ValueError("hidden_states contains NaN or infinite values.")
    if not np.isin(mask, (0, 1)).all():
        raise ValueError("attention_mask must contain only 0 and 1.")
    if np.any(mask.sum(axis=1) == 0):
        raise ValueError("Every sequence must contain at least one non-padding token.")

    if pooling == "mean":
        weights = mask.astype(np.float64)[..., None]
        return (states * weights).sum(axis=1) / weights.sum(axis=1)
    if pooling == "cls":
        if np.any(mask[:, 0] == 0):
            raise ValueError("CLS pooling requires the first token to be unmasked.")
        return states[:, 0, :]
    if pooling == "last_token":
        positions = np.arange(mask.shape[1])[None, :]
        last_positions = np.where(mask == 1, positions, -1).max(axis=1)
        return states[np.arange(states.shape[0]), last_positions, :]
    raise ValueError(f"Unsupported pooling mode: {pooling!r}.")


def extract_hf_activations(
    model_name_or_path: str,
    texts: Sequence[str],
    *,
    layer: int = -1,
    pooling: Pooling = "mean",
    batch_size: int = 8,
    max_length: int = 512,
    device: str | None = None,
) -> NDArray[np.float64]:
    """Extract pooled hidden states from a Hugging Face AutoModel.

    Layer indexes the output.hidden_states tuple: 0 is embeddings and -1 is
    the final transformer layer. Custom remote model code is disabled.
    """
    if not texts:
        raise ValueError("texts must contain at least one input.")
    if any(not isinstance(item, str) for item in texts):
        raise TypeError("Every item in texts must be a string.")
    if batch_size < 1 or max_length < 1:
        raise ValueError("batch_size and max_length must be positive integers.")

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            'Install Hugging Face support with: pip install -e ".[huggingface]"'
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError(
                "Tokenizer has no pad token or EOS token; configure a padding token explicitly."
            )
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        model_name_or_path, output_hidden_states=True, trust_remote_code=False
    )
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(selected_device)
    model.eval()

    batches: list[NDArray[np.float64]] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = tokenizer(
                batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            )
            encoded = {key: value.to(selected_device) for key, value in encoded.items()}
            output = model(**encoded)
            hidden_states = output.hidden_states
            if hidden_states is None:
                raise RuntimeError("The selected model did not return hidden states.")
            if not -len(hidden_states) <= layer < len(hidden_states):
                raise ValueError(
                    f"layer={layer} is out of range for {len(hidden_states)} hidden-state tensors."
                )
            hidden = hidden_states[layer].detach().float().cpu().numpy()
            mask = encoded["attention_mask"].detach().cpu().numpy()
            batches.append(pool_token_activations(hidden, mask, pooling=pooling))

    return np.concatenate(batches, axis=0)
