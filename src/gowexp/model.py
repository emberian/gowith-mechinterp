"""Gemma-3-12B-it loading, chat-templating, generation, and residual capture.

Imports torch/transformers lazily so the rest of the package stays importable on a
laptop with no CUDA. Only run_white / steer (on the GPU box) touch this.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_determinism(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class LM:
    model: Any
    tokenizer: Any
    device: str
    n_layers: int

    def tok_len(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))


def load_lm(model_id: str, dtype: str = "bfloat16", revision: str = "main") -> LM:
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=getattr(torch, dtype),
        device_map="cuda",
        # sdpa is fast and fully compatible with our module forward-hooks (we hook the
        # decoder layer output directly, not output_hidden_states).
        attn_implementation="sdpa",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    return LM(model=model, tokenizer=tok, device="cuda", n_layers=n_layers)


def _decoder_layers(model: Any) -> Any:
    """Locate the list of decoder layers across HF Gemma-3 nestings."""
    for path in ("model.layers", "model.model.layers", "language_model.model.layers"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("could not locate decoder layers on model")


def build_prompt_ids(lm: LM, system: str, user: str) -> torch.Tensor:
    """Apply the chat template robustly. Gemma templates historically reject a
    'system' role, so fall back to folding system into the user turn."""
    tok = lm.tokenizer
    try:
        ids = tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            add_generation_prompt=True, return_tensors="pt",
        )
    except Exception:
        merged = f"{system}\n\n{user}" if system else user
        ids = tok.apply_chat_template(
            [{"role": "user", "content": merged}],
            add_generation_prompt=True, return_tensors="pt",
        )
    return ids.to(lm.device)


@torch.no_grad()
def generate(lm: LM, input_ids: torch.Tensor, max_new_tokens: int,
             do_sample: bool = False, temperature: float = 0.7, top_p: float = 0.95,
             seed: int | None = None) -> tuple[str, torch.Tensor]:
    """Return (decoded_completion, full_sequence_ids). Greedy when do_sample=False."""
    if seed is not None:
        set_determinism(seed)
    kwargs: dict[str, Any] = dict(max_new_tokens=max_new_tokens, do_sample=do_sample,
                                  pad_token_id=lm.tokenizer.eos_token_id)
    if do_sample:
        kwargs.update(temperature=temperature, top_p=top_p)
    out = lm.model.generate(input_ids, **kwargs)
    full = out[0]
    completion = lm.tokenizer.decode(full[input_ids.shape[1]:], skip_special_tokens=True)
    return completion, full


@contextlib.contextmanager
def capture_resid(lm: LM, layers: list[int]) -> Iterator[dict[int, list[torch.Tensor]]]:
    """Capture each target layer's *output* residual (resid_post) on every forward.
    Yields a dict layer -> list of tensors; we typically run a single full forward."""
    store: dict[int, list[torch.Tensor]] = {L: [] for L in layers}
    decoder = _decoder_layers(lm.model)
    handles = []

    def mk(L: int):
        def hook(_mod, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            store[L].append(hs.detach())
        return hook

    for L in layers:
        handles.append(decoder[L].register_forward_hook(mk(L)))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def resid_at_positions(lm: LM, full_ids: torch.Tensor, layers: list[int]) -> dict[int, torch.Tensor]:
    """One clean forward over the full sequence; return {layer: [seq, d_model]} resid_post.

    resid_post of decoder layer L == that layer's output hidden state. (Equivalently
    hidden_states[L+1] from output_hidden_states.) We hook the layer module so the
    convention is unambiguous and matches Gemma Scope's `resid_post` site."""
    ids = full_ids.unsqueeze(0) if full_ids.dim() == 1 else full_ids
    with capture_resid(lm, layers) as store:
        lm.model(ids, use_cache=False)
    return {L: store[L][-1][0] for L in layers}  # [seq, d_model] per layer
