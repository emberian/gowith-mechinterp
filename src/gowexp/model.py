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


def load_lm(model_id: str, dtype: str = "bfloat16", revision: str = "main",
            attn_implementation: str = "sdpa") -> LM:
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=getattr(torch, dtype),
        device_map="cuda",
        # sdpa is fast and fully compatible with our module forward-hooks; but sdpa does
        # NOT return attention weights, so the attn-range probe overrides this to "eager".
        attn_implementation=attn_implementation,
    )
    model.eval()
    cfg = model.config
    # Gemma-3-12b-it is multimodal: text hyperparams live under text_config.
    n_layers = getattr(cfg, "num_hidden_layers", None)
    if n_layers is None and hasattr(cfg, "text_config"):
        n_layers = cfg.text_config.num_hidden_layers
    return LM(model=model, tokenizer=tok, device="cuda", n_layers=int(n_layers))


def _decoder_layers(model: Any) -> Any:
    """Locate the TEXT decoder layer list across HF Gemma-3 (multimodal) nestings.
    Try known paths, then fall back to the longest ModuleList whose elements look
    like decoder layers (have a self_attn submodule)."""
    import torch.nn as nn

    for path in ("model.layers", "model.model.layers", "model.language_model.layers",
                 "language_model.model.layers", "model.text_model.layers"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if isinstance(obj, nn.ModuleList) and len(obj) and hasattr(obj[0], "self_attn"):
                return obj
        except AttributeError:
            continue
    best = None
    for _name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) and hasattr(mod[0], "self_attn"):
            if best is None or len(mod) > len(best):
                best = mod
    if best is None:
        raise RuntimeError("could not locate decoder layers on model")
    return best


def _as_ids(out: Any) -> torch.Tensor:
    """apply_chat_template may return a Tensor or a BatchEncoding/dict; normalize to
    a [1, seq] LongTensor."""
    if hasattr(out, "shape") and not hasattr(out, "data"):
        t = out
    elif isinstance(out, dict) or hasattr(out, "data"):
        t = out["input_ids"]
    else:
        t = torch.as_tensor(out)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    return t


def build_prompt_ids(lm: LM, system: str, user: str) -> torch.Tensor:
    """Apply the chat template robustly. Gemma templates historically reject a
    'system' role, so fall back to folding system into the user turn."""
    tok = lm.tokenizer
    try:
        out = tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            add_generation_prompt=True, return_tensors="pt",
        )
    except Exception:
        merged = f"{system}\n\n{user}" if system else user
        out = tok.apply_chat_template(
            [{"role": "user", "content": merged}],
            add_generation_prompt=True, return_tensors="pt",
        )
    return _as_ids(out).to(lm.device)


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
def generate_batch(lm: LM, items: list[tuple[str, str]], layers: list[int],
                   max_new_tokens: int, steer_vec: torch.Tensor | None = None,
                   steer_layer: int | None = None) -> list[dict]:
    """Left-padded batched greedy generation with DURING-generation residual capture.

    For each (system, user) item returns {completion, n_in, n_out, resids:{layer:[n_out,d_model]}}.
    Capture accumulates the last-position resid_post at each decode step (no separate
    re-forward). Optional steer_vec is added at steer_layer on every forward.
    """
    tok = lm.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    id_list = [build_prompt_ids(lm, s, u)[0] for (s, u) in items]
    lens = [int(t.shape[0]) for t in id_list]
    maxlen = max(lens)
    B = len(id_list)
    input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long, device=lm.device)
    attn = torch.zeros((B, maxlen), dtype=torch.long, device=lm.device)
    for i, t in enumerate(id_list):
        input_ids[i, maxlen - lens[i]:] = t
        attn[i, maxlen - lens[i]:] = 1

    decoder = _decoder_layers(lm.model)
    handles = []
    store: dict[int, list[torch.Tensor]] = {L: [] for L in layers}
    for L in layers:
        def mk(L):
            def hook(_m, _i, out):
                hs = out[0] if isinstance(out, tuple) else out
                store[L].append(hs[:, -1, :].detach().to(torch.float16))
            return hook
        handles.append(decoder[L].register_forward_hook(mk(L)))
    if steer_vec is not None:
        def shook(_m, _i, out):
            if isinstance(out, tuple):
                return (out[0] + steer_vec.to(out[0].dtype),) + tuple(out[1:])
            return out + steer_vec.to(out.dtype)
        handles.append(decoder[steer_layer].register_forward_hook(shook))

    try:
        out = lm.model.generate(input_ids=input_ids, attention_mask=attn,
                                max_new_tokens=max_new_tokens, do_sample=False,
                                pad_token_id=pad_id)
    finally:
        for h in handles:
            h.remove()

    gen = out[:, maxlen:]
    stacked = {L: torch.stack(store[L][1:], dim=0) for L in layers if len(store[L]) > 1}
    results = []
    for i in range(B):
        gi = gen[i]
        eos = (gi == tok.eos_token_id).nonzero()
        n_out = int(eos[0].item()) if eos.numel() else int(gi.shape[0])
        completion = tok.decode(gi[:n_out], skip_special_tokens=True)
        resids = {L: stacked[L][:n_out, i, :] for L in layers if L in stacked}
        results.append({"completion": completion, "n_in": lens[i], "n_out": n_out,
                        "gen_ids": gi[:n_out].tolist(), "resids": resids})
    return results


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
