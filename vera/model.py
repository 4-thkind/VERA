"""Load CheXagent-2-3b for attention extraction.

v1 loaded with device_map={"":0}, which left the SigLIP tower on the meta device;
it then re-downloaded *base* XraySigLIP weights, which may not equal CheXagent's
trained vision encoder. Here we load fully on CPU first (no meta tensors), move to
GPU, and report any checkpoint keys that failed to load.
"""
from __future__ import annotations

import os

import torch

MODEL_ID = "StanfordAIMI/CheXagent-2-3b"


def load_chexagent(model_id: str = MODEL_ID, device: str = "cuda"):
    import tokenizers
    tokenizers.__version__ = "0.19.1"  # CheXagent remote code pins this string
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get("HF_TOKEN") or None
    model, info = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=False,      # no meta tensors
        output_loading_info=True,
        token=token,
    )
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=token)

    missing = [k for k in info.get("missing_keys", []) if "visual" in k]
    unexpected = [k for k in info.get("unexpected_keys", []) if "visual" in k]
    meta = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if meta:
        raise RuntimeError(f"{len(meta)} parameters still on meta device, e.g. {meta[:3]}")

    # harmonise dtype (pos_embed is created in float32 by the remote code)
    for p in model.parameters():
        if p.dtype == torch.float32:
            p.data = p.data.half()
    for b in model.buffers():
        if b.dtype == torch.float32:
            b.data = b.data.half()

    model = model.to(device).eval()
    diag = {
        "visual_missing_keys": len(missing),
        "visual_unexpected_keys": len(unexpected),
        "example_missing": missing[:5],
        "n_layers": model.config.num_hidden_layers,
        "n_heads": model.config.num_attention_heads,
    }
    return model, tok, diag
