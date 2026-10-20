"""Generate a report and store sentence-level attention to the image tokens.

Key fixes over v1:
  * CheXagent-2-3b inserts 1024 visual tokens (32 x 32 grid; the image is
    resized to 512 px and SigLIP uses 16 px patches). v1 kept the first 576
    tokens and reshaped them to 24 x 24, which scrambled the spatial layout and
    dropped the bottom 44% of the image. Here the grid size is read from the
    actual image span in the input ids and asserted to be square.
  * Token -> character offsets come from incremental decoding, so each sentence
    gets exactly its own tokens (v1 used a proportional char->token guess).
  * We keep every layer, and every head (pooled to 8 x 8), plus the
    pre-renormalisation share of attention that went to the image at all.

Per image we save  <out>/<image_id>.npz  and  <out>/<image_id>.json
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .text import split_sentences

PROMPT = (
    "You are an expert radiologist. Generate a detailed radiology report "
    "for this chest X-ray image. Include findings about the lungs, heart, "
    "mediastinum, and any abnormalities. Structure your response with "
    "Findings and Impression sections."
)
HEAD_POOL = 4  # 32x32 -> 8x8 for per-head maps (storage)


def _build_inputs(tok, image_path: str, prompt: str, device):
    query = tok.from_list_format([{"image": image_path}, {"text": prompt}])
    conv = [{"from": "system", "value": "You are a helpful assistant."},
            {"from": "human", "value": query}]
    text = tok.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    enc = tok(text=text, return_tensors="pt").to(device)
    ids = enc.input_ids[0].tolist()
    a = ids.index(tok.img_start_id)
    b = ids.index(tok.img_end_id)
    n_vis = b - a - 1
    G = int(math.isqrt(n_vis))
    if G * G != n_vis:
        raise ValueError(f"image span has {n_vis} tokens, not a square grid")
    return enc, a + 1, b, G


def _token_char_ends(tok, gen_ids) -> list:
    ends, prev = [], 0
    for t in range(len(gen_ids)):
        s = tok.decode(gen_ids[: t + 1], skip_special_tokens=True)
        prev = max(prev, len(s))
        ends.append(prev)
    return ends


@torch.no_grad()
def extract_one(model, tok, image_path: str, max_new_tokens: int = 256, prompt: str = PROMPT):
    dev = next(model.parameters()).device
    enc, v0, v1, G = _build_inputs(tok, image_path, prompt, dev)
    L0 = enc.input_ids.shape[1]
    t0 = time.time()
    out = model.generate(
        input_ids=enc.input_ids, attention_mask=enc.attention_mask,
        max_new_tokens=max_new_tokens, do_sample=False,
        output_attentions=True, output_scores=True, return_dict_in_generate=True,
    )
    gen = out.sequences[0, L0:]
    T = min(len(gen), len(out.attentions))
    n_layers = len(out.attentions[0])
    n_heads = out.attentions[0][0].shape[1]
    nv = v1 - v0

    P = torch.empty((T, n_layers, n_heads, nv), dtype=torch.float16, device=dev)
    lp = torch.empty(T, dtype=torch.float32)
    for t in range(T):
        rows = torch.stack([la[0, :, -1, :] for la in out.attentions[t]])  # [L,H,S]
        P[t] = rows[:, :, v0:v1].half()
        lp[t] = torch.log_softmax(out.scores[t][0].float(), -1)[gen[t]].item()
    del out
    torch.cuda.empty_cache()

    gen_ids = gen[:T].tolist()
    text = tok.decode(gen_ids, skip_special_tokens=True)
    ends = _token_char_ends(tok, gen_ids)
    starts = [0] + ends[:-1]

    sents = split_sentences(text)
    keep, tok_spans = [], []
    for s in sents:
        idx = [t for t in range(T) if starts[t] < s.end and ends[t] > s.start and ends[t] > starts[t]]
        if idx:
            keep.append(s)
            tok_spans.append((idx[0], idx[-1] + 1))
    S = len(keep)
    g = G // HEAD_POOL
    layer_map = np.zeros((S, n_layers, G, G), np.float16)
    head_pool = np.zeros((S, n_layers, n_heads, g, g), np.float16)
    vis_mass = np.zeros((S, n_layers, n_heads), np.float32)
    conf = np.zeros((S, 2), np.float32)
    for i, (ta, tb) in enumerate(tok_spans):
        M = P[ta:tb].float().mean(0)                       # [L,H,nv]
        vis_mass[i] = M.sum(-1).cpu().numpy()
        layer_map[i] = M.mean(1).reshape(n_layers, G, G).cpu().numpy()
        hp = M.reshape(n_layers, n_heads, g, HEAD_POOL, g, HEAD_POOL).sum((3, 5))
        head_pool[i] = hp.cpu().numpy()
        conf[i] = (lp[ta:tb].mean().item(), lp[ta:tb].min().item())
    report_map = P.float().mean((0, 2)).reshape(n_layers, G, G).cpu().numpy().astype(np.float16)
    del P

    arrays = dict(layer_map=layer_map, head_pool=head_pool, vis_mass=vis_mass, conf=conf,
                  report_map=report_map, token_logprob=lp.numpy())
    meta = dict(image_path=image_path, report=text, grid=G, n_layers=n_layers, n_heads=n_heads,
                n_tokens=T, sentences=[[s.text, s.start, s.end] for s in keep],
                sentence_token_spans=tok_spans, seconds=round(time.time() - t0, 1))
    return arrays, meta


def run_batch(model, tok, items, out_dir, max_new_tokens=256, log=print):
    """items: list of dicts with image_id, image_path. Resumable."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [it for it in items if not (out_dir / f"{it['image_id']}.json").exists()]
    log(f"{len(items) - len(todo)} done already, {len(todo)} to go")
    t_start, failures = time.time(), []
    for k, it in enumerate(todo):
        try:
            try:
                arrays, meta = extract_one(model, tok, it["image_path"], max_new_tokens)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                arrays, meta = extract_one(model, tok, it["image_path"], 160)
            np.savez_compressed(out_dir / f"{it['image_id']}.npz", **arrays)
            meta["image_id"] = it["image_id"]
            (out_dir / f"{it['image_id']}.json").write_text(json.dumps(meta))
        except Exception as e:  # keep going, record failure
            failures.append((it["image_id"], repr(e)[:300]))
            torch.cuda.empty_cache()
        if (k + 1) % 10 == 0 or k + 1 == len(todo):
            el = time.time() - t_start
            eta = el / (k + 1) * (len(todo) - k - 1)
            log(f"[{k + 1}/{len(todo)}] {el / 60:.1f} min elapsed, ETA {eta / 60:.1f} min, failures {len(failures)}")
    if failures:
        (out_dir / "_failures.json").write_text(json.dumps(failures, indent=1))
    return failures
