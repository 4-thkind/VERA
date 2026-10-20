"""Stage runner shared by the molab notebook and the CLI."""
from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path


class Paths:
    def __init__(self, work=None):
        self.work = Path(work or os.environ.get("VERA_WORK", "vera_work")).resolve()
        self.manifest = self.work / "manifest.json"
        self.attn = self.work / "attention"
        self.seg = self.work / "segmentation"
        self.results = self.work / "results"
        self.review = self.work / "results" / "review_pack"
        for p in (self.work, self.attn, self.seg, self.results):
            p.mkdir(parents=True, exist_ok=True)


def stage_data(P, n_total=3000, iu_root=None, log=print):
    from .data import build_manifest, download_iu, save_manifest
    root = iu_root or download_iu()
    log(f"IU CXR at {root}")
    items = build_manifest(root, n_total=n_total)
    save_manifest(items, P.manifest)
    n_cal = sum(x["split"] == "calib" for x in items)
    log(f"manifest: {len(items)} frontal studies ({n_cal} calib / {len(items) - n_cal} test), "
        f"{sum(x['ref_abnormal'] for x in items)} with abnormal reference")
    return items


def stage_extract(P, max_new_tokens=256, limit=None, log=print):
    from .data import load_manifest
    from .extract import run_batch
    from .model import load_chexagent
    items = load_manifest(P.manifest)[:limit]
    model, tok, diag = load_chexagent()
    log("model diagnostics:", diag)
    (P.work / "model_diagnostics.json").write_text(json.dumps(diag, indent=1))
    return run_batch(model, tok, items, P.attn, max_new_tokens, log=log)


def stage_segment(P, device="cuda", log=print):
    from .data import load_manifest
    from .segment import run_segmentation
    items = load_manifest(P.manifest)
    G = 32
    metas = sorted(P.attn.glob("*.json"))
    metas = [m for m in metas if not m.name.startswith("_")]
    if metas:
        G = json.loads(metas[0].read_text())["grid"]
    run_segmentation(items, P.seg, G=G, device=device, log=log)


def stage_analyze(P, log=print):
    from .analysis import build_table, evaluate
    from .data import load_manifest
    from .review import make_review_pack
    items = load_manifest(P.manifest)
    rows, A = build_table(items, P.attn, P.seg, log=log)
    R, risk = evaluate(rows, A, P.results, log=log)
    make_review_pack(rows, A, risk, P.review, P.attn)
    return R


def stage_bundle(P, log=print):
    """Zip only small outputs (results, review pack, diagnostics) to send back."""
    out = P.work / "VERA_v3_results_for_claude.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in P.results.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(P.work))
        for f in [P.work / "model_diagnostics.json", P.attn / "_failures.json", P.manifest]:
            if f.exists():
                z.write(f, f.relative_to(P.work))
        # 5 raw examples so the attention maps can be inspected directly
        for m in sorted(P.attn.glob("*.json"))[:5]:
            if not m.name.startswith("_"):
                z.write(m, m.relative_to(P.work))
                z.write(m.with_suffix(".npz"), m.with_suffix(".npz").relative_to(P.work))
    log(f"bundle: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out
