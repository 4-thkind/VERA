"""Run chest anatomy segmentation once per image and store zone masks on the grid."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .anatomy import seg_zone_masks, segment


def run_segmentation(items, out_dir, G=32, device="cuda", log=print):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = n_bad = 0
    t0 = time.time()
    for k, it in enumerate(items):
        f = out_dir / f"{it['image_id']}.npz"
        if f.exists():
            continue
        try:
            zones = seg_zone_masks(segment(it["image_path"], device=device), G)
        except Exception as e:  # noqa
            log(f"seg failed {it['image_id']}: {e!r}"[:200])
            zones = None
        if zones is None:
            np.savez_compressed(f, valid=np.array(False))
            n_bad += 1
        else:
            np.savez_compressed(f, valid=np.array(True), **zones)
            n_ok += 1
        if (k + 1) % 100 == 0:
            log(f"[seg {k + 1}/{len(items)}] {(time.time() - t0) / 60:.1f} min")
    log(f"segmentation: {n_ok} valid, {n_bad} rejected (not frontal-looking or failed)")
