"""IU CXR manifest: frontal views only, one image per study, split by study.

Fixes over v1: lateral views were mixed in (the frontal atlas cannot be valid on
them) and splits were by image, so two views of one study could land in
different splits. Here: projection == Frontal from indiana_projections.csv,
one frontal image per study uid, and the calibration/test split is by study.

We oversample studies whose reference report is abnormal so the evaluation has
enough asserted findings to label. This is disclosed in the paper.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List

KAGGLE_SLUG = "raddar/chest-xrays-indiana-university"


def download_iu(target: str | None = None) -> Path:
    """Download via kagglehub. Set KAGGLE_USERNAME / KAGGLE_KEY as env vars (secrets)."""
    import kagglehub
    path = Path(kagglehub.dataset_download(KAGGLE_SLUG))
    if target:
        Path(target).mkdir(parents=True, exist_ok=True)
    return path


def _find(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]


def build_manifest(iu_root: str | Path, n_total: int = 3000, abnormal_frac: float = 0.75,
                   calib_frac: float = 0.4, seed: int = 42) -> List[Dict]:
    """n_total frontal studies, ~abnormal_frac with an abnormal reference; any shortfall is
    filled from the other group. calib_frac is larger than v2 because the probes train on it."""
    import pandas as pd

    root = Path(iu_root)
    reports = pd.read_csv(_find(root, "indiana_reports.csv"))
    proj = pd.read_csv(_find(root, "indiana_projections.csv"))
    img_dir = _find(root, "images_normalized") if list(root.rglob("images_normalized")) else root
    files = {p.name: p for p in img_dir.rglob("*.png")}

    frontal = proj[proj["projection"].str.lower() == "frontal"]
    first_frontal = frontal.groupby("uid")["filename"].first().to_dict()

    rows = []
    for _, r in reports.iterrows():
        uid = int(r["uid"])
        fn = first_frontal.get(uid)
        if fn is None or fn not in files:
            continue
        findings = "" if pd.isna(r.get("findings")) else str(r["findings"])
        impression = "" if pd.isna(r.get("impression")) else str(r["impression"])
        if not (findings.strip() or impression.strip()):
            continue
        problems = "" if pd.isna(r.get("Problems")) else str(r["Problems"])
        mesh = "" if pd.isna(r.get("MeSH")) else str(r["MeSH"])
        abnormal = problems.strip().lower() not in ("normal", "")
        rows.append(dict(image_id=fn.replace(".png", ""), uid=uid, image_path=str(files[fn]),
                         findings=findings, impression=impression, problems=problems,
                         mesh=mesh, ref_abnormal=abnormal))

    rng = random.Random(seed)
    ab = [x for x in rows if x["ref_abnormal"]]
    no = [x for x in rows if not x["ref_abnormal"]]
    rng.shuffle(ab)
    rng.shuffle(no)
    n_ab = min(len(ab), int(round(abnormal_frac * n_total)))
    n_no = min(len(no), n_total - n_ab)
    n_ab = min(len(ab), n_total - n_no)
    pick = ab[:n_ab] + no[:n_no]
    rng.shuffle(pick)
    n_cal = int(round(calib_frac * len(pick)))
    for i, x in enumerate(pick):
        x["split"] = "calib" if i < n_cal else "test"
    return pick


def save_manifest(items, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(items, indent=1))


def load_manifest(path):
    return json.loads(Path(path).read_text())
