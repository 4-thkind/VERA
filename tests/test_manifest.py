"""build_manifest on a fake IU layout: frontal only, one per study, study-level split."""
import sys, tempfile
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vera.data import build_manifest

with tempfile.TemporaryDirectory() as d:
    d = Path(d); img = d / "images" / "images_normalized"; img.mkdir(parents=True)
    rep, proj = [], []
    for uid in range(1, 41):
        rep.append(dict(uid=uid, MeSH="normal" if uid % 2 else "Cardiomegaly/mild", Problems="normal" if uid % 2 else "Cardiomegaly",
                        findings="Heart size is normal." if uid % 2 else "Mild cardiomegaly.", impression="No acute disease."))
        for v, kind in ((1, "Frontal"), (2, "Lateral")):
            fn = f"{uid}_IM-{uid:04d}-{v}001.dcm.png"; (img / fn).write_bytes(b"x")
            proj.append(dict(uid=uid, filename=fn, projection=kind))
    pd.DataFrame(rep).to_csv(d / "indiana_reports.csv", index=False)
    pd.DataFrame(proj).to_csv(d / "indiana_projections.csv", index=False)
    m = build_manifest(d, n_total=25, abnormal_frac=0.6, calib_frac=0.32)
    assert sum(x["ref_abnormal"] for x in m) == 15
    assert len(build_manifest(d, n_total=100)) == 40, "caps at what exists"
    assert len(m) == 25 and all("-1001" in x["image_id"] for x in m), "frontal only"
    assert len({x["uid"] for x in m}) == 25, "one image per study"
    assert sum(x["split"] == "calib" for x in m) == 8
    print("manifest test passed:", len(m), "studies,", sum(x['ref_abnormal'] for x in m), "abnormal")
