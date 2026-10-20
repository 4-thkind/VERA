"""Radiologist review pack: overlays + a CSV sheet for a stratified sample."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def make_review_pack(rows, A, risk, out_dir, attn_dir, n_each=30, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    out_dir = Path(out_dir)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
    idx = [i for i, r in enumerate(rows) if r["split"] == "test" and r["kind"] == "abnormal"
           and not r["negated"] and not r["uncertain"] and r["label"] is not None]
    idx.sort(key=lambda i: -risk[i])
    rng = np.random.default_rng(seed)
    hi = idx[: n_each]
    lo = idx[-n_each:]
    pick = list(hi) + [i for i in lo if i not in hi]
    rng.shuffle(pick)  # reviewers should not see the ranking order
    G = int(round(np.sqrt(A["Pl"].shape[-1])))
    with open(out_dir / "review_sheet.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "image_id", "generated_sentence", "finding", "location",
                    "Q1_finding_present_on_image(yes/no/unsure)", "Q2_location_correct(yes/no/na)",
                    "Q3_highlighted_region_helps_decide(yes/no)", "seconds_to_decide", "notes"])
        for case, i in enumerate(pick):
            r = rows[i]
            meta = json.loads((Path(attn_dir) / f"{r['image_id']}.json").read_text())
            img = Image.open(meta["image_path"]).convert("L").resize((512, 512))
            att = A["Pl"][i].astype(np.float32).mean(0).reshape(G, G)
            fig, ax = plt.subplots(1, 2, figsize=(8, 4))
            ax[0].imshow(img, cmap="gray")
            ax[0].set_title("image")
            ax[1].imshow(img, cmap="gray")
            ax[1].imshow(np.kron(att, np.ones((512 // G, 512 // G))), cmap="jet", alpha=.4)
            ax[1].contour(np.kron(A["Ms"][i].reshape(G, G), np.ones((512 // G, 512 // G))), [0.5], colors="w")
            ax[1].set_title("model attention + named region")
            for a in ax:
                a.axis("off")
            fig.suptitle(r["sentence"][:110], fontsize=8)
            fig.savefig(out_dir / "overlays" / f"case_{case:03d}.png", dpi=110)
            plt.close(fig)
            w.writerow([case, r["image_id"], r["sentence"], r["finding"], r["location_text"], "", "", "", "", ""])
    # answer key kept separate so reviewers stay blind
    with open(out_dir / "review_key.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "label_from_reference", "label_reason", "vera_risk"])
        for case, i in enumerate(pick):
            w.writerow([case, rows[i]["label"], rows[i]["label_reason"], float(risk[i])])
