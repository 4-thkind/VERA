"""Generates two plain, run-all notebooks from vera/*.py (no buttons, no widgets):
  VERA_v3.ipynb      - Jupyter / Colab / Kaggle
  VERA_v3_molab.py   - marimo / molab (cells chained so they run strictly in order)
"""
import json
from pathlib import Path

root = Path(__file__).parent
files = {f"vera/{p.name}": p.read_text() for p in sorted((root / "vera").glob("*.py"))}

TITLE = """# VERA v3: run all, come back later

Run all cells top to bottom. Every heavy step is resumable, so if the session drops just run all again.

1. Keys: put `KAGGLE_USERNAME`, `KAGGLE_KEY`, `HF_TOKEN` in your platform's secrets panel (molab Secrets, Colab Secrets, Kaggle Add-ons > Secrets).
2. Needs a GPU. ~3000 frontal IU CXR studies: extraction takes roughly 2-3 h on a T4.
3. At the end, send Claude the file `vera_work/VERA_v3_results_for_claude.zip`."""

# (code, inputs, outputs). inputs/outputs only matter for marimo's dependency order.
CELLS = [
    ("""import os
def _load_secret(k):
    if os.environ.get(k):
        return
    try:
        from google.colab import userdata
        os.environ[k] = userdata.get(k)
        return
    except Exception:
        pass
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ[k] = UserSecretsClient().get_secret(k)
    except Exception:
        pass
for _k in ("KAGGLE_USERNAME", "KAGGLE_KEY", "HF_TOKEN"):
    _load_secret(_k)
    print(_k, "set" if os.environ.get(_k) else "MISSING")""", [], ["os"]),
    ("""import sys
from pathlib import Path
PKG = __PKG__
for _rel, _src in PKG.items():
    (Path("vera_src") / _rel).parent.mkdir(parents=True, exist_ok=True)
    (Path("vera_src") / _rel).write_text(_src)
sys.path.insert(0, str(Path("vera_src").resolve()))
pkg_ok = True
print("vera package written:", len(PKG), "modules")""", ["os"], ["pkg_ok"]),
    ("""from vera import setup_env
env_ok = setup_env.ensure()
assert env_ok, "transformers 4.40.0 not active: restart the kernel and run all again"
""", ["pkg_ok"], ["env_ok"]),
    ("""# settings
N_TOTAL = 3000        # frontal studies (IU has ~3.8k; 75% abnormal reference, rest normal)
MAX_TOKENS = 256
from vera.pipeline import Paths
P = Paths("vera_work")
print("work dir:", P.work)""", ["env_ok"], ["N_TOTAL", "MAX_TOKENS", "P"]),
    ("""from vera.pipeline import stage_data
items = stage_data(P, n_total=N_TOTAL)""", ["P", "N_TOTAL"], ["items"]),
    ("""# smoke test: 3 images. Check grid 32x32 and the heat map sits on the chest.
import json as _json
import numpy as _np
import matplotlib.pyplot as _plt
from PIL import Image as _Image
from vera.pipeline import stage_extract
stage_extract(P, max_new_tokens=MAX_TOKENS, limit=3)
_metas = [P.attn / f"{it['image_id']}.json" for it in items[:3]]
smoke_fig, _ax = _plt.subplots(1, 3, figsize=(15, 5))
for _a, _m in zip(_ax, _metas):
    _meta = _json.loads(_m.read_text())
    _lm = _np.load(_m.with_suffix(".npz"))["layer_map"].astype(float)
    _G = _meta["grid"]
    _att = _lm[:, _lm.shape[1] // 2].mean(0)
    _a.imshow(_Image.open(_meta["image_path"]).convert("L").resize((512, 512)), cmap="gray")
    _a.imshow(_np.kron(_att / _att.max(), _np.ones((512 // _G, 512 // _G))), cmap="jet", alpha=0.35)
    _a.set_title(f"grid {_G}x{_G}, {_meta['seconds']} s", fontsize=9)
    _a.axis("off")
    print(_meta["report"][:300], "\\n")
_plt.close(smoke_fig)
smoke_fig""", ["P", "items", "MAX_TOKENS"], ["smoke_fig", "stage_extract"]),
    ("""# full extraction (resumable; prints progress + ETA every 10 images)
fails = stage_extract(P, max_new_tokens=MAX_TOKENS)
print("failures:", len(fails))""", ["smoke_fig", "stage_extract", "P", "MAX_TOKENS"], ["fails"]),
    ("""import torch
from vera.pipeline import stage_segment
stage_segment(P, device="cuda" if torch.cuda.is_available() else "cpu")
seg_done = True""", ["fails", "P"], ["seg_done"]),
    ("""from vera.pipeline import stage_analyze
from vera.analysis import summary_md
R = stage_analyze(P)
print(summary_md(R))""", ["seg_done", "P"], ["R"]),
    ("""from vera.pipeline import stage_bundle
bundle = stage_bundle(P)
try:
    from google.colab import files as _files
    _files.download(str(bundle))
except Exception:
    print("download this file and send it to Claude:", bundle)""", ["R", "P"], ["bundle"]),
]


def ipynb():
    def cell(kind, src):
        c = {"cell_type": kind, "id": f"c{len(src) % 99991}{kind[0]}", "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            c.update(outputs=[], execution_count=None)
        return c
    cells = [cell("markdown", TITLE)] + [cell("code", c.replace("__PKG__", repr(files))) for c, _, _ in CELLS]
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                       "accelerator": "GPU"}}
    (root / "VERA_v3.ipynb").write_text(json.dumps(nb, indent=1))


def marimo():
    out = ['import marimo\n\n__generated_with = "0.25.1"\napp = marimo.App(width="medium")\n\n',
           '@app.cell\ndef _():\n    import marimo as mo\n    mo.md(r"""\n' + TITLE + '\n""")\n    return\n\n']
    for code, ins, outs in CELLS:
        body = "\n".join("    " + l if l else "" for l in code.replace("__PKG__", repr(files)).splitlines())
        ret = f"    return ({', '.join(outs)},)" if outs else "    return"
        out.append(f"@app.cell\ndef _({', '.join(ins)}):\n{body}\n{ret}\n\n")
    out.append('if __name__ == "__main__":\n    app.run()\n')
    (root / "VERA_v3_molab.py").write_text("".join(out))


ipynb()
marimo()
print("wrote VERA_v3.ipynb and VERA_v3_molab.py,", len(files), "modules embedded")
