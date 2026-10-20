"""Install the pinned stack CheXagent-2-3b needs (transformers==4.40.0).

Strategy that worked on Colab (Python 3.13), kept as the fallback:
  1. try the clean pinned install (tokenizers 0.19.1 wheels exist for <=3.12)
  2. otherwise: install a modern tokenizers, then transformers==4.40.0 --no-deps,
     and neutralise transformers' strict dependency version check.
"""
from __future__ import annotations

import importlib
import shutil
import subprocess
import sys

BASE = ["albumentations", "opencv-python-headless", "einops", "pyarrow", "regex", "tiktoken",
        "matplotlib", "pandas", "kagglehub", "torchxrayvision", "scikit-image", "sentencepiece", "scikit-learn"]


def _pip(args):
    cmds = [[sys.executable, "-m", "pip", "install", "-q", *args]]
    if shutil.which("uv"):
        cmds.append(["uv", "pip", "install", "--python", sys.executable, "-q", *args])
    for c in cmds:
        r = subprocess.run(c, capture_output=True, text=True)
        if r.returncode == 0:
            return True, r.stdout[-500:]
        last = r.stderr[-1500:]
    return False, last


def _patch_dep_check():
    import importlib.util, os
    spec = importlib.util.find_spec("transformers")
    f = os.path.join(os.path.dirname(spec.origin), "dependency_versions_check.py")
    with open(f, "w") as fh:
        fh.write("def dep_version_check(pkg, hint=None):\n    pass\n")


def ensure(log=print):
    if "transformers" in sys.modules and sys.modules["transformers"].__version__ != "4.40.0":
        log("transformers is already imported at another version -> restart the kernel, then rerun setup.")
    ok, msg = _pip(BASE)
    log("base packages:", "ok" if ok else msg)
    ok, msg = _pip(["transformers==4.40.0", "tokenizers==0.19.1", "huggingface_hub==0.25.2"])
    if not ok:
        log("pinned install failed, using the --no-deps fallback")
        _pip(["tokenizers>=0.19", "huggingface_hub==0.25.2", "safetensors"])
        ok, msg = _pip(["transformers==4.40.0", "--no-deps", "--force-reinstall"])
        if not ok:
            raise RuntimeError(msg)
        _patch_dep_check()
    importlib.invalidate_caches()
    import transformers
    log("transformers", transformers.__version__)
    import torch
    log("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(),
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    return transformers.__version__ == "4.40.0"
