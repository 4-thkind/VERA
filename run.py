"""CLI alternative to the notebook:  python run.py setup|data|smoke|extract|segment|analyze|bundle|all"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

stage = sys.argv[1] if len(sys.argv) > 1 else "all"
if stage == "setup":
    from vera.setup_env import ensure
    ensure()
    sys.exit()
from vera import pipeline as pl
P = pl.Paths()
if stage in ("data", "all"):
    pl.stage_data(P)
if stage == "smoke":
    pl.stage_extract(P, limit=3)
if stage in ("extract", "all"):
    pl.stage_extract(P)
if stage in ("segment", "all"):
    pl.stage_segment(P)
if stage in ("analyze", "all"):
    pl.stage_analyze(P)
if stage in ("bundle", "all"):
    pl.stage_bundle(P)
