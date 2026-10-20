"""Finding-level, negation-aware labels against the radiologist reference.

Replaces v1's templated NLI hypotheses ("There is no pleura in the cardiac.").
For each generated claim we ask a structured question of the reference report
(findings + impression, plus IU's MeSH / Problems tags when available):

  positive abnormal claim  F  -> 0 supported   if reference asserts F (side-compatible)
                               1 hallucinated if reference negates F or never mentions it
                               1 (mislocalized) if F is asserted only on the opposite side
                               None            if the reference is only hedged about F
  negated claim  "no F"     -> 1 if reference definitely asserts F, else 0
  normal claim  (organ ok)  -> 1 if reference asserts an abnormality of that organ, else 0
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from .text import _FINDING_RE, split_sentences, polarity

ORGAN_FINDINGS = {
    "lungs": {"pneumothorax", "pleural_effusion", "mass", "pneumonia", "edema", "consolidation",
              "opacity", "infiltrate", "atelectasis", "nodule", "vascular_congestion"},
    "heart": {"cardiomegaly"},
    "mediastinum": {"mediastinal_widening"},
    "cardiomediastinal": {"cardiomegaly", "mediastinal_widening"},
}
_RIGHT = re.compile(r"\bright\b", re.I)
_LEFT = re.compile(r"\bleft\b", re.I)
_BOTH = re.compile(r"\b(?:bilateral\w*|bibas\w*|both)\b", re.I)


def _side_of(text: str) -> Optional[str]:
    if _BOTH.search(text):
        return "both"
    r, l = bool(_RIGHT.search(text)), bool(_LEFT.search(text))
    if r and l:
        return "both"
    return "right" if r else "left" if l else None


def reference_index(findings: str, impression: str, mesh: str = "", problems: str = "") -> Dict[str, dict]:
    """status per canonical finding: positive / uncertain / negative (absent = missing key)."""
    text = " ".join(x for x in [findings, impression] if x and x.lower() != "nan")
    idx: Dict[str, dict] = {}

    def put(name, status, side):
        cur = idx.get(name)
        rank = {"positive": 3, "uncertain": 2, "negative": 1}
        if cur is None or rank[status] > rank[cur["status"]]:
            idx[name] = {"status": status, "sides": set()}
        if status == "positive" and side:
            idx[name]["sides"].add(side)

    for s in split_sentences(text):
        for name, rx in _FINDING_RE.items():
            for m in rx.finditer(s.text):
                neg, unc = polarity(s.text, m.start(), m.end())
                put(name, "negative" if neg else "uncertain" if unc else "positive", _side_of(s.text))
    # MeSH / Problems tags, e.g. "Calcified Granuloma/lung/upper lobe/right; Cardiomegaly/mild"
    for tag in re.split(r"[;|]", f"{mesh};{problems}"):
        tag = tag.strip()
        if not tag or tag.lower() in ("normal", "nan", "no indexing"):
            continue
        for name, rx in _FINDING_RE.items():
            if rx.search(tag):
                put(name, "positive", _side_of(tag))
    return idx


def label_claim(claim: dict, ref: Dict[str, dict]) -> dict:
    kind, f = claim["kind"], claim["finding"]
    if kind == "normal":
        bad = [x for x in ORGAN_FINDINGS.get(claim["location_text"], set())
               if ref.get(x, {}).get("status") == "positive"]
        return {"label": int(bool(bad)), "label_reason": "organ_abnormal_in_ref" if bad else "organ_normal_in_ref"}
    r = ref.get(f)
    if claim["negated"]:
        hit = r is not None and r["status"] == "positive"
        return {"label": int(hit), "label_reason": "ref_asserts_finding" if hit else "ref_consistent"}
    if r is None:
        return {"label": 1, "label_reason": "absent_in_ref"}
    if r["status"] == "negative":
        return {"label": 1, "label_reason": "negated_in_ref"}
    if r["status"] == "uncertain":
        return {"label": None, "label_reason": "hedged_in_ref"}
    gen_side = _side_of(claim.get("location_text") or "") or _side_of(claim.get("sentence") or "")
    sides = r["sides"]
    if gen_side in ("right", "left") and sides and "both" not in sides and gen_side not in sides:
        return {"label": 1, "label_reason": "mislocalized"}
    return {"label": 0, "label_reason": "supported"}
