"""Sentence splitting, claim extraction and negation for radiology text.

Fixes over v1:
  * CheXagent reports look like "[Breathing: Lungs] **Text.** [Cardiac: ...] Text."
    v1's splitter needed an uppercase letter after the period, so whole reports
    came back as ONE sentence and every finding was paired with every location.
    Here tags and markdown are treated as boundaries and every sentence keeps its
    character span in the ORIGINAL text (needed for exact token alignment).
  * Locations are paired with the nearest finding inside the same clause,
    not as a cross-product.
  * Negation is scoped (trigger shortly before the finding, clause-bounded, plus
    post-negations like "is not seen"). v1 negated anything after any "no".
  * Hedged statements ("possible", "cannot exclude") are flagged as uncertain.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

# --------------------------------------------------------------------------
# Sentence splitting with offsets into the original string
# --------------------------------------------------------------------------
# [Breathing: Lungs] section tags, CheXagent grounding tokens <|ref|> <|/ref|>,
# and <|box|> (x,y),(x,y) <|/box|> coordinate blocks are all sentence boundaries.
_TAG = re.compile(r"\[[^\]]*\]|<\|box\|>.*?<\|/box\|>|<\|[^|]{1,12}\|>")
_SENT_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Sentence:
    text: str
    start: int   # char offset in original text
    end: int


def split_sentences(text: str) -> List[Sentence]:
    """Split into sentences, dropping [tags] and ** markers, keeping offsets."""
    out: List[Sentence] = []
    if not text:
        return out
    # chunks between tags
    cuts, last = [], 0
    for m in _TAG.finditer(text):
        cuts.append((last, m.start()))
        last = m.end()
    cuts.append((last, len(text)))
    for a, b in cuts:
        chunk = text[a:b]
        pos = 0
        for piece in _SENT_END.split(chunk):
            idx = chunk.find(piece, pos)
            if idx < 0:
                continue
            pos = idx + len(piece)
            s, e = a + idx, a + idx + len(piece)
            # trim whitespace and markdown asterisks while tracking offsets
            while s < e and (text[s].isspace() or text[s] == "*"):
                s += 1
            while e > s and (text[e - 1].isspace() or text[e - 1] == "*"):
                e -= 1
            if e - s >= 4 and re.search(r"[A-Za-z]", text[s:e]):
                out.append(Sentence(text[s:e], s, e))
    return out


# --------------------------------------------------------------------------
# Findings lexicon: canonical name -> (regex, severity tier, default location)
# --------------------------------------------------------------------------
FINDINGS = {
    # critical
    "pneumothorax":       (r"pneumothora(?:x|ces)", "critical", "lungs"),
    "pleural_effusion":   (r"(?:pleural\s+)?effusions?|blunting\s+of\s+the\s+(?:\w+\s+){0,2}costophrenic|costophrenic\s+(?:angle\s+)?blunting",
                           "critical", "costophrenic"),
    "mass":               (r"\bmass(?:es)?\b", "critical", None),
    "pneumonia":          (r"pneumonias?", "critical", "lungs"),
    "edema":              (r"(?:pulmonary\s+|interstitial\s+|alveolar\s+)?edema", "critical", "lungs"),
    "fracture":           (r"fractures?", "critical", None),
    # moderate
    "consolidation":      (r"consolidat(?:ion|ions|ive)", "moderate", "lungs"),
    "opacity":            (r"opacit(?:y|ies)|opacification|airspace\s+(?:disease|opacit\w*)", "moderate", "lungs"),
    "infiltrate":         (r"infiltrat(?:e|es|ion|ive)", "moderate", "lungs"),
    "atelectasis":        (r"atelecta(?:sis|tic)", "moderate", "lungs"),
    "cardiomegaly":       (r"cardiomegaly|(?:heart|cardiac\s+silhouette|cardiac\s+size|heart\s+size)\s+(?:is\s+|appears\s+|remains\s+)?(?:\w+\s+)?(?:enlarged|enlargement)|enlarged\s+(?:heart|cardiac\s+silhouette)",
                           "moderate", "heart"),
    "mediastinal_widening": (r"mediastinal\s+widening|widen(?:ed|ing)\s+(?:of\s+the\s+)?mediastinum", "moderate", "mediastinum"),
    "nodule":             (r"nodul(?:e|es|ar)", "moderate", None),
    "vascular_congestion": (r"(?:pulmonary\s+)?vascular\s+congestion|cephalization", "moderate", "lungs"),
    "pleural_thickening": (r"pleural\s+thickening", "moderate", "pleural"),
    # mild
    "granuloma":          (r"granulomas?|granulomatous", "mild", None),
    "calcification":      (r"calcifications?|calcified\s+(?:lymph\s+)?nodes?", "mild", None),
    "scarring":           (r"scarring|\bscar\b|fibrosis|fibrotic", "mild", "lungs"),
    "hyperinflation":     (r"hyperinflat\w+|hyperexpan\w+|emphysema\w*|copd", "mild", "lungs"),
    "tortuous_aorta":     (r"tortuous\s+(?:thoracic\s+)?aorta|aortic\s+tortuosity|ectatic\s+aorta|unfolded\s+aorta|aorta\s+is\s+(?:\w+\s+){0,3}tortuous|aortic\s+enlargement|(?:enlarged|dilated)\s+aorta", "mild", "mediastinum"),
    "degenerative":       (r"degenerative\s+(?:changes?|disease)|spondylosis|osteophyt\w+", "mild", None),
    "scoliosis":          (r"scoliosis|curvature\s+of\s+the\s+spine", "mild", None),
}
_FINDING_RE = {k: re.compile(r"\b(?:" + v[0] + r")", re.I) for k, v in FINDINGS.items()}
TIER = {k: v[1] for k, v in FINDINGS.items()}
DEFAULT_LOC = {k: v[2] for k, v in FINDINGS.items()}

# Normal statements -> organ
NORMAL_PATTERNS = [
    (re.compile(r"\blungs?\s+(?:are|is|appear|remain)\s+(?:\w+\s+)?clear|\bclear\s+lungs?\b|lungs?\s+(?:are\s+)?(?:well\s+)?expanded\s+and\s+clear", re.I), "lungs"),
    (re.compile(r"\b(?:heart|cardiac)\s+(?:size|silhouette)?\s*(?:is|are|appears?|remains?)\s+(?:\w+\s+)?(?:normal|within\s+normal|unremarkable)", re.I), "heart"),
    (re.compile(r"cardiomediastinal\s+(?:silhouette|contours?)\s+(?:is|are|appears?)\s+(?:\w+\s+)?(?:normal|within\s+normal|unremarkable|stable)", re.I), "cardiomediastinal"),
    (re.compile(r"mediastinal\s+(?:contours?|silhouette)\s+(?:is|are|appears?)\s+(?:\w+\s+)?(?:normal|within\s+normal|unremarkable)", re.I), "mediastinum"),
]

# --------------------------------------------------------------------------
# Location parsing: laterality x region -> zone names
# --------------------------------------------------------------------------
_LOC_RE = re.compile(
    r"\b(?:(right|left|bilateral(?:ly)?|bibasilar|bibasal|both)\s+)?"
    r"(?:(upper|middle|mid|lower)\s+)?"
    r"(lobes?|lungs?|lung\s+fields?|lung\s+bases?|bases?|basilar|basal|apex|apices|apical|"
    r"midlung|mid\s+lung|hil(?:um|a|ar)|perihilar|costophrenic(?:\s+angles?)?|"
    r"hemidiaphragms?|diaphragms?|pleural|lingula(?:r)?|retrocardiac|"
    r"heart|cardiac(?:\s+silhouette)?|cardiomediastinal|mediastin(?:um|al)|aort(?:a|ic)|trachea)\b",
    re.I,
)
# also bare laterality words like "right-sided effusion" / "on the left"
_SIDE_RE = re.compile(r"\b(right|left|bilateral(?:ly)?|bibasilar)\b", re.I)

ZONES_ALL_LUNG = ["right_upper_lobe", "right_middle_lobe", "right_lower_lobe",
                  "left_upper_lobe", "left_middle_zone", "left_lower_lobe"]


def _sides(lat: Optional[str]) -> List[str]:
    if not lat:
        return ["right", "left"]
    lat = lat.lower()
    if lat.startswith("right"):
        return ["right"]
    if lat.startswith("left"):
        return ["left"]
    return ["right", "left"]


def location_to_zones(lat: Optional[str], level: Optional[str], region: str) -> List[str]:
    region = region.lower()
    sides = _sides(lat)
    lv = (level or "").lower()
    z: List[str] = []
    if region.startswith(("lobe", "lung", "midlung", "mid lung")):
        for s in sides:
            if region.startswith(("midlung", "mid lung")) or lv in ("middle", "mid"):
                z.append(f"{s}_middle_lobe" if s == "right" else "left_middle_zone")
            elif lv == "upper":
                z.append(f"{s}_upper_lobe")
            elif lv == "lower":
                z.append(f"{s}_lower_lobe")
            else:
                z += [f"{s}_upper_lobe", f"{s}_middle_lobe" if s == "right" else "left_middle_zone", f"{s}_lower_lobe"]
        if "base" in region:
            z = [f"{s}_lower_lobe" for s in sides]
    elif region.startswith(("base", "basilar", "basal")) or lat and lat.lower().startswith(("bibas",)):
        z = [f"{s}_lower_lobe" for s in sides]
    elif region.startswith(("apex", "apices", "apical")):
        z = [f"{s}_apex" for s in sides]
    elif region.startswith(("hil", "perihilar")):
        z = [f"{s}_hilum" for s in sides]
    elif region.startswith("costophrenic"):
        z = [f"{s}_costophrenic_angle" for s in sides]
    elif region.startswith(("hemidiaphragm", "diaphragm")):
        z = [f"{s}_hemidiaphragm" for s in sides]
    elif region.startswith("pleural"):
        z = [f"{s}_costophrenic_angle" for s in sides] + [f"{s}_lower_lobe" for s in sides]
    elif region.startswith("lingula"):
        z = ["left_middle_zone"]
    elif region.startswith("retrocardiac"):
        z = ["left_lower_lobe", "cardiac_silhouette"]
    elif region.startswith(("heart", "cardiac")) and not region.startswith("cardiomediastinal"):
        z = ["cardiac_silhouette"]
    elif region.startswith("cardiomediastinal"):
        z = ["cardiac_silhouette", "mediastinum"]
    elif region.startswith(("mediastin", "aort", "trachea")):
        z = ["mediastinum"]
    return list(dict.fromkeys(z))


def default_zones(name: Optional[str]) -> List[str]:
    if name is None:
        return []
    return {
        "lungs": ZONES_ALL_LUNG,
        "costophrenic": ["right_costophrenic_angle", "left_costophrenic_angle"],
        "pleural": ["right_costophrenic_angle", "left_costophrenic_angle", "right_lower_lobe", "left_lower_lobe"],
        "heart": ["cardiac_silhouette"],
        "mediastinum": ["mediastinum"],
        "cardiomediastinal": ["cardiac_silhouette", "mediastinum"],
    }[name]


# --------------------------------------------------------------------------
# Negation / uncertainty (NegEx-lite, clause scoped)
# --------------------------------------------------------------------------
_PRE_NEG = re.compile(r"\b(?:no|not|without|negative\s+for|free\s+of|clear\s+of|absence\s+of|"
                      r"resolution\s+of|resolved|removed|rules?\s+out|ruled\s+out|nor)\b", re.I)
_POST_NEG = re.compile(r"^\W*(?:\w+\W+){0,3}?(?:is|are|was|were)\s+(?:not\s+(?:seen|identified|present|visualized|evident|demonstrated|appreciated)|absent|resolved)", re.I)
_UNCERTAIN = re.compile(r"\b(?:possib(?:le|ly)|probab(?:le|ly)|likely|may|might|could|questionable|represent\w*|indicat\w*|"
                        r"cannot\s+(?:be\s+)?(?:excluded|ruled\s+out)|can\s?not\s+exclude|suggest\w*|"
                        r"concerning\s+for|versus|vs\.?|equivocal|suspicious|borderline)\b", re.I)
_RELATIONAL = re.compile(r"\b(?:interval|compared|comparison|prior|previous(?:ly)?|since|"
                         r"increas\w*|decreas\w*|improv\w*|worsen\w*|unchanged|stable|new|resolv\w*|persistent)\b", re.I)
_CLAUSE_BREAK = re.compile(r"[;,:]|\b(?:but|however|although|though|whereas|except)\b", re.I)


def _clause_window(text: str, start: int, end: int) -> Tuple[str, str]:
    """Text before/after the mention, cut at clause boundaries."""
    before = text[:start]
    m = list(_CLAUSE_BREAK.finditer(before))
    if m:
        before = before[m[-1].end():]
    after = text[end:]
    m2 = _CLAUSE_BREAK.search(after)
    if m2:
        after = after[:m2.start()]
    return before, after


def polarity(text: str, start: int, end: int) -> Tuple[bool, bool]:
    """Return (negated, uncertain) for the mention at text[start:end]."""
    before, after = _clause_window(text, start, end)
    words_before = before.split()[-7:]
    neg = bool(_PRE_NEG.search(" ".join(words_before))) or bool(_POST_NEG.search(after))
    unc = bool(_UNCERTAIN.search(before)) or bool(_UNCERTAIN.search(after))
    return neg, unc


# --------------------------------------------------------------------------
# Claim extraction
# --------------------------------------------------------------------------
@dataclass
class Claim:
    sentence_idx: int
    sentence: str
    sent_start: int
    sent_end: int
    kind: str                 # "abnormal" | "normal"
    finding: str              # canonical finding or "normal"
    mention: str
    tier: str                 # critical | moderate | mild
    negated: bool
    uncertain: bool
    relational: bool          # temporal/comparison language without a prior study
    location_text: str
    laterality: Optional[str]
    zones: List[str] = field(default_factory=list)

    @property
    def localizable(self) -> bool:
        return len(self.zones) > 0

    def to_dict(self):
        d = asdict(self)
        d["localizable"] = self.localizable
        return d


def _locations(sentence: str):
    out = []
    for m in _LOC_RE.finditer(sentence):
        lat, lvl, reg = m.group(1), m.group(2), m.group(3)
        zones = location_to_zones(lat, lvl, reg)
        if zones:
            out.append((m.start(), m.end(), m.group(0), lat, zones))
    return out


def extract_claims(report: str) -> List[Claim]:
    claims: List[Claim] = []
    for si, s in enumerate(split_sentences(report)):
        t = s.text
        locs = _locations(t)
        found_any = False
        taken: List[Tuple[int, int]] = []
        for name, rx in _FINDING_RE.items():
            for m in rx.finditer(t):
                if any(not (m.end() <= a or m.start() >= b) for a, b in taken):
                    continue  # overlapping a longer/earlier match
                taken.append((m.start(), m.end()))
                found_any = True
                neg, unc = polarity(t, m.start(), m.end())
                # nearest location within the same clause
                zones, loc_text, lat = [], "", None
                best = None
                for (a, b, txt, la, zz) in locs:
                    lo, hi = (min(b, m.start()), max(a, m.end()))
                    between = t[lo:hi]
                    if _CLAUSE_BREAK.search(between) and not (a <= m.start() <= b):
                        continue
                    d = 0 if (a <= m.start() < b or m.start() <= a < m.end()) else min(abs(a - m.end()), abs(m.start() - b))
                    if best is None or d < best[0]:
                        best = (d, txt, la, zz)
                if best is not None:
                    _, loc_text, lat, zones = best
                else:
                    side = _SIDE_RE.search(t)
                    dz = default_zones(DEFAULT_LOC[name])
                    if side and side.group(1).lower().startswith("bibas"):
                        dz, lat = ["right_lower_lobe", "left_lower_lobe"], "bibasilar"
                    elif side and dz:
                        keep = _sides(side.group(1))
                        dz = [z for z in dz if z.split("_")[0] in keep] or dz
                        lat = side.group(1)
                    zones, loc_text = dz, (DEFAULT_LOC[name] or "")
                rel = bool(_RELATIONAL.search(t))
                claims.append(Claim(si, t, s.start, s.end, "abnormal", name, m.group(0), TIER[name],
                                    neg, unc, rel, loc_text, lat, zones))
        if not found_any:
            for rx, organ in NORMAL_PATTERNS:
                m = rx.search(t)
                if m:
                    claims.append(Claim(si, t, s.start, s.end, "normal", "normal", m.group(0), "mild",
                                        False, False, False, organ, None, default_zones(organ)))
                    break
    return claims
