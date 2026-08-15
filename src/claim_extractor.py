"""
VERA Claim Extractor
Extracts structured anatomical claims from radiology reports using NLP.
"""
import re
from typing import Dict, List, Optional, Tuple
import warnings


# ============================================================
# Medical Term Dictionaries
# ============================================================

# Common radiology findings
FINDING_TERMS = [
    # Critical
    "mass", "masses", "tumor", "tumour", "nodule", "nodules",
    "pneumothorax", "effusion", "pleural effusion",
    "pneumonia", "edema", "pulmonary edema", "fracture",
    # Moderate
    "consolidation", "opacity", "opacities", "opacification",
    "infiltrate", "infiltrates", "atelectasis",
    "fibrosis", "scarring", "thickening",
    "congestion", "vascular congestion",
    "widening", "mediastinal widening",
    "calcification", "calcifications",
    "granuloma", "granulomas",
    "cardiomegaly", "enlarged heart",
    "hyperinflation", "hyperexpansion",
    # Mild / Descriptive
    "clear", "normal", "unremarkable", "stable",
    "scoliosis", "kyphosis", "degenerative",
    "blunting", "haziness", "density",
    "prominence", "tortuous", "unfolded",
    "lucency", "radiolucency",
]

# Anatomical location terms
ANATOMY_TERMS = [
    "right upper lobe", "right middle lobe", "right lower lobe",
    "left upper lobe", "left lower lobe",
    "right lung", "left lung", "lungs", "bilateral",
    "right apex", "left apex", "apex", "apices",
    "right base", "left base", "bases",
    "mediastinum", "mediastinal",
    "cardiac silhouette", "heart", "cardiac",
    "right hilum", "left hilum", "hilum", "hila", "hilar",
    "right costophrenic angle", "left costophrenic angle", "costophrenic",
    "right hemidiaphragm", "left hemidiaphragm", "diaphragm",
    "pleural", "pleural space",
    "aorta", "aortic", "aortic knob",
    "trachea", "spine", "thoracic spine",
    "retrocardiac", "perihilar", "peribronchial",
    "upper lobe", "lower lobe", "middle lobe",
    "right", "left",
]

# Severity qualifiers
SEVERITY_QUALIFIERS = {
    "severe": "critical",
    "large": "critical",
    "significant": "critical",
    "extensive": "critical",
    "massive": "critical",
    "moderate": "moderate",
    "mild": "mild",
    "small": "mild",
    "minimal": "mild",
    "tiny": "mild",
    "trace": "mild",
    "subtle": "mild",
    "slight": "mild",
    "borderline": "mild",
    "possible": "mild",
    "questionable": "mild",
}

# Relational phrases that suggest comparison to prior scans
RELATIONAL_PATTERNS = [
    r"compared?\s+(?:to|with)\s+(?:prior|previous)",
    r"(?:increased?|decreased?|worsened?|improved?)\s+(?:since|from|compared)",
    r"(?:new|interval)\s+(?:change|development|increase|decrease)",
    r"(?:stable|unchanged|persistent)\s+(?:since|from|compared)",
    r"previous(?:ly)?",
    r"prior\s+(?:exam|study|x-?ray|film|radiograph)",
    r"(?:change|changes)\s+(?:since|from|compared)",
    r"(?:worse|better|resolved)\s+(?:than|since|compared)",
]

# Negation patterns
NEGATION_PATTERNS = [
    r"no\s+",
    r"(?:without|absence\s+of|negative\s+for)\s+",
    r"(?:free\s+of|clear\s+of|devoid\s+of)\s+",
    r"(?:denies|denied|exclude[sd]?)\s+",
    r"(?:unlikely|improbable)\s+",
    r"rule(?:d|\s+)out\s+",
]


# ============================================================
# NLP Model Loading
# ============================================================

def load_nlp_model(model_name: str = "en_core_sci_sm"):
    """
    Load scispaCy NLP model for medical NER.
    
    Falls back to standard spaCy model if scispaCy not installed.
    """
    try:
        import spacy
        try:
            nlp = spacy.load(model_name)
            print(f"  Loaded scispaCy model: {model_name}")
        except OSError:
            print(f"  [WARN] {model_name} not found. Trying en_core_web_sm...")
            try:
                nlp = spacy.load("en_core_web_sm")
                print("  Loaded fallback model: en_core_web_sm")
            except OSError:
                print("  [WARN] No spaCy model found. Using rule-based extraction only.")
                return None
        return nlp
    except ImportError:
        print("  [WARN] spaCy not installed. Using rule-based extraction only.")
        return None


# ============================================================
# Claim Extraction — Core Pipeline
# ============================================================

def split_into_sentences(text: str) -> List[str]:
    """Split report text into sentences."""
    # Simple sentence splitter for radiology reports
    # Handles periods, newlines, and numbered lists
    text = re.sub(r'\n+', '. ', text)
    text = re.sub(r'\s+', ' ', text)
    # Split on period followed by space and uppercase, or period at end
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text)
    # Also split on numbered items
    expanded = []
    for sent in sentences:
        sub_sents = re.split(r'(?<=\.)\s*(\d+[\.\)])', sent)
        expanded.extend(s.strip() for s in sub_sents if s.strip())
    return expanded


def extract_findings_from_sentence(sentence: str) -> List[str]:
    """Extract medical findings from a single sentence."""
    sentence_lower = sentence.lower()
    found = []

    # Sort finding terms by length (longest first) to avoid partial matches
    sorted_terms = sorted(FINDING_TERMS, key=len, reverse=True)

    for term in sorted_terms:
        # Word boundary match
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, sentence_lower):
            # Check it's not a substring of already found term
            if not any(term in f and term != f for f in found):
                found.append(term)

    return found


def extract_locations_from_sentence(sentence: str) -> List[str]:
    """Extract anatomical locations from a single sentence."""
    sentence_lower = sentence.lower()
    found = []

    # Sort by length (longest first) to prefer more specific matches
    sorted_terms = sorted(ANATOMY_TERMS, key=len, reverse=True)

    for term in sorted_terms:
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, sentence_lower):
            found.append(term)

    # Deduplicate: if "right upper lobe" is found, don't also include "right" and "upper lobe"
    filtered = []
    for term in found:
        if not any(term != other and term in other for other in found):
            filtered.append(term)

    return filtered


def extract_severity_from_sentence(sentence: str) -> str:
    """Extract severity qualifier from a sentence."""
    sentence_lower = sentence.lower()
    for qualifier, severity in SEVERITY_QUALIFIERS.items():
        if re.search(r'\b' + re.escape(qualifier) + r'\b', sentence_lower):
            return severity
    return "unspecified"


def is_negated(sentence: str, finding: str) -> bool:
    """Check if a finding is negated in the sentence."""
    sentence_lower = sentence.lower()
    for pattern in NEGATION_PATTERNS:
        neg_finding_pattern = pattern + r'.*?' + re.escape(finding)
        if re.search(neg_finding_pattern, sentence_lower):
            return True
    return False


def extract_claims(
    report_text: str,
    nlp_model=None,
    include_negated: bool = True,
) -> List[Dict]:
    """
    Extract structured anatomical claims from a radiology report.
    
    Args:
        report_text: Full report text (findings + impression)
        nlp_model: Optional scispaCy model for entity extraction
        include_negated: Whether to include negated findings (default: True)
    
    Returns:
        List of claim dicts, each containing:
        - finding: str (e.g., "opacity")
        - location: str (e.g., "right lower lobe")
        - severity: str (e.g., "mild", "moderate", "critical", "unspecified")
        - negated: bool
        - sentence: str (source sentence)
        - sentence_idx: int
        - char_span: (start, end) character positions in original text
    """
    if not report_text or not report_text.strip():
        return []

    claims = []
    sentences = split_into_sentences(report_text)

    for sent_idx, sentence in enumerate(sentences):
        if len(sentence.strip()) < 5:
            continue

        # Extract components
        findings = extract_findings_from_sentence(sentence)
        locations = extract_locations_from_sentence(sentence)
        severity = extract_severity_from_sentence(sentence)

        # Also try scispaCy NER if available
        if nlp_model is not None:
            doc = nlp_model(sentence)
            for ent in doc.ents:
                ent_lower = ent.text.lower()
                # Add any new findings from NER
                if ent_lower not in [f.lower() for f in findings]:
                    # Check if it looks like a medical finding
                    for term in FINDING_TERMS:
                        if term in ent_lower or ent_lower in term:
                            findings.append(ent_lower)
                            break

        # If no findings found, check if the whole sentence is a finding
        if not findings:
            # The sentence might describe a normal finding
            sentence_lower = sentence.lower()
            if any(term in sentence_lower for term in ["clear", "normal", "unremarkable", "no acute"]):
                findings = ["normal"]
                if not locations:
                    locations = ["lungs"]  # Default for general normal statements

        # If no location found but we have a finding, try to infer
        if findings and not locations:
            # Some findings have default locations
            for finding in findings:
                if finding in ["cardiomegaly", "enlarged heart"]:
                    locations = ["cardiac"]
                elif finding in ["mediastinal widening"]:
                    locations = ["mediastinum"]

        # Create claims (cross-product of findings × locations)
        if findings and locations:
            for finding in findings:
                negated = is_negated(sentence, finding)
                if not include_negated and negated:
                    continue

                for location in locations:
                    # Find character span of the finding in the full text
                    finding_match = re.search(
                        re.escape(finding), report_text, re.IGNORECASE
                    )
                    char_span = (finding_match.start(), finding_match.end()) if finding_match else (0, 0)

                    claim = {
                        "finding": finding,
                        "location": location,
                        "severity": severity,
                        "negated": negated,
                        "sentence": sentence,
                        "sentence_idx": sent_idx,
                        "char_span": char_span,
                    }
                    claims.append(claim)

        elif findings:
            # Findings without locations (will be unlocalizable in VERA)
            for finding in findings:
                negated = is_negated(sentence, finding)
                if not include_negated and negated:
                    continue
                claim = {
                    "finding": finding,
                    "location": "",  # Empty = unlocalizable
                    "severity": severity,
                    "negated": negated,
                    "sentence": sentence,
                    "sentence_idx": sent_idx,
                    "char_span": (0, 0),
                }
                claims.append(claim)

    return claims


# ============================================================
# Relational Hallucination Detection
# ============================================================

def detect_relational_hallucinations(report_text: str) -> List[Dict]:
    """
    Detect comparative phrases suggesting comparison to prior scans.
    
    These are flagged as potential relational hallucinations if the model
    was only given a single image with no prior scan.
    
    Returns:
        List of dicts with matched relational phrases
    """
    if not report_text:
        return []

    flags = []
    for pattern in RELATIONAL_PATTERNS:
        matches = list(re.finditer(pattern, report_text, re.IGNORECASE))
        for match in matches:
            # Get surrounding context (±30 chars)
            start = max(0, match.start() - 30)
            end = min(len(report_text), match.end() + 30)
            context = report_text[start:end]

            flags.append({
                "type": "relational_hallucination",
                "matched_pattern": match.group(),
                "context": context,
                "char_span": (match.start(), match.end()),
            })

    return flags


# ============================================================
# Token Span Mapping (for attention alignment)
# ============================================================

def map_claims_to_token_spans(
    claims: List[Dict],
    report_text: str,
    tokenizer,
) -> List[Dict]:
    """
    Map each claim's character span to token span in the tokenized report.
    
    This is needed for VERA scoring — we need to know which generated
    tokens correspond to each claim so we can extract the right
    attention maps.
    
    Args:
        claims: List of claim dicts with char_span
        report_text: The generated report text
        tokenizer: HuggingFace tokenizer
    
    Returns:
        Claims with added 'token_span' field: (start_token, end_token)
    """
    # Tokenize the full report
    encoding = tokenizer(report_text, return_offsets_mapping=True)
    offset_mapping = encoding.offset_mapping  # [(char_start, char_end), ...]

    for claim in claims:
        # Find tokens that overlap with the claim's sentence
        sentence = claim.get("sentence", "")
        # Find sentence position in full text
        sent_start = report_text.find(sentence)
        if sent_start == -1:
            claim["token_span"] = (0, 0)
            continue

        sent_end = sent_start + len(sentence)

        # Find token indices that overlap with this sentence
        token_start = None
        token_end = None
        for idx, (cs, ce) in enumerate(offset_mapping):
            if cs is None:
                continue
            if cs >= sent_start and ce <= sent_end:
                if token_start is None:
                    token_start = idx
                token_end = idx + 1

        claim["token_span"] = (
            token_start if token_start is not None else 0,
            token_end if token_end is not None else 0,
        )

    return claims


# ============================================================
# Summary / Statistics
# ============================================================

def summarize_claims(claims: List[Dict]) -> Dict:
    """Generate summary statistics for extracted claims."""
    total = len(claims)
    with_location = sum(1 for c in claims if c.get("location"))
    negated = sum(1 for c in claims if c.get("negated"))
    severity_counts = {}
    for c in claims:
        sev = c.get("severity", "unspecified")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    finding_counts = {}
    for c in claims:
        f = c.get("finding", "unknown")
        finding_counts[f] = finding_counts.get(f, 0) + 1

    return {
        "total_claims": total,
        "with_location": with_location,
        "without_location": total - with_location,
        "negated": negated,
        "affirmed": total - negated,
        "severity_distribution": severity_counts,
        "top_findings": dict(sorted(finding_counts.items(), key=lambda x: -x[1])[:10]),
        "localization_rate": with_location / total * 100 if total > 0 else 0,
    }
