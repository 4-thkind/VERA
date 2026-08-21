"""
VERA Alignment Scorer
Computes VERA scores for hallucination detection.

VERA(c) = Σ a(i,j) for (i,j) ∈ R(c)  /  Σ a(i,j) for all (i,j)

If VERA(c) < T(c) → flag claim c as likely hallucination
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json


# ============================================================
# Severity Classification
# ============================================================

SEVERITY_THRESHOLDS = {
    "critical": 0.35,
    "moderate": 0.25,
    "mild": 0.15,
}

SEVERITY_LOOKUP = {
    # Critical findings
    "mass": "critical", "masses": "critical",
    "tumor": "critical", "tumour": "critical",
    "nodule": "critical", "nodules": "critical",
    "pneumothorax": "critical",
    "effusion": "critical", "pleural effusion": "critical",
    "pneumonia": "critical",
    "edema": "critical", "pulmonary edema": "critical",
    "fracture": "critical",
    # Moderate findings
    "consolidation": "moderate",
    "opacity": "moderate", "opacities": "moderate",
    "opacification": "moderate",
    "infiltrate": "moderate", "infiltrates": "moderate",
    "atelectasis": "moderate",
    "fibrosis": "moderate", "scarring": "moderate",
    "thickening": "moderate",
    "congestion": "moderate", "vascular congestion": "moderate",
    "widening": "moderate", "mediastinal widening": "moderate",
    "calcification": "moderate", "calcifications": "moderate",
    "granuloma": "moderate", "granulomas": "moderate",
    "cardiomegaly": "moderate", "enlarged heart": "moderate",
    "hyperinflation": "moderate",
    # Mild / Normal
    "clear": "mild", "normal": "mild",
    "unremarkable": "mild", "stable": "mild",
    "no acute": "mild",
    "scoliosis": "mild", "kyphosis": "mild",
    "degenerative": "mild",
    "prominence": "mild", "tortuous": "mild",
    "blunting": "mild", "haziness": "moderate",
    "density": "moderate",
}


def classify_severity(finding: str) -> str:
    """
    Classify a finding into a severity tier.
    
    Returns: "critical", "moderate", or "mild"
    """
    finding_lower = finding.lower().strip()

    # Exact match
    if finding_lower in SEVERITY_LOOKUP:
        return SEVERITY_LOOKUP[finding_lower]

    # Partial match
    for term, severity in SEVERITY_LOOKUP.items():
        if term in finding_lower or finding_lower in term:
            return severity

    # Default to moderate (conservative)
    return "moderate"


def get_threshold(finding: str, thresholds: Dict[str, float] = None) -> float:
    """Get the VERA threshold for a finding based on its severity."""
    if thresholds is None:
        thresholds = SEVERITY_THRESHOLDS
    severity = classify_severity(finding)
    return thresholds.get(severity, thresholds.get("moderate", 0.25))


# ============================================================
# VERA Score Computation
# ============================================================

def compute_vera_score(
    attention_map: np.ndarray,
    region_mask: np.ndarray,
) -> float:
    """
    Compute VERA alignment score for a single claim.
    
    VERA(c) = sum(attention[R(c)]) / sum(attention[all])
    
    Args:
        attention_map: Attention map of shape (H, W) — normalized or unnormalized
        region_mask: Binary mask of shape (H, W) — 1 where region is active
    
    Returns:
        VERA score in [0, 1]. Higher = more aligned (less likely hallucination).
    """
    # Ensure same shape
    if attention_map.shape != region_mask.shape:
        # Try to resize mask to match attention
        from scipy.ndimage import zoom
        zoom_factors = (
            attention_map.shape[0] / region_mask.shape[0],
            attention_map.shape[1] / region_mask.shape[1],
        )
        region_mask = zoom(region_mask, zoom_factors, order=0)

    # Compute attention in region vs. total
    attention_in_region = (attention_map * region_mask).sum()
    attention_total = attention_map.sum()

    if attention_total <= 0:
        return 0.0

    vera_score = float(attention_in_region / attention_total)
    return vera_score


def compute_attention_entropy(attention_map: np.ndarray) -> float:
    """
    Compute entropy of the attention distribution.
    
    High entropy = diffuse attention (less certain)
    Low entropy = focused attention (more certain)
    
    Used for severity hallucination detection (Section 5.2 extension).
    """
    # Flatten and normalize
    attn_flat = attention_map.flatten().astype(np.float64)
    attn_flat = attn_flat / (attn_flat.sum() + 1e-10)

    # Compute entropy
    entropy = -np.sum(attn_flat * np.log2(attn_flat + 1e-10))
    return float(entropy)


def score_single_claim(
    claim: Dict,
    attention_maps: np.ndarray,
    patch_grid: Tuple[int, int] = (24, 24),
    thresholds: Dict[str, float] = None,
) -> Dict:
    """
    Score a single claim using VERA.
    
    Args:
        claim: Dict with 'finding', 'location', 'token_span', etc.
        attention_maps: Array of shape [num_tokens, H, W]
        patch_grid: Vision encoder patch grid
        thresholds: Severity-calibrated thresholds
    
    Returns:
        Claim dict augmented with VERA fields
    """
    from src.anatomy_atlas import claim_to_region

    # Make a copy to avoid mutating input
    scored_claim = dict(claim)
    scored_claim["vera_score"] = None
    scored_claim["vera_flagged"] = False
    scored_claim["vera_threshold"] = None
    scored_claim["attention_entropy"] = None
    scored_claim["localizable"] = False

    location = claim.get("location", "")
    finding = claim.get("finding", "")

    # Get region mask for this claim's location
    if not location:
        scored_claim["skip_reason"] = "no_location"
        return scored_claim

    region_mask = claim_to_region(location, patch_grid)
    if region_mask is None:
        scored_claim["skip_reason"] = "unmappable_location"
        return scored_claim

    scored_claim["localizable"] = True

    # Get attention map for this claim's tokens
    token_span = claim.get("token_span", (0, 0))
    start_tok, end_tok = token_span
    num_maps = len(attention_maps)

    if (start_tok >= end_tok or end_tok > num_maps) and "char_span" in claim and "sentence" in claim:
        # Map character span in report to token indices proportionally
        char_span = claim.get("char_span", (0, 0))
        report_text = claim.get("sentence", "")
        if char_span[1] > char_span[0] and len(report_text) > 0 and num_maps > 1:
            ratio_start = max(0.0, min(1.0, char_span[0] / len(report_text)))
            ratio_end = max(0.0, min(1.0, char_span[1] / len(report_text)))
            start_tok = int(ratio_start * num_maps)
            end_tok = max(start_tok + 1, int(ratio_end * num_maps))
            # Expand slightly by 1 token for context
            start_tok = max(0, start_tok - 1)
            end_tok = min(num_maps, end_tok + 1)

    if start_tok < end_tok and end_tok <= num_maps:
        # Claim-specific token attention
        claim_attention = attention_maps[start_tok:end_tok].mean(axis=0)
    else:
        # Fallback to report-level mean attention
        claim_attention = attention_maps.mean(axis=0)

    # Compute VERA score
    vera_score = compute_vera_score(claim_attention, region_mask)
    scored_claim["vera_score"] = vera_score

    # Get severity tier and threshold
    severity = classify_severity(finding)
    scored_claim["severity_tier"] = severity
    threshold = get_threshold(finding, thresholds)
    scored_claim["vera_threshold"] = threshold

    # Flag as hallucination if below threshold
    scored_claim["vera_flagged"] = vera_score < threshold

    # Compute attention entropy
    scored_claim["attention_entropy"] = compute_attention_entropy(claim_attention)

    return scored_claim


def score_all_claims(
    claims: List[Dict],
    attention_maps: np.ndarray,
    patch_grid: Tuple[int, int] = (24, 24),
    thresholds: Dict[str, float] = None,
) -> List[Dict]:
    """
    Score all claims for an image using VERA.
    
    Args:
        claims: List of claim dicts
        attention_maps: Array of shape [num_tokens, H, W]
        patch_grid: Vision encoder patch grid
        thresholds: Severity-calibrated thresholds
    
    Returns:
        List of scored claim dicts
    """
    scored_claims = []
    for claim in claims:
        scored = score_single_claim(claim, attention_maps, patch_grid, thresholds)
        scored_claims.append(scored)
    return scored_claims


# ============================================================
# Threshold Calibration
# ============================================================

def calibrate_thresholds(
    all_scores: List[Dict],
    ground_truth: List[bool],
    severity_tiers: List[str] = None,
    threshold_range: Tuple[float, float] = (0.05, 0.60),
    threshold_step: float = 0.05,
) -> Dict[str, float]:
    """
    Calibrate VERA thresholds on validation data.
    
    Grid search over threshold values per severity tier to maximize F1.
    
    Args:
        all_scores: List of dicts with 'vera_score' and 'severity_tier'
        ground_truth: List of bool (True = hallucination)
        severity_tiers: Optional list of tiers to calibrate
        threshold_range: (min, max) threshold values to search
        threshold_step: Step size for grid search
    
    Returns:
        Dict mapping severity tier → optimal threshold
    """
    from sklearn.metrics import f1_score

    if severity_tiers is None:
        severity_tiers = ["critical", "moderate", "mild"]

    thresholds = np.arange(
        threshold_range[0], threshold_range[1] + threshold_step, threshold_step
    )

    optimal = {}
    for tier in severity_tiers:
        # Get indices for this tier
        tier_indices = [
            i for i, s in enumerate(all_scores)
            if s.get("severity_tier") == tier and s.get("vera_score") is not None
        ]

        if len(tier_indices) < 5:
            print(f"  [WARN] Too few samples for tier '{tier}' ({len(tier_indices)}). Using default.")
            optimal[tier] = SEVERITY_THRESHOLDS.get(tier, 0.25)
            continue

        tier_scores = [all_scores[i]["vera_score"] for i in tier_indices]
        tier_gt = [ground_truth[i] for i in tier_indices]

        best_f1 = 0
        best_threshold = SEVERITY_THRESHOLDS.get(tier, 0.25)

        for t in thresholds:
            predictions = [score < t for score in tier_scores]
            if sum(predictions) == 0 or sum(predictions) == len(predictions):
                continue
            f1 = f1_score(tier_gt, predictions, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = round(float(t), 3)

        optimal[tier] = best_threshold
        print(f"  Tier '{tier}': optimal threshold = {best_threshold:.3f} (F1 = {best_f1:.3f})")

    return optimal


# ============================================================
# Batch Scoring
# ============================================================

def score_dataset(
    inference_results: List[Dict],
    claims_dir: str,
    attention_dir: str,
    patch_grid: Tuple[int, int] = (24, 24),
    thresholds: Dict[str, float] = None,
) -> List[Dict]:
    """
    Score all claims across the dataset.
    
    Args:
        inference_results: List of dicts from model inference
        claims_dir: Directory with claims JSON files
        attention_dir: Directory with attention .npz files
        patch_grid: Vision encoder patch grid
        thresholds: Severity-calibrated thresholds
    
    Returns:
        List of per-image result dicts with scored claims
    """
    from tqdm import tqdm

    claims_dir = Path(claims_dir)
    attention_dir = Path(attention_dir)
    all_results = []

    for entry in tqdm(inference_results, desc="VERA scoring"):
        image_id = entry.get("image_id", "")
        if entry.get("error"):
            continue

        # Load claims
        claims_path = claims_dir / f"{image_id}_claims.json"
        if not claims_path.exists():
            continue
        with open(claims_path, "r") as f:
            claims = json.load(f)

        # Load attention maps
        attn_path = attention_dir / f"{image_id}_attention.npz"
        if not attn_path.exists():
            continue
        attn_data = np.load(str(attn_path))
        attention_maps = attn_data["attention_maps"]

        # Score claims
        scored_claims = score_all_claims(claims, attention_maps, patch_grid, thresholds)

        result = {
            "image_id": image_id,
            "generated_report": entry.get("generated_report", ""),
            "claims": scored_claims,
            "num_claims": len(scored_claims),
            "num_flagged": sum(1 for c in scored_claims if c.get("vera_flagged")),
            "num_localizable": sum(1 for c in scored_claims if c.get("localizable")),
        }
        all_results.append(result)

    print(f"\n  Scored {len(all_results)} images")
    total_claims = sum(r["num_claims"] for r in all_results)
    total_flagged = sum(r["num_flagged"] for r in all_results)
    total_localizable = sum(r["num_localizable"] for r in all_results)
    print(f"  Total claims: {total_claims}")
    print(f"  Localizable: {total_localizable} ({total_localizable/total_claims*100:.1f}%)" if total_claims > 0 else "")
    print(f"  Flagged: {total_flagged} ({total_flagged/total_claims*100:.1f}%)" if total_claims > 0 else "")

    return all_results
