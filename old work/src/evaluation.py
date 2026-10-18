"""
VERA Evaluation
Metrics computation, baseline implementations, and paper-quality plotting.
"""
import importlib.metadata
_orig = importlib.metadata.version
importlib.metadata.version = lambda pkg: '0.19.1' if pkg == 'tokenizers' else _orig(pkg)

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# ============================================================
# Ground Truth Labeling via NLI
# ============================================================

def load_nli_model(model_id: str = "cross-encoder/nli-deberta-v3-base"):
    """Load a cross-encoder NLI model for entailment checking."""
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(model_id, max_length=512)
        print(f"  Loaded NLI model: {model_id}")
        return model
    except ImportError:
        print("  [WARN] sentence-transformers not installed. NLI-based ground truth unavailable.")
        return None


def check_entailment(
    nli_model,
    premise: str,
    hypothesis: str,
) -> Dict[str, float]:
    """
    Check if the premise entails the hypothesis.
    
    Returns:
        Dict with 'entailment', 'contradiction', 'neutral' probabilities
    """
    scores = nli_model.predict([(premise, hypothesis)])
    # CrossEncoder returns [contradiction, entailment, neutral] or similar
    # The order depends on the model; nli-deberta-v3-base returns:
    # [contradiction, entailment, neutral]
    if len(scores.shape) == 1:
        scores = scores.reshape(1, -1)

    labels = ["contradiction", "entailment", "neutral"]
    result = {label: float(score) for label, score in zip(labels, scores[0])}
    return result


def compute_ground_truth_nli(
    generated_claims: List[Dict],
    reference_report: str,
    nli_model,
    entailment_threshold: float = 0.5,
) -> List[bool]:
    """
    Use NLI to determine ground truth hallucination labels.
    
    A claim is labeled as hallucination if the reference report
    does NOT entail it (i.e., entailment score < threshold).
    
    Args:
        generated_claims: List of claim dicts with 'sentence' or 'finding'+'location'
        reference_report: Original radiologist report
        nli_model: Cross-encoder NLI model
        entailment_threshold: Threshold for entailment (default 0.5)
    
    Returns:
        List of booleans: True = hallucination, False = supported
    """
    labels = []
    for claim in generated_claims:
        # Construct hypothesis from claim
        finding = claim.get("finding", "")
        location = claim.get("location", "")
        negated = claim.get("negated", False)

        if negated:
            hypothesis = f"There is no {finding}"
            if location:
                hypothesis += f" in the {location}"
        else:
            hypothesis = f"There is {finding}"
            if location:
                hypothesis += f" in the {location}"

        hypothesis += "."

        # Check entailment
        if nli_model is not None:
            result = check_entailment(nli_model, reference_report, hypothesis)
            is_hallucination = result.get("entailment", 0) < entailment_threshold
        else:
            # Fallback: simple keyword matching
            ref_lower = reference_report.lower()
            is_hallucination = finding.lower() not in ref_lower
        
        labels.append(is_hallucination)

    return labels


# ============================================================
# Metrics
# ============================================================

def compute_metrics(
    predictions: List[bool],
    ground_truth: List[bool],
) -> Dict[str, float]:
    """
    Compute precision, recall, F1 for hallucination detection.
    
    Args:
        predictions: List of bool (True = flagged as hallucination)
        ground_truth: List of bool (True = actual hallucination)
    
    Returns:
        Dict with precision, recall, f1, accuracy, num_total, etc.
    """
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, accuracy_score,
        confusion_matrix,
    )

    preds_array = np.array(predictions, dtype=int)
    gt_array = np.array(ground_truth, dtype=int)

    metrics = {
        "precision": float(precision_score(gt_array, preds_array, zero_division=0)),
        "recall": float(recall_score(gt_array, preds_array, zero_division=0)),
        "f1": float(f1_score(gt_array, preds_array, zero_division=0)),
        "accuracy": float(accuracy_score(gt_array, preds_array)),
        "num_total": len(predictions),
        "num_positive_pred": int(preds_array.sum()),
        "num_positive_gt": int(gt_array.sum()),
        "hallucination_rate_gt": float(gt_array.mean()),
        "flag_rate_pred": float(preds_array.mean()),
    }

    # Confusion matrix
    if len(set(gt_array)) > 1:
        cm = confusion_matrix(gt_array, preds_array)
        metrics["true_positives"] = int(cm[1, 1])
        metrics["false_positives"] = int(cm[0, 1])
        metrics["true_negatives"] = int(cm[0, 0])
        metrics["false_negatives"] = int(cm[1, 0])

    return metrics


def compute_per_severity_metrics(
    scored_claims: List[Dict],
    ground_truth: List[bool],
) -> Dict[str, Dict[str, float]]:
    """Compute metrics per severity tier."""
    tiers = defaultdict(lambda: {"preds": [], "gts": []})

    for claim, gt in zip(scored_claims, ground_truth):
        tier = claim.get("severity_tier", "unknown")
        flagged = claim.get("vera_flagged", False)
        tiers[tier]["preds"].append(flagged)
        tiers[tier]["gts"].append(gt)

    results = {}
    for tier, data in tiers.items():
        results[tier] = compute_metrics(data["preds"], data["gts"])
        results[tier]["num_claims"] = len(data["preds"])

    return results


def compute_per_severity_auroc(
    scored_claims: List[Dict],
    ground_truth: List[bool],
    score_key: str = "vera_score_norm",
) -> Dict[str, float]:
    """
    Compute AUROC per severity tier.
    
    Groups claims by severity tier and computes AUROC separately for each,
    revealing whether ranking signal is strong in specific tiers but
    washed out by pooling across all claims.
    
    Args:
        scored_claims: List of claim dicts with vera scores and severity_tier
        ground_truth: List of bool (True = hallucination)
        score_key: Which score field to use (default: area-normalized)
    
    Returns:
        Dict mapping tier name -> AUROC value (or None if insufficient data)
    """
    from sklearn.metrics import roc_auc_score

    tiers = defaultdict(lambda: {"scores": [], "gts": []})

    for claim, gt in zip(scored_claims, ground_truth):
        tier = claim.get("severity_tier", "unknown")
        score = claim.get(score_key)
        if score is not None:
            tiers[tier]["scores"].append(score)
            tiers[tier]["gts"].append(gt)

    results = {}
    for tier, data in tiers.items():
        gt_array = np.array(data["gts"], dtype=int)
        # Need both classes present and at least 5 samples
        if len(set(gt_array)) < 2 or len(gt_array) < 5:
            results[tier] = None
            continue
        # Invert scores: lower VERA = higher hallucination risk
        risk_scores = 1.0 - np.array(data["scores"], dtype=float)
        try:
            results[tier] = float(roc_auc_score(gt_array, risk_scores))
        except ValueError:
            results[tier] = None

    return results


# ============================================================
# Baselines
# ============================================================

def random_baseline(
    n_claims: int,
    hallucination_rate: float = 0.3,
    seed: int = 42,
) -> List[bool]:
    """Random baseline: flag claims randomly at the observed hallucination rate."""
    np.random.seed(seed)
    return list(np.random.random(n_claims) < hallucination_rate)


def confidence_baseline(
    generated_logits: List[float],
    threshold: float = 0.5,
) -> List[bool]:
    """
    Confidence baseline: flag claims whose generation tokens
    have low average probability.
    """
    return [logit < threshold for logit in generated_logits]


def nli_only_baseline(
    generated_claims: List[Dict],
    reference_reports: List[str],
    nli_model,
    threshold: float = 0.5,
) -> List[bool]:
    """
    NLI-only baseline: flag claims not entailed by reference report.
    This requires reference text (unlike VERA).
    """
    all_labels = []
    for claims, ref in zip(generated_claims, reference_reports):
        labels = compute_ground_truth_nli(claims, ref, nli_model, threshold)
        all_labels.extend(labels)
    return all_labels


# ============================================================
# ROC / AUROC
# ============================================================

def compute_roc(
    vera_scores: List[float],
    ground_truth: List[bool],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute ROC curve and AUROC for VERA scores.
    
    Note: lower VERA score → more likely hallucination,
    so we invert for ROC (higher = more suspicious).
    """
    from sklearn.metrics import roc_curve, auc

    gt_array = np.array(ground_truth, dtype=int)
    # Invert scores: lower VERA = higher hallucination risk
    risk_scores = 1.0 - np.array(vera_scores, dtype=float)

    fpr, tpr, thresholds = roc_curve(gt_array, risk_scores)
    auroc = auc(fpr, tpr)

    return fpr, tpr, auroc


# ============================================================
# Paper-Quality Plotting
# ============================================================

def setup_plot_style():
    """Configure matplotlib for paper-quality plots."""
    import matplotlib.pyplot as plt
    import matplotlib

    plt.style.use('seaborn-v0_8-whitegrid')
    matplotlib.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.figsize': (8, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def plot_vera_distribution(
    vera_scores_hallucinated: List[float],
    vera_scores_clean: List[float],
    save_path: str = None,
    title: str = "VERA Score Distribution",
):
    """
    Plot VERA score distribution for hallucinated vs. clean claims.
    (Paper Figure 2)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot distributions
    if vera_scores_clean:
        sns.histplot(
            vera_scores_clean, bins=30, alpha=0.6, color="#2ecc71",
            label=f"Clean Claims (n={len(vera_scores_clean)})",
            stat="density", kde=True, ax=ax
        )
    if vera_scores_hallucinated:
        sns.histplot(
            vera_scores_hallucinated, bins=30, alpha=0.6, color="#e74c3c",
            label=f"Hallucinated Claims (n={len(vera_scores_hallucinated)})",
            stat="density", kde=True, ax=ax
        )

    ax.set_xlabel("VERA Score", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.legend(fontsize=12)
    ax.axvline(x=0.25, color='gray', linestyle='--', alpha=0.5, label="Default Threshold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auroc: float,
    save_path: str = None,
    title: str = "VERA ROC Curve",
):
    """Plot ROC curve with AUROC. (Paper supplementary)"""
    import matplotlib.pyplot as plt

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(fpr, tpr, color="#3498db", lw=2, label=f"VERA (AUROC = {auroc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.legend(loc="lower right", fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()


def plot_threshold_sensitivity(
    vera_scores: List[float],
    ground_truth: List[bool],
    severity_tiers: List[str],
    save_path: str = None,
    title: str = "Threshold Sensitivity Analysis",
):
    """
    Plot F1 vs. threshold per severity tier.
    (Paper Figure 3)
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import f1_score

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    thresholds = np.arange(0.05, 0.65, 0.02)
    colors = {"critical": "#e74c3c", "moderate": "#f39c12", "mild": "#2ecc71"}
    markers = {"critical": "o", "moderate": "s", "mild": "^"}

    unique_tiers = sorted(set(severity_tiers))
    for tier in unique_tiers:
        tier_mask = [t == tier for t in severity_tiers]
        tier_scores = [s for s, m in zip(vera_scores, tier_mask) if m]
        tier_gt = [g for g, m in zip(ground_truth, tier_mask) if m]

        if len(tier_scores) < 5:
            continue

        f1_values = []
        for t in thresholds:
            preds = [s < t for s in tier_scores]
            if sum(preds) == 0 or sum(preds) == len(preds):
                f1_values.append(0)
            else:
                f1_values.append(f1_score(tier_gt, preds, zero_division=0))

        color = colors.get(tier, "#3498db")
        marker = markers.get(tier, "o")
        ax.plot(
            thresholds, f1_values,
            color=color, marker=marker, markevery=5, markersize=6,
            label=f"{tier.capitalize()} (n={len(tier_scores)})", lw=2
        )

    ax.set_xlabel("Threshold T", fontsize=14)
    ax.set_ylabel("F1 Score", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.legend(fontsize=12)
    ax.set_xlim([0.05, 0.60])
    ax.set_ylim([0.0, 1.0])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()


def plot_attention_heatmap(
    image: np.ndarray,
    attention_map: np.ndarray,
    claim_text: str = "",
    vera_score: float = None,
    is_hallucination: bool = None,
    region_mask: np.ndarray = None,
    save_path: str = None,
):
    """
    Plot attention heatmap overlaid on X-ray image.
    (Paper Figure 1 — individual panels)
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from scipy.ndimage import zoom

    setup_plot_style()

    ncols = 3 if region_mask is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))

    # Panel 1: Original image
    axes[0].imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    axes[0].set_title("Original CXR", fontsize=14)
    axes[0].axis("off")

    # Upscale attention map to image size
    H_img, W_img = image.shape[:2]
    H_attn, W_attn = attention_map.shape
    attn_upscaled = zoom(attention_map, (H_img / H_attn, W_img / W_attn), order=1)

    # Panel 2: Attention heatmap overlay
    axes[1].imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    im = axes[1].imshow(attn_upscaled, cmap='jet', alpha=0.5, norm=Normalize())
    title = "Attention Map"
    if vera_score is not None:
        title += f"\nVERA = {vera_score:.3f}"
    if is_hallucination is not None:
        status = "🔴 HALLUCINATION" if is_hallucination else "🟢 CLEAN"
        title += f" | {status}"
    axes[1].set_title(title, fontsize=13)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Region mask (if provided)
    if region_mask is not None:
        mask_upscaled = zoom(region_mask, (H_img / region_mask.shape[0], W_img / region_mask.shape[1]), order=0)
        axes[2].imshow(image, cmap='gray' if len(image.shape) == 2 else None)
        axes[2].imshow(mask_upscaled, cmap='Greens', alpha=0.4)
        axes[2].set_title(f"Expected Region\n({claim_text})", fontsize=13)
        axes[2].axis("off")

    plt.suptitle(f"Claim: \"{claim_text}\"", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()


def plot_comparison_figure(
    image: np.ndarray,
    clean_attention: np.ndarray,
    clean_claim: str,
    clean_vera: float,
    hallucinated_attention: np.ndarray,
    hallucinated_claim: str,
    hallucinated_vera: float,
    save_path: str = None,
):
    """
    Create Figure 1 for the paper: side-by-side clean vs. hallucinated attention.
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom

    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    H_img, W_img = image.shape[:2]

    # Panel 1: Original X-ray
    axes[0].imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    axes[0].set_title("Original Chest X-ray", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Clean claim attention
    attn_clean_up = zoom(clean_attention, (H_img / clean_attention.shape[0], W_img / clean_attention.shape[1]), order=1)
    axes[1].imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    axes[1].imshow(attn_clean_up, cmap='jet', alpha=0.5)
    axes[1].set_title(f"🟢 Clean Claim\n\"{clean_claim}\"\nVERA = {clean_vera:.3f}", fontsize=12)
    axes[1].axis("off")

    # Panel 3: Hallucinated claim attention
    attn_hall_up = zoom(hallucinated_attention, (H_img / hallucinated_attention.shape[0], W_img / hallucinated_attention.shape[1]), order=1)
    axes[2].imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    axes[2].imshow(attn_hall_up, cmap='jet', alpha=0.5)
    axes[2].set_title(f"🔴 Hallucinated Claim\n\"{hallucinated_claim}\"\nVERA = {hallucinated_vera:.3f}", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()


# ============================================================
# Results Table Generation
# ============================================================

def generate_results_table(
    method_results: Dict[str, Dict[str, float]],
    save_path: str = None,
) -> str:
    """
    Generate Table 1 (Main Results) as formatted string and CSV.
    
    Args:
        method_results: Dict mapping method name → metrics dict
    
    Returns:
        Formatted markdown table string
    """
    import pandas as pd

    rows = []
    for method, metrics in method_results.items():
        rows.append({
            "Method": method,
            "Precision": f"{metrics.get('precision', 0) * 100:.1f}%",
            "Recall": f"{metrics.get('recall', 0) * 100:.1f}%",
            "F1": f"{metrics.get('f1', 0) * 100:.1f}%",
            "Requires Labels": metrics.get("requires_labels", "No"),
        })

    df = pd.DataFrame(rows)

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"  Saved: {save_path}")

    # Generate markdown table
    md_table = df.to_markdown(index=False)
    return md_table


def generate_model_comparison_table(
    model_results: Dict[str, Dict[str, float]],
    save_path: str = None,
) -> str:
    """Generate Table 2 (Per Model Comparison)."""
    import pandas as pd

    rows = []
    for model, metrics in model_results.items():
        rows.append({
            "Model": model,
            "VERA F1": f"{metrics.get('f1', 0) * 100:.1f}%",
            "Hallucination Rate": f"{metrics.get('hallucination_rate_gt', 0) * 100:.1f}%",
            "Avg VERA (flagged)": f"{metrics.get('avg_vera_flagged', 0):.3f}",
            "Avg VERA (clean)": f"{metrics.get('avg_vera_clean', 0):.3f}",
        })

    df = pd.DataFrame(rows)

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"  Saved: {save_path}")

    return df.to_markdown(index=False)
