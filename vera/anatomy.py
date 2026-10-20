"""Anatomical zone masks on the visual-token grid.

Two atlases, both returned as soft masks on a G x G grid (G = 32 for CheXagent-2-3b):
  * "rect": the v1 hand-drawn rectangles (normalized image coords), kept for ablation.
  * "seg":  patient-specific zones derived from torchxrayvision's chest anatomy
            segmentation (PSPNet, ChestX-Det targets). This answers the reviewer
            concern that the rectangles were never validated.

Convention: radiological view, patient RIGHT is on the IMAGE LEFT (x < 0.5).
CheXagent resizes the full image to a square without cropping, so normalized
image coordinates map directly onto the token grid.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

RECT_ZONES = {  # (x_min, y_min, x_max, y_max)
    "right_upper_lobe":         (0.08, 0.08, 0.42, 0.38),
    "right_middle_lobe":        (0.12, 0.33, 0.42, 0.55),
    "right_lower_lobe":         (0.12, 0.48, 0.48, 0.82),
    "left_upper_lobe":          (0.58, 0.08, 0.92, 0.38),
    "left_middle_zone":         (0.55, 0.33, 0.88, 0.55),
    "left_lower_lobe":          (0.52, 0.48, 0.88, 0.82),
    "right_apex":               (0.10, 0.05, 0.42, 0.22),
    "left_apex":                (0.58, 0.05, 0.90, 0.22),
    "mediastinum":              (0.35, 0.05, 0.65, 0.65),
    "cardiac_silhouette":       (0.28, 0.35, 0.72, 0.78),
    "right_hilum":              (0.33, 0.22, 0.48, 0.48),
    "left_hilum":               (0.52, 0.22, 0.67, 0.48),
    "right_costophrenic_angle": (0.08, 0.72, 0.32, 0.95),
    "left_costophrenic_angle":  (0.68, 0.72, 0.92, 0.95),
    "right_hemidiaphragm":      (0.08, 0.68, 0.48, 0.88),
    "left_hemidiaphragm":       (0.52, 0.68, 0.92, 0.88),
}
ZONE_NAMES = list(RECT_ZONES)


def rect_masks(G: int) -> Dict[str, np.ndarray]:
    """Soft (fractional-coverage) rectangle masks on a G x G grid."""
    out = {}
    edges = np.linspace(0, 1, G + 1)
    for z, (x0, y0, x1, y1) in RECT_ZONES.items():
        cx = np.clip(np.minimum(edges[1:], x1) - np.maximum(edges[:-1], x0), 0, None) * G
        cy = np.clip(np.minimum(edges[1:], y1) - np.maximum(edges[:-1], y0), 0, None) * G
        out[z] = np.outer(cy, cx).astype(np.float32)
    return out


# --------------------------------------------------------------------------
# Segmentation-derived zones
# --------------------------------------------------------------------------
_SEG_MODEL = None


def _load_seg(device):
    global _SEG_MODEL
    if _SEG_MODEL is None:
        import torchxrayvision as xrv
        _SEG_MODEL = xrv.baseline_models.chestx_det.PSPNet().to(device).eval()
    return _SEG_MODEL


def segment(image_path: str, device="cpu") -> Dict[str, np.ndarray]:
    """Run chest anatomy segmentation; returns {target: bool mask 512x512} on the
    full image resized to a square (same geometry CheXagent sees)."""
    import torch
    import torch.nn.functional as F
    import torchxrayvision as xrv
    from PIL import Image

    img = np.array(Image.open(image_path).convert("L")).astype(np.float32)
    img = xrv.datasets.normalize(img, 255)
    x = torch.from_numpy(img)[None, None].to(device)
    x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
    model = _load_seg(device)
    with torch.no_grad():
        y = torch.sigmoid(model(x))[0].cpu().numpy() > 0.5
    return {t: y[i] for i, t in enumerate(model.targets)}


def _centroid_x(m: np.ndarray) -> float:
    ys, xs = np.nonzero(m)
    return float(xs.mean()) / m.shape[1] if len(xs) else 0.5


def _rows_band(m: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Part of mask m between fractions lo..hi of ITS OWN vertical extent."""
    ys = np.nonzero(m.any(1))[0]
    if len(ys) == 0:
        return np.zeros_like(m)
    top, bot = ys[0], ys[-1] + 1
    a, b = int(top + lo * (bot - top)), int(top + hi * (bot - top))
    out = np.zeros_like(m)
    out[a:b] = m[a:b]
    return out


def _lateral_half(m: np.ndarray, patient_side: str) -> np.ndarray:
    xs = np.nonzero(m.any(0))[0]
    if len(xs) == 0:
        return m
    mid = (xs[0] + xs[-1] + 1) // 2
    out = m.copy()
    if patient_side == "right":   # image left -> lateral = small x
        out[:, mid:] = False
    else:
        out[:, :mid] = False
    return out


def seg_zone_masks(seg: Dict[str, np.ndarray], G: int) -> Optional[Dict[str, np.ndarray]]:
    """Derive the 16 zones from segmentation; None if it does not look frontal."""
    lungs = [seg["Left Lung"], seg["Right Lung"]]
    lungs = [m for m in lungs if m.mean() > 0.02]
    if len(lungs) != 2:
        return None
    lungs.sort(key=_centroid_x)              # image-left first = patient right
    R, L = lungs
    if not (_centroid_x(R) < 0.5 < _centroid_x(L)) or seg["Heart"].mean() < 0.01:
        return None
    hil = [seg["Left Hilus Pulmonis"], seg["Right Hilus Pulmonis"]]
    hil_r = np.logical_or.reduce([h for h in hil if h.any() and _centroid_x(h) < 0.5] or [np.zeros_like(R)])
    hil_l = np.logical_or.reduce([h for h in hil if h.any() and _centroid_x(h) >= 0.5] or [np.zeros_like(R)])
    dia = seg["Facies Diaphragmatica"]
    half = dia.shape[1] // 2
    dia_r, dia_l = dia.copy(), dia.copy()
    dia_r[:, half:] = False
    dia_l[:, :half] = False
    full = {
        "right_upper_lobe": _rows_band(R, 0, 1 / 3),
        "right_middle_lobe": _rows_band(R, 1 / 3, 2 / 3),
        "right_lower_lobe": _rows_band(R, 2 / 3, 1),
        "left_upper_lobe": _rows_band(L, 0, 1 / 3),
        "left_middle_zone": _rows_band(L, 1 / 3, 2 / 3),
        "left_lower_lobe": _rows_band(L, 2 / 3, 1),
        "right_apex": _rows_band(R, 0, 0.2),
        "left_apex": _rows_band(L, 0, 0.2),
        "mediastinum": seg["Mediastinum"] | seg["Aorta"] | seg["Weasand"],
        "cardiac_silhouette": seg["Heart"],
        "right_hilum": hil_r,
        "left_hilum": hil_l,
        "right_costophrenic_angle": _lateral_half(_rows_band(R, 0.8, 1), "right"),
        "left_costophrenic_angle": _lateral_half(_rows_band(L, 0.8, 1), "left"),
        "right_hemidiaphragm": dia_r,
        "left_hemidiaphragm": dia_l,
    }
    return {z: downsample(m, G) for z, m in full.items()}


def downsample(m: np.ndarray, G: int) -> np.ndarray:
    """Fraction of each grid cell covered by mask m (soft mask)."""
    H = m.shape[0]
    k = H // G
    return m[:k * G, :k * G].reshape(G, k, G, k).mean((1, 3)).astype(np.float32)


def union(masks: Dict[str, np.ndarray], zones) -> Optional[np.ndarray]:
    ms = [masks[z] for z in zones if z in masks]
    if not ms:
        return None
    return np.maximum.reduce(ms)
