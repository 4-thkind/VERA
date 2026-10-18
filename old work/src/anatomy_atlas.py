"""
VERA Anatomy Atlas
Defines chest anatomy zones and mapping functions for VERA scoring.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
import re


# ============================================================
# Zone Definitions (imported from config, but can be used standalone)
# ============================================================

CHEST_ZONES = {
    "right_upper_lobe":          (0.08, 0.08, 0.42, 0.38),
    "right_middle_lobe":         (0.12, 0.33, 0.42, 0.55),
    "right_lower_lobe":          (0.12, 0.48, 0.48, 0.82),
    "left_upper_lobe":           (0.58, 0.08, 0.92, 0.38),
    "left_lower_lobe":           (0.52, 0.48, 0.88, 0.82),
    "mediastinum":               (0.35, 0.05, 0.65, 0.65),
    "cardiac_silhouette":        (0.28, 0.35, 0.72, 0.78),
    "right_hilum":               (0.33, 0.22, 0.48, 0.48),
    "left_hilum":                (0.52, 0.22, 0.67, 0.48),
    "right_costophrenic_angle":  (0.08, 0.72, 0.32, 0.95),
    "left_costophrenic_angle":   (0.68, 0.72, 0.92, 0.95),
    "right_hemidiaphragm":       (0.08, 0.68, 0.48, 0.88),
    "left_hemidiaphragm":        (0.52, 0.68, 0.92, 0.88),
}

# Extended location string → zone name mapping
LOCATION_MAPPING = {
    # Right lung zones
    "right upper lobe": ["right_upper_lobe"],
    "rul": ["right_upper_lobe"],
    "right middle lobe": ["right_middle_lobe"],
    "rml": ["right_middle_lobe"],
    "right lower lobe": ["right_lower_lobe"],
    "rll": ["right_lower_lobe"],
    "right lung": ["right_upper_lobe", "right_middle_lobe", "right_lower_lobe"],
    "right apex": ["right_upper_lobe"],
    "right base": ["right_lower_lobe"],
    # Left lung zones
    "left upper lobe": ["left_upper_lobe"],
    "lul": ["left_upper_lobe"],
    "left lower lobe": ["left_lower_lobe"],
    "lll": ["left_lower_lobe"],
    "left lung": ["left_upper_lobe", "left_lower_lobe"],
    "left apex": ["left_upper_lobe"],
    "left base": ["left_lower_lobe"],
    # Bilateral / General
    "upper lobe": ["right_upper_lobe", "left_upper_lobe"],
    "upper lobes": ["right_upper_lobe", "left_upper_lobe"],
    "lower lobe": ["right_lower_lobe", "left_lower_lobe"],
    "lower lobes": ["right_lower_lobe", "left_lower_lobe"],
    "bilateral": ["right_upper_lobe", "right_middle_lobe", "right_lower_lobe",
                   "left_upper_lobe", "left_lower_lobe"],
    "lungs": ["right_upper_lobe", "right_middle_lobe", "right_lower_lobe",
              "left_upper_lobe", "left_lower_lobe"],
    "lung fields": ["right_upper_lobe", "right_middle_lobe", "right_lower_lobe",
                    "left_upper_lobe", "left_lower_lobe"],
    "lung": ["right_upper_lobe", "right_middle_lobe", "right_lower_lobe",
             "left_upper_lobe", "left_lower_lobe"],
    # Central structures
    "mediastinum": ["mediastinum"],
    "mediastinal": ["mediastinum"],
    "cardiac": ["cardiac_silhouette"],
    "cardiac silhouette": ["cardiac_silhouette"],
    "heart": ["cardiac_silhouette"],
    "heart size": ["cardiac_silhouette"],
    "right hilum": ["right_hilum"],
    "left hilum": ["left_hilum"],
    "hilum": ["right_hilum", "left_hilum"],
    "hila": ["right_hilum", "left_hilum"],
    "hilar": ["right_hilum", "left_hilum"],
    # Diaphragm / Costophrenic
    "right costophrenic angle": ["right_costophrenic_angle"],
    "left costophrenic angle": ["left_costophrenic_angle"],
    "costophrenic": ["right_costophrenic_angle", "left_costophrenic_angle"],
    "costophrenic angles": ["right_costophrenic_angle", "left_costophrenic_angle"],
    "right hemidiaphragm": ["right_hemidiaphragm"],
    "left hemidiaphragm": ["left_hemidiaphragm"],
    "diaphragm": ["right_hemidiaphragm", "left_hemidiaphragm"],
    "diaphragms": ["right_hemidiaphragm", "left_hemidiaphragm"],
    # Pleural
    "pleural": ["right_costophrenic_angle", "left_costophrenic_angle",
                "right_lower_lobe", "left_lower_lobe"],
    "right pleural": ["right_costophrenic_angle", "right_lower_lobe"],
    "left pleural": ["left_costophrenic_angle", "left_lower_lobe"],
    "pleural space": ["right_costophrenic_angle", "left_costophrenic_angle"],
    # Additional anatomy
    "aorta": ["mediastinum"],
    "aortic": ["mediastinum"],
    "aortic knob": ["mediastinum"],
    "trachea": ["mediastinum"],
    "spine": ["mediastinum"],
    "thoracic spine": ["mediastinum"],
    "retrocardiac": ["left_lower_lobe", "cardiac_silhouette"],
    "perihilar": ["right_hilum", "left_hilum"],
    "peribronchial": ["right_hilum", "left_hilum"],
}


# ============================================================
# Fuzzy Location Matching
# ============================================================

# Regex patterns for anatomical location extraction
LATERALITY_PATTERN = re.compile(
    r'\b(right|left|bilateral|bibasilar)\b', re.IGNORECASE
)
ZONE_PATTERN = re.compile(
    r'\b(upper|middle|lower|apical|basal)\s*(lobe|zone|lung)?\b', re.IGNORECASE
)
STRUCTURE_PATTERN = re.compile(
    r'\b(hilum|hila|hilar|mediastinum|mediastinal|cardiac|heart|'
    r'costophrenic|diaphragm|hemidiaphragm|pleural|aorta|aortic|'
    r'trachea|spine|retrocardiac|perihilar|peribronchial)\b',
    re.IGNORECASE
)


def location_to_zones(location_str: str) -> List[str]:
    """
    Map a natural language location string to atlas zone names.
    
    Uses exact lookup first, then falls back to fuzzy regex matching.
    
    Args:
        location_str: e.g., "right lower lobe", "cardiac silhouette", "lungs"
    
    Returns:
        List of zone names from CHEST_ZONES
    """
    location_lower = location_str.lower().strip()

    # Exact lookup
    if location_lower in LOCATION_MAPPING:
        return LOCATION_MAPPING[location_lower]

    # Fuzzy matching: try to parse laterality + zone
    zones = []
    laterality = LATERALITY_PATTERN.findall(location_lower)
    zone_words = ZONE_PATTERN.findall(location_lower)
    structures = STRUCTURE_PATTERN.findall(location_lower)

    # If a known structure is mentioned, map it
    for struct in structures:
        struct_lower = struct.lower()
        if struct_lower in LOCATION_MAPPING:
            zones.extend(LOCATION_MAPPING[struct_lower])

    # If laterality + zone-level words are present
    if laterality and zone_words:
        for lat in laterality:
            for zone_word, _ in zone_words:
                key = f"{lat.lower()} {zone_word.lower()} lobe"
                if key in LOCATION_MAPPING:
                    zones.extend(LOCATION_MAPPING[key])

    # If only laterality, map to full lung side
    elif laterality and not zones:
        for lat in laterality:
            key = f"{lat.lower()} lung"
            if key in LOCATION_MAPPING:
                zones.extend(LOCATION_MAPPING[key])

    # Deduplicate while preserving order
    seen = set()
    unique_zones = []
    for z in zones:
        if z not in seen:
            seen.add(z)
            unique_zones.append(z)

    return unique_zones if unique_zones else []


# ============================================================
# Patch Mask Generation
# ============================================================

def zone_to_bbox(zone_name: str) -> Optional[Tuple[float, float, float, float]]:
    """Get the normalized bounding box for a zone."""
    return CHEST_ZONES.get(zone_name)


def bbox_to_patch_mask(
    bbox: Tuple[float, float, float, float],
    patch_grid: Tuple[int, int] = (24, 24),
) -> np.ndarray:
    """
    Convert a normalized bounding box to a binary patch mask.
    
    Args:
        bbox: (x1, y1, x2, y2) in [0, 1] normalized coordinates
        patch_grid: (height, width) of the vision encoder's patch grid
    
    Returns:
        Binary mask of shape (H, W) where 1 = inside zone
    """
    x1, y1, x2, y2 = bbox
    H, W = patch_grid
    mask = np.zeros((H, W), dtype=np.float32)

    # Convert normalized coords to patch indices
    col_start = int(np.floor(x1 * W))
    col_end = int(np.ceil(x2 * W))
    row_start = int(np.floor(y1 * H))
    row_end = int(np.ceil(y2 * H))

    # Clip to valid range
    col_start = max(0, min(col_start, W))
    col_end = max(0, min(col_end, W))
    row_start = max(0, min(row_start, H))
    row_end = max(0, min(row_end, H))

    mask[row_start:row_end, col_start:col_end] = 1.0
    return mask


def claim_to_region(
    location_str: str,
    patch_grid: Tuple[int, int] = (24, 24),
) -> Optional[np.ndarray]:
    """
    Map a claim's location string to a patch mask.
    
    Combines masks from all matched zones (union).
    
    Args:
        location_str: Natural language location, e.g., "right lower lobe"
        patch_grid: (H, W) of vision encoder patch grid
    
    Returns:
        Binary mask of shape (H, W), or None if location cannot be mapped
    """
    zones = location_to_zones(location_str)
    if not zones:
        return None

    H, W = patch_grid
    combined_mask = np.zeros((H, W), dtype=np.float32)

    for zone_name in zones:
        bbox = zone_to_bbox(zone_name)
        if bbox is not None:
            zone_mask = bbox_to_patch_mask(bbox, patch_grid)
            combined_mask = np.maximum(combined_mask, zone_mask)

    return combined_mask if combined_mask.sum() > 0 else None


def get_all_zone_masks(
    patch_grid: Tuple[int, int] = (24, 24),
) -> Dict[str, np.ndarray]:
    """Get masks for all zones in the atlas."""
    masks = {}
    for zone_name, bbox in CHEST_ZONES.items():
        masks[zone_name] = bbox_to_patch_mask(bbox, patch_grid)
    return masks


# ============================================================
# Atlas Visualization (for Notebook 04)
# ============================================================

def get_zone_colors() -> Dict[str, Tuple[int, int, int]]:
    """Get consistent colors for each zone (for visualization)."""
    colors = {
        "right_upper_lobe":         (255, 100, 100),  # Red
        "right_middle_lobe":        (255, 180, 100),  # Orange
        "right_lower_lobe":         (255, 255, 100),  # Yellow
        "left_upper_lobe":          (100, 255, 100),  # Green
        "left_lower_lobe":          (100, 255, 255),  # Cyan
        "mediastinum":              (100, 100, 255),  # Blue
        "cardiac_silhouette":       (255, 100, 255),  # Magenta
        "right_hilum":              (200, 150, 100),  # Brown
        "left_hilum":               (150, 200, 100),  # Olive
        "right_costophrenic_angle": (255, 150, 200),  # Pink
        "left_costophrenic_angle":  (200, 255, 150),  # Lime
        "right_hemidiaphragm":      (150, 150, 255),  # Periwinkle
        "left_hemidiaphragm":       (255, 200, 150),  # Peach
    }
    return colors


def visualize_atlas_on_image(
    image: np.ndarray,
    patch_grid: Tuple[int, int] = (24, 24),
    alpha: float = 0.3,
) -> np.ndarray:
    """
    Overlay all atlas zones on an image for visualization.
    
    Args:
        image: RGB image as numpy array (H, W, 3)
        patch_grid: Vision encoder patch grid size
        alpha: Transparency of overlay
    
    Returns:
        Overlaid image as numpy array
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from PIL import Image as PILImage

    H_img, W_img = image.shape[:2]
    overlay = image.copy().astype(np.float32)
    colors = get_zone_colors()

    for zone_name, bbox in CHEST_ZONES.items():
        x1, y1, x2, y2 = bbox
        # Convert to pixel coordinates
        px1, py1 = int(x1 * W_img), int(y1 * H_img)
        px2, py2 = int(x2 * W_img), int(y2 * H_img)

        color = np.array(colors.get(zone_name, (200, 200, 200)), dtype=np.float32)
        overlay[py1:py2, px1:px2] = (
            overlay[py1:py2, px1:px2] * (1 - alpha) + color * alpha
        )

    return overlay.astype(np.uint8)
