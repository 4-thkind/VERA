"""
VERA: Visual Evidence–Report Alignment
Central Configuration File

All notebooks and source modules import from this file.
Auto-detects Kaggle vs local environment.
"""
import os
from pathlib import Path

# ============================================================
# Project Paths & Dataset Settings
# ============================================================
IU_DATA_DIR = "/kaggle/input/chest-xrays-indiana-university/"
OUTPUT_DIR = "/kaggle/working/"
REX_SUBSET_SIZE = 500
HF_DATASET_ID = "rajpurkarlab/ReXGradient-160K"

PROJECT_ROOT = Path(OUTPUT_DIR)
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ATTENTION_DIR = DATA_DIR / "attention_maps"
CLAIMS_DIR = DATA_DIR / "claims"
RESULTS_DIR = DATA_DIR / "results"
FIGURES_DIR = DATA_DIR / "figures"

# We assume running on Kaggle, no need for IS_KAGGLE check.
IS_KAGGLE = True

# Create all writable directories on import
for _d in [PROCESSED_DIR, ATTENTION_DIR, CLAIMS_DIR, RESULTS_DIR, FIGURES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ============================================================
# HuggingFace Authentication
# ============================================================
# Priority: environment variable > hardcoded fallback
# WARNING: Do not commit this file with a real token to public repos.
HF_DATASET_TOKEN = os.environ.get(
    "HF_DATASET_TOKEN", 
    os.environ.get("HF_TOKEN", "YOUR_HF_DATASET_TOKEN_HERE")
)
HF_MODEL_TOKEN = os.environ.get(
    "HF_MODEL_TOKEN", 
    os.environ.get("HF_TOKEN", "YOUR_HF_MODEL_TOKEN_HERE")
)
# Kept for backward compatibility if scripts still use it
HF_TOKEN = os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN_HERE")

# ============================================================
# Model Configuration
# ============================================================
CHEXAGENT_MODEL_ID = "StanfordAIMI/CheXagent-2-3b"
LLAVA_MED_MODEL_ID = "microsoft/llava-med-v1.5-mistral-7b"

# Default model to use (switch between models here)
DEFAULT_MODEL = CHEXAGENT_MODEL_ID

# Generation parameters
MAX_NEW_TOKENS = 256
TEMPERATURE = 1.0  # Greedy decoding (no sampling)
DO_SAMPLE = False
NUM_BEAMS = 1  # Greedy; set to 4 for beam search

# ============================================================
# Dataset URLs
# ============================================================
INDIANA_IMAGES_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz"
INDIANA_REPORTS_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"
KAGGLE_DATASET = "raddar/chest-xrays-indiana-university"

# ============================================================
# VERA Severity-Calibrated Thresholds
# ============================================================
# These are starting values — calibrate on validation split (Notebook 05)
SEVERITY_THRESHOLDS = {
    "critical": 0.35,   # mass, tumour, pneumothorax, effusion
    "moderate": 0.25,   # consolidation, opacity, infiltrate
    "mild": 0.15,       # clear lungs, no cardiomegaly, normal
}

# Finding → severity tier lookup
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
    "infiltrate": "moderate", "infiltrates": "moderate",
    "atelectasis": "moderate",
    "fibrosis": "moderate",
    "thickening": "moderate",
    "congestion": "moderate",
    "widening": "moderate",
    "calcification": "moderate", "calcifications": "moderate",
    "granuloma": "moderate", "granulomas": "moderate",
    # Mild / Normal
    "clear": "mild", "normal": "mild",
    "unremarkable": "mild",
    "stable": "mild",
    "no acute": "mild",
    "cardiomegaly": "moderate",  # borderline moderate/critical
    "scoliosis": "mild",
    "degenerative": "mild",
}

# ============================================================
# Chest Anatomy Atlas — Normalized Bounding Boxes [x1, y1, x2, y2]
# ============================================================
# Coordinates are in [0, 1] range on a standard PA chest X-ray
# (0,0) = top-left, (1,1) = bottom-right
# Note: In radiology convention, "right" = patient's right = image left
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

# Location string → zone name mapping (handles natural language variations)
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
    # Bilateral / Ambiguous
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
    # Pleural
    "pleural": ["right_costophrenic_angle", "left_costophrenic_angle",
                "right_lower_lobe", "left_lower_lobe"],
    "right pleural": ["right_costophrenic_angle", "right_lower_lobe"],
    "left pleural": ["left_costophrenic_angle", "left_lower_lobe"],
}

# ============================================================
# Data Split Ratios
# ============================================================
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

# ============================================================
# NLI Model for Ground Truth Labelling
# ============================================================
NLI_MODEL_ID = "cross-encoder/nli-deberta-v3-base"

# ============================================================
# Random Seed
# ============================================================
RANDOM_SEED = 42

# ============================================================
# Attention Extraction Settings
# ============================================================
# Number of attention layers to average (from the end)
# Using last N layers captures the highest-level cross-modal interactions
NUM_ATTENTION_LAYERS = 4

# Patch grid size for SigLIP (CheXagent) vision encoder
# SigLIP with 384x384 input and patch_size=16 → 24x24 patches
PATCH_GRID_CHEXAGENT = (24, 24)

# Patch grid size for CLIP (LLaVA-Med) vision encoder
# CLIP ViT-L/14 with 336x336 input and patch_size=14 → 24x24 patches
PATCH_GRID_LLAVA = (24, 24)

# ============================================================
# Report Generation Prompt
# ============================================================
REPORT_PROMPT = (
    "You are an expert radiologist. Generate a detailed radiology report "
    "for this chest X-ray image. Include findings about the lungs, heart, "
    "mediastinum, and any abnormalities. Structure your response with "
    "Findings and Impression sections."
)
