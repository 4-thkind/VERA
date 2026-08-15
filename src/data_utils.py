"""
VERA Data Utilities
Handles downloading, parsing, and preparing the Indiana U CXR dataset.
"""
import os
import json
import random
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import requests
from tqdm import tqdm
from PIL import Image


# ============================================================
# Download Helpers
# ============================================================

def download_file(url: str, save_path: str, chunk_size: int = 8192) -> str:
    """Download a file with progress bar."""
    save_path = Path(save_path)
    if save_path.exists():
        print(f"  [SKIP] Already exists: {save_path.name}")
        return str(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url} ...")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    with open(save_path, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc=save_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))
    return str(save_path)


def extract_tgz(tgz_path: str, extract_dir: str) -> str:
    """Extract a .tgz archive."""
    extract_dir = Path(extract_dir)
    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"  [SKIP] Already extracted: {extract_dir}")
        return str(extract_dir)

    print(f"  Extracting {Path(tgz_path).name} ...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
    return str(extract_dir)


def download_indiana_cxr(raw_dir: str) -> Tuple[str, str]:
    """
    Download Indiana U CXR dataset (images + reports).
    
    Returns:
        (images_dir, reports_dir) paths
    """
    from config import INDIANA_IMAGES_URL, INDIANA_REPORTS_URL

    raw_dir = Path(raw_dir)

    # Download images
    images_tgz = download_file(INDIANA_IMAGES_URL, raw_dir / "NLMCXR_png.tgz")
    images_dir = extract_tgz(images_tgz, raw_dir / "images")

    # Download reports
    reports_tgz = download_file(INDIANA_REPORTS_URL, raw_dir / "NLMCXR_reports.tgz")
    reports_dir = extract_tgz(reports_tgz, raw_dir / "reports")

    return str(images_dir), str(reports_dir)


# ============================================================
# XML Report Parsing
# ============================================================

def parse_xml_report(xml_path: str) -> Optional[Dict]:
    """
    Parse a single Indiana U CXR XML report file.
    
    Returns dict with:
        - uid: unique identifier
        - findings: text of findings section
        - impression: text of impression section
        - mesh_terms: MeSH terms (if available)
        - image_ids: list of associated image IDs
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except (ET.ParseError, Exception) as e:
        print(f"  [WARN] Failed to parse {xml_path}: {e}")
        return None

    report = {
        "uid": Path(xml_path).stem,
        "findings": "",
        "impression": "",
        "mesh_terms": [],
        "image_ids": [],
        "source_file": str(xml_path),
    }

    # Extract abstract text sections
    for abstract in root.iter("AbstractText"):
        label = abstract.get("Label", "").upper()
        text = abstract.text or ""
        text = text.strip()
        if label == "FINDINGS":
            report["findings"] = text
        elif label == "IMPRESSION":
            report["impression"] = text

    # Extract MeSH terms
    for mesh in root.iter("major"):
        term = mesh.text
        if term:
            report["mesh_terms"].append(term.strip())
    for mesh in root.iter("automatic"):
        term = mesh.text
        if term:
            report["mesh_terms"].append(term.strip())

    # Extract image references
    for parent_image in root.iter("parentImage"):
        img_id = parent_image.get("id", "")
        if img_id:
            report["image_ids"].append(img_id)

    # Skip if both findings and impression are empty
    if not report["findings"] and not report["impression"]:
        return None

    return report


def parse_all_reports(reports_dir: str) -> List[Dict]:
    """Parse all XML reports in a directory (recursive)."""
    reports_dir = Path(reports_dir)
    xml_files = list(reports_dir.rglob("*.xml"))
    print(f"  Found {len(xml_files)} XML files")

    reports = []
    for xml_path in tqdm(xml_files, desc="Parsing reports"):
        report = parse_xml_report(str(xml_path))
        if report is not None:
            reports.append(report)

    print(f"  Successfully parsed {len(reports)} reports (skipped {len(xml_files) - len(reports)})")
    return reports


# ============================================================
# Kaggle CSV Parsing (Indiana U CXR on Kaggle)
# ============================================================
# The Kaggle dataset "raddar/chest-xrays-indiana-university" has:
#   - images/images_normalized/  (PNG files)
#   - indiana_reports.csv   (uid, findings, impression, mesh terms)
#   - indiana_projections.csv (uid, filename, projection type)

def parse_kaggle_reports(kaggle_input_dir: str) -> List[Dict]:
    """
    Parse Indiana U CXR reports from Kaggle CSV format.
    
    Kaggle dataset has:
    - indiana_reports.csv with columns: uid, MeSH, Problems, image, indication, comparison, findings, impression
    - indiana_projections.csv with columns: uid, filename, projection
    
    Returns list of report dicts compatible with the rest of the pipeline.
    """
    import pandas as pd

    kaggle_dir = Path(kaggle_input_dir)

    # Load reports CSV
    reports_csv = kaggle_dir / "indiana_reports.csv"
    if not reports_csv.exists():
        # Try alternative locations
        for alt in ["indiana_reports.csv", "reports.csv"]:
            alt_path = kaggle_dir / alt
            if alt_path.exists():
                reports_csv = alt_path
                break
    
    if not reports_csv.exists():
        raise FileNotFoundError(
            f"Reports CSV not found in {kaggle_dir}. "
            f"Files found: {[f.name for f in kaggle_dir.iterdir()]}"
        )

    df_reports = pd.read_csv(reports_csv)
    print(f"  Loaded reports CSV: {len(df_reports)} rows")
    print(f"  Columns: {list(df_reports.columns)}")

    # Load projections CSV (image filename mapping)
    projections_csv = kaggle_dir / "indiana_projections.csv"
    df_proj = None
    if projections_csv.exists():
        df_proj = pd.read_csv(projections_csv)
        print(f"  Loaded projections CSV: {len(df_proj)} rows")

    reports = []
    for _, row in df_reports.iterrows():
        uid = str(row.get("uid", ""))
        findings = str(row.get("findings", "")) if pd.notna(row.get("findings")) else ""
        impression = str(row.get("impression", "")) if pd.notna(row.get("impression")) else ""
        mesh = str(row.get("MeSH", "")) if pd.notna(row.get("MeSH")) else ""
        problems = str(row.get("Problems", "")) if pd.notna(row.get("Problems")) else ""

        # Skip empty reports
        if not findings.strip() and not impression.strip():
            continue

        # Get image IDs from projections CSV
        image_ids = []
        if df_proj is not None:
            proj_rows = df_proj[df_proj["uid"] == int(uid)] if uid.isdigit() else pd.DataFrame()
            for _, proj_row in proj_rows.iterrows():
                filename = str(proj_row.get("filename", ""))
                if filename:
                    # Remove .png extension for image_id
                    img_id = filename.replace(".png", "").strip()
                    image_ids.append(img_id)
        
        # If no projections CSV, try the 'image' column from reports
        if not image_ids:
            image_col = str(row.get("image", "")) if pd.notna(row.get("image")) else ""
            if image_col:
                # May be semicolon or comma separated
                for img_ref in image_col.replace(";", ",").split(","):
                    img_ref = img_ref.strip()
                    if img_ref:
                        image_ids.append(img_ref.replace(".png", ""))

        # Parse MeSH terms
        mesh_terms = [m.strip() for m in mesh.split(";") if m.strip()] if mesh else []

        report = {
            "uid": uid,
            "findings": findings.strip(),
            "impression": impression.strip(),
            "mesh_terms": mesh_terms,
            "problems": problems,
            "image_ids": image_ids,
            "source_file": str(reports_csv),
        }
        reports.append(report)

    print(f"  Successfully parsed {len(reports)} reports from Kaggle CSV")
    return reports


def find_kaggle_images(kaggle_input_dir: str) -> Dict[str, str]:
    """
    Find all PNG images in the Kaggle dataset directory.
    Searches common subdirectory patterns.
    """
    kaggle_dir = Path(kaggle_input_dir)
    image_map = {}

    # Search in common Kaggle image locations
    search_dirs = [
        kaggle_dir / "images" / "images_normalized",
        kaggle_dir / "images",
        kaggle_dir,
    ]

    for search_dir in search_dirs:
        if search_dir.exists():
            for img_path in search_dir.rglob("*.png"):
                img_id = img_path.stem
                if img_id not in image_map:  # Don't overwrite
                    image_map[img_id] = str(img_path)

    print(f"  Found {len(image_map)} PNG images in Kaggle dataset")
    return image_map


def prepare_kaggle_data(kaggle_input_dir: str) -> Tuple[List[Dict], Dict[str, str]]:
    """
    One-step preparation for Kaggle: parse CSVs + find images.
    No download needed — data is already mounted.
    
    Returns:
        (reports, image_map)
    """
    print("  Parsing Kaggle CSV reports...")
    reports = parse_kaggle_reports(kaggle_input_dir)
    
    print("  Finding Kaggle images...")
    image_map = find_kaggle_images(kaggle_input_dir)
    
    return reports, image_map


# ============================================================
# Anatomical Filtering
# ============================================================

# Anatomical location keywords
ANATOMY_KEYWORDS = [
    "lobe", "lung", "lungs", "hilum", "hila", "hilar",
    "costophrenic", "mediastinum", "mediastinal", "cardiac",
    "heart", "diaphragm", "hemidiaphragm", "pleural",
    "trachea", "aorta", "aortic", "rib", "ribs", "spine",
    "thoracic", "upper", "lower", "middle", "right", "left",
    "bilateral", "apex", "apical", "base", "basal",
    "perihilar", "peribronchial", "retrocardiac",
]


def has_anatomy_mention(report: Dict) -> bool:
    """Check if a report mentions any anatomical location."""
    text = (report.get("findings", "") + " " + report.get("impression", "")).lower()
    return any(kw in text for kw in ANATOMY_KEYWORDS)


def filter_reports_with_anatomy(reports: List[Dict]) -> List[Dict]:
    """Filter reports that contain anatomical location mentions."""
    filtered = [r for r in reports if has_anatomy_mention(r)]
    pct = len(filtered) / len(reports) * 100 if reports else 0
    print(f"  Filtered: {len(filtered)}/{len(reports)} reports have anatomy mentions ({pct:.1f}%)")
    return filtered


# ============================================================
# Image-Report Linking
# ============================================================

def find_images(images_dir: str) -> Dict[str, str]:
    """Find all PNG images and return {image_id: image_path} dict."""
    images_dir = Path(images_dir)
    image_map = {}
    for img_path in images_dir.rglob("*.png"):
        image_map[img_path.stem] = str(img_path)
    print(f"  Found {len(image_map)} PNG images")
    return image_map


def link_reports_to_images(reports: List[Dict], image_map: Dict[str, str]) -> List[Dict]:
    """
    Link each report to its associated images.
    Creates one entry per image (flattened from one-to-many report-image relationship).
    Only keeps frontal views (CXR*_IM-*-*001.png pattern suggests frontal).
    """
    linked = []
    for report in reports:
        for img_id in report.get("image_ids", []):
            if img_id in image_map:
                entry = {
                    "image_id": img_id,
                    "image_path": image_map[img_id],
                    "report_uid": report["uid"],
                    "findings": report["findings"],
                    "impression": report["impression"],
                    "reference_report": (
                        report["findings"] + " " + report["impression"]
                    ).strip(),
                    "mesh_terms": report["mesh_terms"],
                }
                linked.append(entry)

    print(f"  Linked {len(linked)} image-report pairs")
    return linked


# ============================================================
# Data Splitting
# ============================================================

def create_splits(
    data: List[Dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """
    Split data into train/val/test sets.
    
    Train: used for model inference (generate reports)
    Val: threshold calibration
    Test: final evaluation
    """
    random.seed(seed)
    indices = list(range(len(data)))
    random.shuffle(indices)

    n_train = int(len(data) * train_ratio)
    n_val = int(len(data) * val_ratio)

    splits = {
        "train": [data[i] for i in indices[:n_train]],
        "val": [data[i] for i in indices[n_train : n_train + n_val]],
        "test": [data[i] for i in indices[n_train + n_val :]],
    }

    for split_name, split_data in splits.items():
        print(f"  {split_name}: {len(split_data)} samples")

    return splits


# ============================================================
# Save / Load Utilities
# ============================================================

def save_json(data, path: str):
    """Save data as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {path}")


def load_json(path: str):
    """Load data from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_split_data(splits: Dict[str, List[Dict]], output_dir: str):
    """Save split data to separate JSON files."""
    output_dir = Path(output_dir)
    for split_name, split_data in splits.items():
        save_json(split_data, output_dir / f"{split_name}.json")


def load_split_data(output_dir: str) -> Dict[str, List[Dict]]:
    """Load split data from JSON files."""
    output_dir = Path(output_dir)
    splits = {}
    for split_name in ["train", "val", "test"]:
        path = output_dir / f"{split_name}.json"
        if path.exists():
            splits[split_name] = load_json(str(path))
    return splits


def load_image(image_path: str, size: Tuple[int, int] = (384, 384)) -> Image.Image:
    """Load and resize an image."""
    img = Image.open(image_path).convert("RGB")
    if size:
        img = img.resize(size, Image.Resampling.LANCZOS)
    return img

# ============================================================
# Kaggle / Dual-Dataset Loading Helpers
# ============================================================

def load_indiana_u() -> List[Dict]:
    """Load Indiana U dataset from Kaggle mount."""
    from config import IU_DATA_DIR
    
    iu_dir = Path(IU_DATA_DIR)
    images_dir = iu_dir / "images"
    reports_dir = iu_dir / "reports"
    
    # 1. Map images
    image_map = {}
    if images_dir.exists():
        for img_path in images_dir.rglob("*.png"):
            image_map[img_path.stem] = str(img_path)
    
    # 2. Parse XML reports
    samples = []
    if reports_dir.exists():
        for xml_path in tqdm(list(reports_dir.rglob("*.xml")), desc="Parsing IU XMLs"):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                # Extract text
                findings = ""
                impression = ""
                for abstract in root.findall(".//AbstractText"):
                    label = abstract.get("Label", "").upper()
                    text = abstract.text if abstract.text else ""
                    if label == "FINDINGS":
                        findings = text
                    elif label == "IMPRESSION":
                        impression = text
                
                # Extract image IDs
                img_nodes = root.findall(".//parentImage")
                img_ids = [node.get("id") for node in img_nodes if node.get("id")]
                
                # Link each image ID
                for img_id in img_ids:
                    if img_id in image_map:
                        report_text = f"{findings} {impression}".strip()
                        if report_text:
                            samples.append({
                                "image_id": img_id,
                                "image_path": image_map[img_id],
                                "report": report_text,
                                "source": "indiana_u"
                            })
            except Exception as e:
                continue
                
    return samples

def load_rexgradient() -> List[Dict]:
    """Load subset of ReXGradient dataset from HuggingFace via streaming."""
    from config import HF_DATASET_ID, HF_TOKEN, REX_SUBSET_SIZE, OUTPUT_DIR
    from datasets import load_dataset
    
    print(f"Loading {REX_SUBSET_SIZE} samples from {HF_DATASET_ID}...")
    dataset = load_dataset(
        HF_DATASET_ID, 
        split="train", 
        streaming=True, 
        token=HF_TOKEN
    )
    
    out_dir = Path(OUTPUT_DIR) / "rexgradient"
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    samples = []
    count = 0
    for item in dataset:
        if count >= REX_SUBSET_SIZE:
            break
            
        # The dataset provides 'image' (PIL Image) and 'report'
        if "image" in item and "report" in item:
            image_id = f"rex_{count:06d}"
            img_path = img_dir / f"{image_id}.png"
            
            # Save image to local disk
            item["image"].save(img_path)
            
            samples.append({
                "image_id": image_id,
                "image_path": str(img_path),
                "report": item["report"],
                "source": "rexgradient"
            })
            count += 1
            
    # Save metadata
    save_json(samples, out_dir / "metadata.json")
    
    return samples

def combine_datasets(iu_samples: List[Dict], rex_samples: List[Dict]) -> Dict[str, List[Dict]]:
    """Merge datasets, split 70/20/10, and save to PROCESSED_DIR."""
    from config import PROCESSED_DIR
    
    all_samples = iu_samples + rex_samples
    random.seed(42)
    random.shuffle(all_samples)
    
    n_total = len(all_samples)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.20)
    
    splits = {
        "train": all_samples[:n_train],
        "val": all_samples[n_train : n_train + n_val],
        "test": all_samples[n_train + n_val :]
    }
    
    out_dir = Path(PROCESSED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, split_data in splits.items():
        save_json(split_data, out_dir / f"{split_name}.json")
        
    return splits
