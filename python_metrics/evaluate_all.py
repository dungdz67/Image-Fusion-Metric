"""
Batch image fusion metrics evaluator.
Scans results/ directory, computes all metrics per model/modality,
and outputs one CSV per modality with per-model averages.

Directory structure expected:
  results/
    <ModelName>/
      <Modality>/       e.g. CT-MRI, PET-MRI, SPECT-MRI
        *.png / *.jpg   fused images

  dataset/Harvard/
    <Modality>/
      CT/ (or first source)  *.png
      MRI/ (or second source) *.png
"""

import os
import csv
import traceback
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

from structure_metrics import *
from img_metrics import *
from info_metrics import *
from quality_metrics import *
from visual_metrics import *
from others_metrics import *

from tqdm import tqdm
# ── Configuration ─────────────────────────────────────────────────────────────

RESULTS_DIR = Path("results")
DATASET_DIR = Path("dataset/Harvard")
OUTPUT_DIR  = Path("output_metrics")

# Map modality folder name → (source_A_subfolder, source_B_subfolder)
MODALITY_SOURCES = {
    "CT-MRI":    ("CT",   "MRI"),
    "PET-MRI":   ("PET",  "MRI"),
    "SPECT-MRI": ("SPECT","MRI"),
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")).astype(np.float64)


def compute_metrics(A, B, F) -> dict:
    m = {}
    # Quality
    m["MLI"]   = mli_error(F)
    m["SD"]    = sd(F)
    m["AG"]    = ag(F)
    m["MSE"]   = mse_f(A, B, F)
    m["PSNR"]  = psnr_f(A, B, F)
    # Info
    m["EN"]    = en(F)
    m["MI"]    = mi(A, B, F)
    m["NCIE"]  = ncie(A, B, F)
    m["SCD"]   = scd(A, B, F)
    m["FMI_pixel"]   = fmi(A, B, F)
    m["FMI_dct"]     = fmi(A, B, F, feature="dct")
    m["FMI_wavelet"] = fmi(A, B, F, feature="wavelet")
    m["FMI_edge"]    = fmi(A, B, F, feature="edge")
    # Image
    m["QABF_fast"] = qabf(A, B, F)
    qabf_full, labf, nabf, nabf1 = petrovic_metrics(A, B, F)
    m["QABF"]  = qabf_full
    m["LABF"]  = labf
    m["NABF"]  = nabf
    m["NABF1"] = nabf1
    m["SF"]    = sf(F)
    # Structure
    m["SSIM"]    = 0.5 * ssim(A, F) + 0.5 * ssim(B, F)
    m["MS_SSIM"] = ms_ssim(np.stack([A, B], axis=2), F)
    m["Q"]       = piella_metrics(A, B, F, sw=1)
    m["Qw"]      = piella_metrics(A, B, F, sw=2)
    m["Qe"]      = piella_metrics(A, B, F, sw=3)
    # Visual
    m["VIF"]  = vif(A, F) + vif(B, F)
    m["VIFF"] = viff(A, B, F)
    # Others
    m["CC"]  = cc(A, B, F)
    m["Qp"] = qp(A, B, F)
    return m


def collect_fused_images(model_dir: Path, modality: str):
    """Return sorted list of image paths for a given model/modality."""
    mod_dir = model_dir / modality
    if not mod_dir.is_dir():
        return []
    return sorted([p for p in mod_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])


def get_source_paths(modality: str, stem: str):
    """Return (pathA, pathB) for a fused image stem, or (None, None) if missing."""
    if modality not in MODALITY_SOURCES:
        return None, None
    src_a, src_b = MODALITY_SOURCES[modality]
    base = DATASET_DIR / modality
    for ext in IMAGE_EXTENSIONS:
        a = base / src_a / (stem + ext)
        b = base / src_b / (stem + ext)
        if a.exists() and b.exists():
            return a, b
    return None, None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Discover models
    models = sorted([d for d in RESULTS_DIR.iterdir() if d.is_dir()])
    if not models:
        print(f"No model folders found in {RESULTS_DIR}")
        return

    # Discover all modalities across all models
    all_modalities = set()
    for m in models:
        for sub in m.iterdir():
            if sub.is_dir():
                all_modalities.add(sub.name)
    all_modalities = sorted(all_modalities)

    print(f"Models    : {[m.name for m in models]}")
    print(f"Modalities: {all_modalities}")

    # Per-modality accumulator: {modality: {model_name: {metric: [values]}}}
    accum = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for model_dir in models:
        model_name = model_dir.name
        for modality in all_modalities:
            fused_paths = collect_fused_images(model_dir, modality)
            if not fused_paths:
                continue
            print(f"\n[{model_name} / {modality}]  {len(fused_paths)} images")
            ok = skip = 0
            for fpath in tqdm(fused_paths, desc="Eval"):
                pa, pb = get_source_paths(modality, fpath.stem)
                if pa is None:
                    skip += 1
                    continue
                A = load_gray(pa)
                B = load_gray(pb)
                F = load_gray(fpath)
                metrics = compute_metrics(A, B, F)
                for k, v in metrics.items():
                    accum[modality][model_name][k].append(v)
                ok += 1

            print(f"  done: {ok} ok, {skip} skipped")

    # Write one CSV per modality
    for modality, model_data in accum.items():
        if not model_data:
            continue
        # Determine metric columns (from first model that has data)
        metric_cols = list(next(iter(model_data.values())).keys())

        csv_path = OUTPUT_DIR / f"{modality}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Model"] + metric_cols)
            for model_name in sorted(model_data.keys()):
                row = [model_name]
                for col in metric_cols:
                    vals = model_data[model_name].get(col, [])
                    row.append(f"{np.mean(vals):.6f}" if vals else "N/A")
                writer.writerow(row)

        print(f"\nSaved → {csv_path}  ({len(model_data)} models)")


if __name__ == "__main__":
    main()