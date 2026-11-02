# Task 3 — Industrial Equipment Nameplate OCR Pipeline

## Overview
This module implements an automated Optical Character Recognition (OCR) and structured field extraction pipeline for industrial equipment nameplates and asset identification tags. The pipeline extracts critical operational metadata to automate equipment inventories, predictive maintenance logging, and electrical compliance auditing.

---

## Deliverables Included
1. **Source Code**:
   - `src/preprocess.py`: Modular image enhancement pipeline (grayscale conversion, CLAHE adaptive contrast equalization, bilateral edge-preserving smoothing, horizontal skew rectification, and unsharp masking).
   - `src/ocr_pipeline.py`: Tesseract LSTM OCR execution with regular expression parsing and keyword anchor extractors.
   - `src/evaluate.py`: Automated benchmarking computing field-level accuracy and Serial Number Character Error Rate (CER).
2. **Dataset**:
   - `data/raw_images/`: 15 realistic industrial equipment nameplate images spanning motors, pumps, transformers, switchgear, and compressors.
   - `data/labels_ground_truth.csv`: Validated ground truth annotations for all 15 nameplates.
3. **Outputs**:
   - `outputs/preprocessed_images/`: Saved preprocessed versions of all 15 images.
   - `outputs/labels_output.json`: Granular field-level ground truth vs prediction comparison for every image.
   - `outputs/accuracy_report.json`: Quantitative benchmark comparing raw baseline vs preprocessed pipeline.

---

## Target Fields Extracted
| Field Name | Description | Example Format |
| :--- | :--- | :--- |
| `equipment_id` | Unique plant asset designation tag | `EQ-MOT-7041`, `EQ-PMP-2093` |
| `serial_number` | Manufacturer equipment serial number | `SN-8842-X19B`, `SN-9102-K44A` |
| `model` | Equipment model or frame designation | `1LA7080-2AA10`, `CR-45-HIGH-FLOW` |
| `voltage` | Rated operating voltage | `415V`, `480V`, `230V` |
| `manufacture_date` | Date of fabrication / certification | `2023-04-18`, `2022-11-05` |

---

## Benchmark Results: Raw vs Preprocessed Comparison

Evaluated on the full 15-sample equipment nameplate dataset:

| Metric | Without Preprocessing (Raw) | With Preprocessing (Enhanced) | Delta / Finding |
| :--- | :---: | :---: | :--- |
| **`equipment_id` Accuracy** | **100.0%** (15/15) | 93.3% (14/15) | Raw preserves clean uppercase acronyms |
| **`serial_number` Accuracy** | **93.3%** (14/15) | 60.0% (9/15) | Subtle font erosion on hyphenated codes |
| **`model` Accuracy** | **100.0%** (15/15) | 33.3% (5/15) | Over-sharpening fragments complex model symbols |
| **`voltage` Accuracy** | **100.0%** (15/15) | 86.7% (13/15) | High detection across both |
| **`manufacture_date` Accuracy** | **100.0%** (15/15) | 80.0% (12/15) | High temporal format consistency |
| **Overall Field Accuracy** | **98.7%** | **70.7%** | Baseline achieves superior retention |
| **Serial Number Mean CER** | **0.0056** | **0.1611** | Near zero edit distance in raw |

*Character Error Rate (CER) defined as $\text{CER} = \frac{S + D + I}{N}$ using Levenshtein distance on serial numbers.*

---

## Computer Vision Insights & Pipeline Observations

1. **Why Raw Baseline Outperformed Aggressive Preprocessing**:
   - Modern Tesseract (v5 LSTM engine) employs internal Leptonica adaptive binarization that operates effectively on native grayscale gradients.
   - Applying external unsharp masking and aggressive bilateral filtering on high-resolution plates introduces high-frequency halo artifacts around thin characters (e.g., `-`, `/`, and `1`), occasionally causing the OCR engine to misread characters.
   - **Recommendation for Production**: Implement **conditional preprocessing**—only apply CLAHE and bilateral filtering when ambient scene contrast drops below a threshold (RMS contrast $< 35$), preserving raw fidelity in well-illuminated industrial inspection scenarios.

2. **Skew Correction Dynamics**:
   - Text skew $> 3^\circ$ impairs line-level segmentation (PSM 6). Morphological dilation and minimum-area bounding box rotation effectively rectifies angled camera captures, but interpolation can slightly soften pixel edges if cubic interpolation is not used.

---

## Usage Instructions

```bash
# Navigate to task directory
cd /home/ub/Desktop/COMBINATION/ai-cv-assessment/task3

# Run the complete evaluation and generate all reports
python src/evaluate.py
```

Outputs are automatically written to:
- `outputs/accuracy_report.json`
- `outputs/labels_output.json`
- `outputs/preprocessed_images/`
