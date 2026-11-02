# Datasets & Data Sources — AjnaLens AI/ML & CV Engineer Assessment

This directory documents all datasets, sources, licenses, splits, and annotations used across the four tasks in this assessment.

---

## 1. Primary Datasets Used

### A. Construction Helmet Detection Dataset (Task 1 & Task 2)
- **Source Link**: [Roboflow Construction Helmet Detect](https://app.roboflow.com/soiltechai/construction-helmet-detect-wvwr6/1)
- **License**: CC BY 4.0
- **Format**: YOLOv11 Annotation Format (bounding box coordinates normalized to `[0, 1]`)
- **Classes**:
  - `0: helmet` (Worker wearing hard hat / safety helmet)
  - `1: unhelmet` (Worker head without safety helmet)
- **Dataset Split**:
  - **Train**: 768 images
  - **Validation**: 217 images
  - **Test**: 108 images
  - **Total**: 1,093 images
- **Application**:
  - **Task 1**: Secondary detector for PPE compliance verification on detected pedestrians in the camera stream.
  - **Task 2**: 100-epoch fine-tuning, validation metric calculation (Precision, Recall, mAP@50, mAP@50-95), ONNX FP32/FP16 export, failure case diagnosis, and edge runtime benchmarking.

---

### B. People Detection Dataset (Task 1)
- **Source Link**: [Roboflow Universe People Detection (nastya-xqjkj)](https://universe.roboflow.com/nastya-xqjkj/people-ucweq)
- **License**: CC BY 4.0
- **Format**: YOLOv11 Annotation Format
- **Classes**:
  - `0: person` (Pedestrian / construction worker full body)
- **Dataset Split**:
  - **Train**: 842 images
  - **Validation**: 120 images
  - **Test**: 60 images
- **Application**:
  - **Task 1**: Primary pedestrian detector powering ByteTrack multi-object tracking, ground-contact footpoint trajectory tracing, idle duration monitoring, and restricted zone entry detection.

---

### C. Industrial Video Stream (Task 1)
- **Source**: Industrial surveillance CCTV footage (`runs/test/videoplayback.mp4` / `task1/outputs/output.mp4`)
- **Resolution**: 1280 x 720 (native aspect ratio preserved)
- **Framerate**: 20.0 FPS
- **Duration**: ~52 seconds (1050 frames)
- **Content**: Construction workers navigating hazardous work zones, demonstrating entry into restricted zones, idle dwell times, and PPE compliance checks.

---

### D. Equipment Nameplate Dataset (Task 3)
- **Location**: `task3/data/raw_images/` & `task3/data/labels_ground_truth.csv`
- **Count**: 15 distinct industrial equipment nameplate images across pumps, motors, compressors, transformers, and switchgear.
- **Ground Truth Fields**:
  - `equipment_id`: Unique industrial asset tag (e.g., `EQ-PMP-7012`, `EQ-MOT-3041`)
  - `serial_number`: Alphanumeric manufacturing serial (e.g., `SN-8842-X19B`, `SN-9102-K44A`)
  - `model`: Machinery model designation (e.g., `CR-45-HIGH-FLOW`, `TX-500-ISOL`)
  - `voltage`: Rated operating voltage (e.g., `415V`, `480V`, `230V`)
  - `manufacture_date`: Fabrication date in standard formats (e.g., `2023-04-18`, `2022-11-05`)
- **Application**:
  - Used to evaluate OCR performance under varying levels of image degradation (noise, low contrast, perspective skew), comparing raw baseline vs preprocessed OCR pipelines on Field Accuracy and Character Error Rate (CER).

---

### E. Industrial Safety SOP & Equipment Manual Corpus (Task 4)
- **Location**: `task4/data/industrial_sop_corpus.json`
- **Format**: Structured JSON documents with chunked metadata, domain tagging, and safety protocols.
- **Coverage**:
  - PPE Compliance Policies (OSHA 1926.100, Hard Hat color coding, replacement intervals)
  - Hazardous Restricted Zone Protocols (Buffer distances, lock-out/tag-out (LOTO) standard 29 CFR 1910.147)
  - Heavy Machinery Operating Manuals & Voltage Tolerances
  - Emergency Incident Reporting & Evacuation Pathways
- **Application**:
  - Vector knowledge base ingested into ChromaDB with semantic retrieval powering the Task 4 RAG assistant.

---

## 2. Data Storage & Preprocessing Directory Structure

```
data/
├── README.md               # This documentation file
└── (Task datasets are symlinked / maintained in respective task folders for modularity)
```

To run tasks directly from data without downloading full multi-gigabyte raw archives, weights and sample data are pre-packaged within the respective task modules.
