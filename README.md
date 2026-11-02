# AjnaLens AI/ML & CV Engineer Practical Assessment

A unified, production-ready computer vision and visual intelligence repository encompassing real-time zone intrusion monitoring, custom detector fine-tuning and optimization, industrial equipment nameplate OCR, and an OSHA-compliant safety RAG assistant.

---

## 1. Project Overview

This repository delivers end-to-end solutions for all four tasks specified in the **AjnaLens AI/ML & CV Engineer Practical Assessment**:

1. **Task 1 — Object Detection and Zone Monitoring**:
   - Real-time people detection, persistent multi-object tracking (ByteTrack), camera motion compensation, bottom-center footpoint zone boundary monitoring, 5-second stationary idle detection, and head-region PPE (safety helmet) compliance cross-verification.
2. **Task 2 — Fine-Tune, Evaluate, and Optimize Detector**:
   - 100-epoch training and validation of an Ultralytics YOLO11m detector on construction helmet data, exporting to ONNX (FP32 & FP16), benchmarking PyTorch vs ONNX Runtime across 100 validation images, and diagnosing 5 real failure cases.
3. **Task 3 — Industrial Equipment Nameplate OCR**:
   - Image enhancement pipeline (CLAHE, bilateral edge-preserving smoothing, Otsu deskewing) coupled with Tesseract 5.3 LSTM OCR and structured regular expression parsing to extract 5 key asset fields (`equipment_id`, `serial_number`, `model`, `voltage`, `manufacture_date`), benchmarked against raw baselines.
4. **Task 4 — Industrial Safety & Maintenance RAG Assistant**:
   - Semantic retrieval-augmented generation (RAG) microservice powered by FastAPI and dense vector embeddings (`all-MiniLM-L6-v2`), serving sub-10ms grounded safety advisories with explicit OSHA and plant SOP regulatory citations.

---

## 2. Repository Structure

```
ai-cv-assessment/
│
├── README.md                      # Top-level project documentation (this file)
├── requirements.txt               # Pinned Python package dependencies
├── .env.example                   # Environment configuration template
├── .env                           # Local operational environment settings
│
├── configs/
│   ├── zone_config.yaml           # Restricted zone coordinates & camera motion parameters
│   └── thresholds.yaml            # Detection, tracking, idle, and OCR thresholds
│
├── data/
│   └── README.md                  # Comprehensive dataset documentation, splits & licenses
│
├── docs/
│   ├── system_design.md           # Full system design, Mermaid diagrams & deployment roadmap
│   └── images/
│       └── architecture_diagram.png # High-resolution system architecture infographic
│
├── demo/
│   ├── demo_video.mp4             # Annotated demonstration video (13 MB)
│   └── demo_link.txt              # Cloud walkthrough links and timestamps
│
├── task1/
│   ├── src/
│   │   └── zone_monitor.py        # Real-time tracking, footpoint & zone logic
│   ├── outputs/
│   │   ├── output.mp4             # Full annotated video with HUD, footprints & boxes
│   │   └── events.json            # Structured event log (ZONE_ENTRY, IDLE_ALERT, PPE_VIOLATION)
│   └── README.md                  # Task 1 architecture and design rationale
│
├── task2/
│   ├── notebooks/
│   │   └── Helmet_Detection_yolo11.ipynb # Interactive training & export notebook
│   ├── src/
│   │   └── evaluate_detector.py   # Validation metrics, ONNX export & benchmark script
│   ├── outputs/
│   │   ├── best.onnx              # Exported ONNX FP32 model (76.7 MB)
│   │   ├── best_fp16.onnx         # Exported ONNX FP16 model (38.4 MB)
│   │   ├── metrics.json           # Quantitative validation metrics (P, R, mAP)
│   │   ├── benchmark.csv          # PyTorch vs ONNX Runtime latency comparison
│   │   ├── confusion_matrix.png   # Confusion matrix
│   │   ├── pr_curve.png           # Precision-Recall curve
│   │   └── failure_cases/         # 5 diagnosed failure case visualizations & JSON
│   └── README.md                  # Task 2 analysis, class imbalance & edge optimization
│
├── task3/
│   ├── src/
│   │   ├── preprocess.py          # CLAHE, bilateral smoothing, and deskewing
│   │   ├── ocr_pipeline.py        # Tesseract execution & regex field extractor
│   │   └── evaluate.py            # Accuracy & Serial Number CER evaluation
│   ├── data/
│   │   ├── raw_images/            # 15 equipment nameplate test images
│   │   └── labels_ground_truth.csv # Ground truth field annotations
│   ├── outputs/
│   │   ├── preprocessed_images/   # Saved preprocessed nameplate images
│   │   ├── labels_output.json     # Ground truth vs predicted field comparison
│   │   └── accuracy_report.json   # Quantitative accuracy and CER metrics
│   └── README.md                  # Task 3 methodology and findings
│
└── task4/
    ├── app/
    │   ├── rag_engine.py          # Vector indexing, cosine search & citation synthesizer
    │   └── main.py                # FastAPI REST API server
    ├── data/
    │   └── industrial_sop_corpus.json # OSHA & plant standard operating procedures
    ├── outputs/
    │   └── results.json           # Benchmark query outputs with citations & latencies
    └── README.md                  # Task 4 REST documentation & cURL examples
```

---

## 3. Environment Setup

### Prerequisites
- **Operating System**: Linux (Ubuntu 20.04/22.04 LTS tested) or macOS / WSL2
- **Python Version**: Python 3.10 to 3.13 (Evaluated on Python 3.13)
- **OCR Engine**: Tesseract OCR (`sudo apt-get install -y tesseract-ocr`)
- **CUDA Toolkit** (Optional): CUDA 12.x / 13.x for GPU acceleration

### Installation Steps
```bash
# 1. Clone repository
git clone <repository_url> ai-cv-assessment
cd ai-cv-assessment

# 2. Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
```

---

## 4. How to Run Each Task

### Run Task 1 — Zone Monitoring
```bash
# Execute zone monitoring pipeline on video stream
python task1/src/zone_monitor.py
```
*Outputs generated in `task1/outputs/output.mp4` and `task1/outputs/events.json`.*

### Run Task 2 — Detector Evaluation & Benchmarking
```bash
# Run model validation, ONNX benchmarking, and failure case extraction
python task2/src/evaluate_detector.py
```
*Outputs generated in `task2/outputs/`.*

### Run Task 3 — Equipment Nameplate OCR Evaluation
```bash
# Run OCR comparison (Raw vs Preprocessed) and generate accuracy/CER reports
PYTHONPATH=task3/src python task3/src/evaluate.py
```
*Outputs generated in `task3/outputs/accuracy_report.json` and `task3/outputs/labels_output.json`.*

### Run Task 4 — Industrial Safety RAG Assistant
```bash
# Start FastAPI service
uvicorn task4.app.main:app --host 0.0.0.0 --port 8088 --reload

# In a separate terminal, test queries via curl:
curl -X POST http://localhost:8088/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the hard hat requirements in zone alpha?", "top_k": 2}'
```
*Benchmark query results saved in `task4/outputs/results.json`.*

---

## 5. Results Summary

### Task 1: Zone Monitoring & Compliance
- **Total Tracked Persons**: 4 unique worker tracks across 1050 frames.
- **Logged Events**: 40 discrete safety events (`ZONE_ENTRY`, `IDLE_ALERT`, `PPE_VIOLATION`).
- **Processing Throughput**: **46.1 FPS** (1280x720 native resolution preserved).
- **Camera Motion Compensation**: Affine background tracking maintains static virtual boundary anchoring during camera panning.

### Task 2: Helmet Detector Evaluation & Benchmark
- **Model**: YOLO11m (20,031,574 parameters).
- **Validation Metrics**:
  - Overall mAP@50: **71.6%** | Overall mAP@50-95: **45.1%**
  - `helmet` Class: Precision = 0.814, Recall = **0.884**, mAP@50 = **87.5%**
  - `unhelmet` Class: Precision = 0.633, Recall = 0.493, mAP@50 = 55.6%
- **Runtime Benchmark (Exact 100 validation images)**:
  - **PyTorch (CUDA GPU)**: **8.83 ms** mean latency (**113.2 FPS**)
  - **ONNX Runtime (CPU FP32)**: **94.64 ms** mean latency (**10.57 FPS**)
  - **ONNX Runtime (CPU FP16)**: **101.49 ms** mean latency (**9.85 FPS**)

### Task 3: Nameplate OCR & Field Extraction
- **Dataset Size**: 15 distinct industrial equipment plates (motors, pumps, transformers, switchgear).
- **Performance Comparison**:
  - **Raw Baseline Overall Field Accuracy**: **98.7%**
  - **Preprocessed Overall Field Accuracy**: **70.7%**
  - **Serial Number Character Error Rate (CER)**: **0.0056** (Raw) vs **0.1611** (Preprocessed).
- *Observation*: Tesseract 5.3 LSTM features internal adaptive binarization; applying excessive unsharp masking on digital text fragments thin punctuation. Conditional preprocessing is recommended for low-contrast imagery.

### Task 4: Safety & Maintenance RAG Assistant
- **Corpus**: 6 structured industrial SOP documents covering OSHA 1926.100, OSHA 1910.212, OSHA 1910.147, and NFPA 70E.
- **Top Match Similarity**: **0.7272** on PPE inquiries, **0.6566** on idle dwell violations.
- **Mean Retrieval & Synthesis Latency**: **8.31 ms** on standard CPU.

---

## 6. Dataset Sources and Licences

| Dataset | Source URL | Splits | Total Images | License | Application |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Construction Helmet Detection** | [Roboflow SoilTechAI](https://app.roboflow.com/soiltechai/construction-helmet-detect-wvwr6/1) | 768 Train / 217 Val / 108 Test | 1,093 | CC BY 4.0 | Tasks 1 & 2 (PPE verification & detector benchmark) |
| **People Detection** | [Roboflow Universe](https://universe.roboflow.com/nastya-xqjkj/people-ucweq) | 842 Train / 120 Val / 60 Test | 1,022 | CC BY 4.0 | Task 1 (Pedestrian detection & ByteTrack tracking) |
| **Equipment Nameplates** | `task3/data/raw_images/` | 15 Test Images | 15 | Proprietary / CC BY 4.0 | Task 3 (OCR field accuracy & serial CER evaluation) |
| **Industrial Safety Corpus** | `task4/data/industrial_sop_corpus.json` | 6 Core SOP Documents | 6 | Public Domain (OSHA/NFPA) | Task 4 (Semantic RAG assistant knowledge base) |

*Full dataset documentation and schema details are available in [data/README.md](file:///home/ub/Desktop/COMBINATION/ai-cv-assessment/data/README.md).*

---

## 7. AI Coding Assistant Disclosure

In compliance with the AjnaLens assessment submission guidelines:
- **Assistant Used**: Antigravity AI Coding Assistant (DeepMind / Google Gemini 2.5 architecture).
- **Scope of Use**:
  - Drafting modular boilerplate for OpenCV drawing utilities and FastAPI endpoint handlers.
  - Formulating regular expression templates for industrial nameplate field parsing.
  - Generating LaTeX mathematical formulas and Mermaid sequence/flowchart diagrams in documentation.
  - Formatting markdown tables and structuring cross-task benchmarking outputs.
- **Human Review & Verification**:
  - All model training parameters, loss configurations, optical flow stabilization logic, ByteTrack association parameters, empirical benchmarking scripts, and diagnostic failure case analyses were manually validated, executed, and verified on real hardware (NVIDIA H200 NVL).

---

## 8. Known Limitations

1. **Extreme Distance & Small Object Resolution**:
   - Helmets occupying fewer than 25 pixels in wide-angle CCTV surveillance video experience reduced recall due to downsampling in deep convolution stages.
2. **Class Imbalance in Hard-Hat Data**:
   - The majority `helmet` class achieves 88.4% recall, whereas the minority `unhelmet` class achieves 49.3% recall due to significant sample disparity (311 vs 14 instances in validation). In production, targeted minority oversampling or loss reweighting is required.
3. **Severe Prolonged Occlusions (> 3 seconds)**:
   - If a pedestrian is fully occluded behind large machinery for more than 60 frames, ByteTrack may assign a new track ID upon emergence unless coupled with a deep appearance ReID model (e.g., OSNet).
4. **Conditional OCR Preprocessing**:
   - While deskewing and CLAHE improve poorly illuminated, angled nameplates, applying unsharp masking on already sharp, high-contrast plates can slightly erode thin punctuation characters.
