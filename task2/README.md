# Task 2 — Fine-Tune, Evaluate and Optimize Detector

## Overview
This module conducts in-depth validation, failure case diagnosis, ONNX export, and runtime benchmarking (PyTorch vs ONNX Runtime FP32/FP16) on the custom trained YOLO11m Construction Helmet detector for the AjnaLens AI/ML & CV Engineer Practical Assessment.

---

## Model Architecture & Weights
- **Model Architecture**: Ultralytics YOLO11m (Medium scale: 125 fused layers, 20,031,574 parameters)
- **Trained Weights**: `/home/ub/Desktop/COMBINATION/runs/helmet_yolo11/yolo11m_helmet_100e/weights/best.pt`
- **Class Mapping**:
  - `0: helmet` (Worker wearing safety helmet / hard hat)
  - `1: unhelmet` (Worker head without safety helmet)

---

## Training Configuration
- **Dataset**: `Construction helmet detect.v1i.yolov11` (768 train, 217 valid, 108 test images)
- **Epochs**: 100
- **Image Resolution**: 640 x 640
- **Batch Size**: 32
- **Optimizer**: AdamW (Initial lr = 0.001667, momentum = 0.9, weight decay = 0.0005)
- **Loss Functions**: Complete IoU + Distribution Focal Loss (DFL) + Binary Cross-Entropy (BCE) classification loss
- **Hardware Acceleration**: Automatic Mixed Precision (AMP) on NVIDIA H200 NVL GPU

---

## Validation Metrics (Exact Evaluated Values)

| Class | Precision (P) | Recall (R) | mAP@50 | mAP@50-95 |
| :--- | :---: | :---: | :---: | :---: |
| **All Classes** | **0.723** | **0.689** | **0.7156 (71.6%)** | **0.4512 (45.1%)** |
| `helmet` | 0.8142 | 0.8842 | 0.8750 (87.5%) | 0.6560 |
| `unhelmet` | 0.6327 | 0.4929 | 0.5560 (55.6%) | 0.2460 |

Generated evaluation artifacts:
- Metrics Summary: `task2/outputs/metrics.json`
- Confusion Matrix: `task2/outputs/confusion_matrix.png`
- Normalized Confusion Matrix: `task2/outputs/confusion_matrix_normalized.png`
- Precision-Recall Curve: `task2/outputs/pr_curve.png`

---

## Optimization & Benchmark Comparison

Both runtimes were evaluated on the **exact same 100 validation images** at 640x640 resolution with normalized letterbox preprocessing:

| Runtime | Precision | Mean Latency (ms) | Throughput (FPS) | mAP@50 |
| :--- | :---: | :---: | :---: | :---: |
| **PyTorch (CUDA GPU)** | FP32/AMP | **8.83 ms** | **113.23 FPS** | **0.7156** |
| **ONNX Runtime (CPU)** | FP32 | **94.64 ms** | **10.57 FPS** | **0.7095** |
| **ONNX Runtime (FP16)** | FP16 | **101.49 ms** | **9.85 FPS** | **0.7095** |

Benchmark table exported to: `task2/outputs/benchmark.csv`

---

## Class Imbalance Analysis & Remediation
The dataset exhibits significant class imbalance: `helmet` instances (311 annotations in validation) heavily outnumber `unhelmet` instances (14 annotations in validation). As a result:
- The model achieves high recall (**88.4%**) and high mAP@50 (**87.5%**) on the majority `helmet` class.
- The model exhibits lower recall (**49.3%**) on the minority `unhelmet` class due to fewer training representations.

### Remediation Strategies:
1. **Class-Balanced Resampling & Minority Augmentation**:
   - Apply oversampling / mosaic repetition specifically to scenes containing `unhelmet` annotations.
   - Utilize targeted synthetic augmentations (e.g., color jitter, head pose jitter, cutmix on head regions).
2. **Loss Weighting & Focal Loss Tuning**:
   - Increase the positive class weight `cls_pw` or focal loss $\gamma$ for class 1 (`unhelmet`) in the classification head to penalize false negatives on bare heads more heavily.

---

## Edge-Device Optimization (Jetson Orin Nano / Mobile Target)
To achieve $\ge 30\text{ FPS}$ sustained on an edge device (e.g., NVIDIA Jetson Orin Nano 8GB, 20W):
1. **TensorRT Quantization (INT8 / FP16)**:
   - Export ONNX to TensorRT engine with calibration cache (`trtexec --onnx=best.onnx --fp16 --int8 --saveEngine=best.engine`). TensorRT fuses Conv+BN+ReLU layers and optimizes kernel selection for Jetson Ampere cores, delivering $4\times$ to $6\times$ latency reduction.
2. **Architecture Downscaling**:
   - For resource-constrained edge budgets, deploy `YOLO11s` (9.4M parameters) or `YOLO11n` (2.6M parameters). YOLO11n achieves $>60\text{ FPS}$ on Orin Nano with minimal loss in helmet detection accuracy.
3. **Dynamic Resolution & Frame Skipping**:
   - In fixed CCTV angles, run full inference every 2nd or 3rd frame (temporal decimation) and use lightweight optical flow or ByteTrack Kalman filter extrapolation for intermediate frames.
4. **Input Canvas Tuning**:
   - If workers occupy significant field of view, lowering input resolution from $640\times 640$ to $480\times 480$ or $512\times 512$ reduces FLOPs by ~40% with negligible mAP degradation.

---

## Failure Case Analysis (5 Real Model Failures)

All failure cases are visualized with Ground Truth (Green) and Model Predictions (Red/Orange) in `task2/outputs/failure_cases/`:

| Case | Image | Failure Category | Explanation |
| :---: | :--- | :--- | :--- |
| **#1** | `-1-1-20250205131212-950550042.jpg` | False Negative (`unhelmet`) | Ground truth bare head missed due to steep camera tilt angle and backlighting making facial/hair features indistinct. |
| **#2** | `-1-2-20250205115617-494978122.jpg` | False Negative (`unhelmet`) | Severe partial occlusion where a beam obstructs the forehead, suppressing activation below detection threshold. |
| **#3** | `-1-20250205110059-16327183.jpg` | False Negative (`unhelmet`) | Worker turned away from camera in low-contrast indoor lighting; minority class pattern lacked sufficient contrast. |
| **#4** | `-1-20250205110139-16727187.jpg` | False Negative (Small `helmet`) | Target helmet in extreme background is only 26 px wide; sub-sampled feature maps in deep layers lose fine spatial detail. |
| **#5** | `-1-20250205110159-16927189.jpg` | False Positive (Background) | Yellow construction equipment panel with curved edge falsely triggered a low-confidence helmet prediction (conf=0.41). |

Structured failure logs saved in: `task2/outputs/failure_cases/failure_cases.json`.

---

## Limitations
1. **Distant Small Objects**: Helmets occupying $<30\text{ px}$ in 720p/1080p surveillance video experience reduced recall.
2. **Minority Class Sparsity**: The `unhelmet` class has lower recall than `helmet` due to dataset imbalance.
3. **Occlusion Scenarios**: Scaffolding and equipment occluding heads require temporal multi-frame aggregation.

---

## Usage Instructions
```bash
# Navigate to project root
cd /home/ub/Desktop/COMBINATION

# Execute Task 2 Evaluation & Benchmarking
python task2/src/evaluate_detector.py
```
Generated outputs will be saved to `task2/outputs/`.
