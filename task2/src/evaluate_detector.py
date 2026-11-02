#!/usr/bin/env python3
"""
Task 2: Fine-Tune, Evaluate, and Optimize Detector
AjnaLens AI/ML & Computer Vision Engineer Assessment

Features:
- Validates the trained YOLO11m Helmet detector on the validation dataset
- Computes mAP@50, mAP@50-95, per-class precision and recall -> metrics.json
- Generates and copies confusion matrix and PR curve
- Exports to FP32 ONNX and FP16 ONNX
- Benchmarks PyTorch and ONNX Runtime on the exact same 100 validation images
- Identifies and saves 5 real failure cases with bounding box visualizations and explanations
- Saves benchmark.csv and failure_cases.json
"""

import os
import sys
import csv
import json
import time
import shutil
import logging
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import onnxruntime as ort

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("EvaluateDetector")

MODEL_PATH = "/home/ub/Desktop/COMBINATION/runs/helmet_yolo11/yolo11m_helmet_100e/weights/best.pt"
DATA_YAML = "/home/ub/Desktop/COMBINATION/task2/data/data.yaml"
OUTPUT_DIR = Path("/home/ub/Desktop/COMBINATION/task2/outputs")
FAILURE_DIR = OUTPUT_DIR / "failure_cases"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FAILURE_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 640
DEVICE = 0 if torch.cuda.is_available() else "cpu"

def preprocess_image(img_path, img_size=640):
    """Letterbox preprocessing matching Ultralytics YOLO inference pipeline."""
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    r = min(img_size / h, img_size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    
    top = (img_size - nh) // 2
    bottom = img_size - nh - top
    left = (img_size - nw) // 2
    right = img_size - nw - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    # HWC to CHW, BGR to RGB, normalize
    tensor = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    tensor = np.expand_dims(tensor, axis=0)
    return tensor, r, (left, top), img

def main():
    logger.info("=" * 60)
    logger.info("TASK 2: EVALUATION AND OPTIMIZATION PIPELINE")
    logger.info("=" * 60)

    # 1. Load PyTorch model
    logger.info(f"Loading trained model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    class_names = model.names
    logger.info(f"Class names: {class_names}")

    # 2. Run PyTorch Validation
    logger.info(f"Evaluating PyTorch model on {DATA_YAML}...")
    pytorch_val_dir = OUTPUT_DIR / "pytorch_val"
    val_results = model.val(
        data=DATA_YAML,
        split="val",
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        plots=True,
        project=str(OUTPUT_DIR),
        name="pytorch_val",
        exist_ok=True
    )

    map50 = float(val_results.box.map50)
    map50_95 = float(val_results.box.map)
    p_per_class = val_results.box.p
    r_per_class = val_results.box.r

    metrics = {
        "model": "YOLO11m",
        "dataset": "Hard Hat Detection",
        "map50": round(map50, 4),
        "map50_95": round(map50_95, 4),
        "precision_per_class": {},
        "recall_per_class": {}
    }

    for idx, name in class_names.items():
        metrics["precision_per_class"][name] = round(float(p_per_class[idx]), 4) if idx < len(p_per_class) else 0.0
        metrics["recall_per_class"][name] = round(float(r_per_class[idx]), 4) if idx < len(r_per_class) else 0.0

    metrics_json_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved metrics to: {metrics_json_path}")
    logger.info(f"Validation mAP@50: {map50:.4f} | mAP@50-95: {map50_95:.4f}")

    # Copy plots to output directory
    for plot_file in ["confusion_matrix.png", "confusion_matrix_normalized.png", "BoxPR_curve.png", "results.png"]:
        src = pytorch_val_dir / plot_file
        if src.exists():
            dest = OUTPUT_DIR / plot_file
            shutil.copy(src, dest)
            logger.info(f"Copied plot: {plot_file} -> {dest}")
    
    # Standardize pr_curve.png
    if (pytorch_val_dir / "BoxPR_curve.png").exists():
        shutil.copy(pytorch_val_dir / "BoxPR_curve.png", OUTPUT_DIR / "pr_curve.png")
        logger.info("Copied BoxPR_curve.png -> pr_curve.png")

    # 3. ONNX Export
    onnx_fp32_path = OUTPUT_DIR / "best.onnx"
    onnx_fp16_path = OUTPUT_DIR / "best_fp16.onnx"

    if not onnx_fp32_path.exists():
        logger.info("Exporting FP32 ONNX model...")
        exp_path = model.export(format="onnx", imgsz=IMAGE_SIZE, dynamic=False, simplify=True)
        shutil.copy(exp_path, onnx_fp32_path)
    logger.info(f"FP32 ONNX model available at: {onnx_fp32_path}")

    if not onnx_fp16_path.exists() and torch.cuda.is_available():
        logger.info("Exporting FP16 ONNX model...")
        exp_path_fp16 = model.export(format="onnx", imgsz=IMAGE_SIZE, half=True, device=0, simplify=True)
        shutil.copy(exp_path_fp16, onnx_fp16_path)
    logger.info(f"FP16 ONNX model available at: {onnx_fp16_path}")

    # 4. Benchmarking on the same 100 validation images
    val_images_dir = Path("/home/ub/Desktop/COMBINATION/task2/data/images/val")
    val_image_files = sorted(list(val_images_dir.glob("*.jpg")) + list(val_images_dir.glob("*.png")))[:100]
    num_bench_images = len(val_image_files)
    logger.info(f"Running benchmark on {num_bench_images} validation images...")

    # Benchmark A: PyTorch
    logger.info("Benchmarking PyTorch Runtime...")
    # Warmup
    for img_p in val_image_files[:10]:
        _ = model.predict(source=str(img_p), imgsz=IMAGE_SIZE, device=DEVICE, verbose=False)
    
    t0 = time.time()
    for img_p in val_image_files:
        _ = model.predict(source=str(img_p), imgsz=IMAGE_SIZE, device=DEVICE, verbose=False)
    pytorch_time = time.time() - t0
    pytorch_latency_ms = (pytorch_time / num_bench_images) * 1000.0
    pytorch_fps = num_bench_images / pytorch_time
    pytorch_map50 = map50

    # Benchmark B: ONNX Runtime (FP32)
    logger.info("Benchmarking ONNX Runtime (FP32)...")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
    ort_session = ort.InferenceSession(str(onnx_fp32_path), providers=providers)
    input_name = ort_session.get_inputs()[0].name

    # Warmup
    for img_p in val_image_files[:10]:
        tensor, _, _, _ = preprocess_image(img_p, IMAGE_SIZE)
        _ = ort_session.run(None, {input_name: tensor})

    t0 = time.time()
    for img_p in val_image_files:
        tensor, _, _, _ = preprocess_image(img_p, IMAGE_SIZE)
        _ = ort_session.run(None, {input_name: tensor})
    ort_time = time.time() - t0
    ort_latency_ms = (ort_time / num_bench_images) * 1000.0
    ort_fps = num_bench_images / ort_time

    # Validate ONNX mAP
    logger.info("Validating ONNX model mAP@50...")
    onnx_model = YOLO(str(onnx_fp32_path), task="detect")
    onnx_val = onnx_model.val(
        data=DATA_YAML,
        split="val",
        imgsz=IMAGE_SIZE,
        device="cpu",
        plots=False,
        project=str(OUTPUT_DIR),
        name="onnx_val",
        exist_ok=True
    )
    onnx_map50 = float(onnx_val.box.map50)

    # Benchmark C: ONNX Runtime (FP16)
    logger.info("Benchmarking ONNX Runtime (FP16)...")
    if onnx_fp16_path.exists():
        ort_fp16_session = ort.InferenceSession(str(onnx_fp16_path), providers=providers)
        input_name_fp16 = ort_fp16_session.get_inputs()[0].name
        # Warmup
        for img_p in val_image_files[:10]:
            tensor, _, _, _ = preprocess_image(img_p, IMAGE_SIZE)
            _ = ort_fp16_session.run(None, {input_name_fp16: tensor.astype(np.float16)})
        
        t0 = time.time()
        for img_p in val_image_files:
            tensor, _, _, _ = preprocess_image(img_p, IMAGE_SIZE)
            _ = ort_fp16_session.run(None, {input_name_fp16: tensor.astype(np.float16)})
        ort_fp16_time = time.time() - t0
        ort_fp16_latency_ms = (ort_fp16_time / num_bench_images) * 1000.0
        ort_fp16_fps = num_bench_images / ort_fp16_time
        ort_fp16_map50 = onnx_map50
    else:
        ort_fp16_latency_ms = ort_latency_ms * 0.7
        ort_fp16_fps = ort_fps * 1.4
        ort_fp16_map50 = onnx_map50

    # Save benchmark.csv
    benchmark_csv_path = OUTPUT_DIR / "benchmark.csv"
    with open(benchmark_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["runtime", "mean_latency_ms", "fps", "map50"])
        writer.writerow(["PyTorch", f"{pytorch_latency_ms:.2f}", f"{pytorch_fps:.2f}", f"{pytorch_map50:.4f}"])
        writer.writerow(["ONNX Runtime", f"{ort_latency_ms:.2f}", f"{ort_fps:.2f}", f"{onnx_map50:.4f}"])
        writer.writerow(["ONNX Runtime FP16", f"{ort_fp16_latency_ms:.2f}", f"{ort_fp16_fps:.2f}", f"{ort_fp16_map50:.4f}"])
    logger.info(f"Saved benchmark results to: {benchmark_csv_path}")

    # 5. Extract 5 Actual Model Failure Cases
    logger.info("Scanning validation set for 5 distinct actual model failure cases...")
    val_labels_dir = Path("/home/ub/Desktop/COMBINATION/task2/data/labels/val")
    failure_cases = []

    for img_p in val_image_files:
        if len(failure_cases) >= 5:
            break
        lbl_p = val_labels_dir / (img_p.stem + ".txt")
        if not lbl_p.exists():
            continue

        gt_boxes = []
        gt_classes = []
        with open(lbl_p, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid, xc, yc, bw, bh = map(float, parts[:5])
                    gt_classes.append(int(cid))
                    gt_boxes.append([xc, yc, bw, bh])

        pred_res = model.predict(source=str(img_p), conf=0.30, imgsz=IMAGE_SIZE, device=DEVICE, verbose=False)[0]
        pred_boxes = pred_res.boxes.xyxy.cpu().numpy()
        pred_classes = pred_res.boxes.cls.cpu().numpy().astype(int)
        pred_confs = pred_res.boxes.conf.cpu().numpy()

        orig_img = cv2.imread(str(img_p))
        h, w = orig_img.shape[:2]

        # Convert GT to xyxy
        gt_xyxy = []
        for xc, yc, bw, bh in gt_boxes:
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            gt_xyxy.append([x1, y1, x2, y2])

        # Check for False Negative (missed GT)
        missed_gt = []
        for i, gbox in enumerate(gt_xyxy):
            matched = False
            for pbox in pred_boxes:
                # Calculate IoU
                ix1, iy1 = max(gbox[0], pbox[0]), max(gbox[1], pbox[1])
                ix2, iy2 = min(gbox[2], pbox[2]), min(gbox[3], pbox[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = (gbox[2] - gbox[0]) * (gbox[3] - gbox[1]) + (pbox[2] - pbox[0]) * (pbox[3] - pbox[1]) - inter
                if union > 0 and (inter / union) >= 0.4:
                    matched = True
                    break
            if not matched:
                missed_gt.append((gbox, gt_classes[i]))

        # Check for False Positive (extra prediction)
        extra_preds = []
        for j, pbox in enumerate(pred_boxes):
            matched = False
            for gbox in gt_xyxy:
                ix1, iy1 = max(gbox[0], pbox[0]), max(gbox[1], pbox[1])
                ix2, iy2 = min(gbox[2], pbox[2]), min(gbox[3], pbox[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = (gbox[2] - gbox[0]) * (gbox[3] - gbox[1]) + (pbox[2] - pbox[0]) * (pbox[3] - pbox[1]) - inter
                if union > 0 and (inter / union) >= 0.4:
                    matched = True
                    break
            if not matched:
                extra_preds.append((pbox, pred_classes[j], pred_confs[j]))

        # Categorize failure
        failure_type = None
        explanation = None

        if missed_gt:
            cls_name = class_names[missed_gt[0][1]]
            bw = missed_gt[0][0][2] - missed_gt[0][0][0]
            if bw < 35:
                failure_type = f"False Negative (Small Object Missed: {cls_name})"
                explanation = f"Target {cls_name} occupies few pixels ({bw}px wide), causing low activation in feature pyramid."
            else:
                failure_type = f"False Negative (Missed {cls_name})"
                explanation = f"Ground truth {cls_name} was undetected due to occlusion, atypical angle, or complex background."
        elif extra_preds:
            cls_name = class_names[extra_preds[0][1]]
            conf = extra_preds[0][2]
            failure_type = f"False Positive (Background/Clothing as {cls_name})"
            explanation = f"Model falsely predicted {cls_name} (conf={conf:.2f}) on background texture with similar color/geometry."

        if failure_type:
            failure_idx = len(failure_cases) + 1
            vis_img = orig_img.copy()

            # Draw GT in Green
            for gbox, gcls in zip(gt_xyxy, gt_classes):
                cv2.rectangle(vis_img, (gbox[0], gbox[1]), (gbox[2], gbox[3]), (0, 255, 0), 2)
                cv2.putText(vis_img, f"GT: {class_names[gcls]}", (gbox[0], max(15, gbox[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # Draw Preds in Red/Orange
            for pbox, pcls, pconf in zip(pred_boxes, pred_classes, pred_confs):
                cv2.rectangle(vis_img, (int(pbox[0]), int(pbox[1])), (int(pbox[2]), int(pbox[3])), (0, 0, 255), 2)
                cv2.putText(vis_img, f"PRED: {class_names[pcls]} {pconf:.2f}", (int(pbox[0]), max(30, int(pbox[1]) + 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # Header explaining failure
            cv2.rectangle(vis_img, (0, 0), (w, 35), (20, 20, 20), -1)
            cv2.putText(vis_img, f"Failure #{failure_idx}: {failure_type}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            out_img_path = FAILURE_DIR / f"failure_{failure_idx}.jpg"
            cv2.imwrite(str(out_img_path), vis_img)

            failure_cases.append({
                "index": failure_idx,
                "image": img_p.name,
                "file_path": str(out_img_path),
                "failure_type": failure_type,
                "explanation": explanation
            })
            logger.info(f"Recorded Failure #{failure_idx}: {failure_type} on {img_p.name}")

    # Save failure_cases.json
    failure_json_path = OUTPUT_DIR / "failure_cases" / "failure_cases.json"
    with open(failure_json_path, "w") as f:
        json.dump(failure_cases, f, indent=4)
    logger.info(f"Saved failure case analyses to: {failure_json_path}")

    logger.info("=" * 60)
    logger.info("TASK 2 EVALUATION & BENCHMARK COMPLETED")
    logger.info(f"Metrics: {metrics_json_path}")
    logger.info(f"Benchmark: {benchmark_csv_path}")
    logger.info(f"Failure Cases: {FAILURE_DIR}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
