"""
Evaluation Script for Task 3 — Nameplate OCR Pipeline.

Computes:
- Field-level accuracy for: equipment_id, serial_number, model, voltage, manufacture_date
- Character Error Rate (CER) on serial_number
- Comparative analysis: With Preprocessing vs Without Preprocessing (Raw)
- Exports:
    - task3/outputs/preprocessed_images/
    - task3/outputs/labels_output.json
    - task3/outputs/accuracy_report.json
"""

import os
import csv
import json
from typing import Dict, List, Tuple
import cv2
import numpy as np

from ocr_pipeline import NameplateOCRPipeline
from preprocess import preprocess_nameplate


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate character edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def compute_cer(gt: str, pred: str) -> float:
    """Compute Character Error Rate (CER)."""
    gt = gt.strip()
    pred = pred.strip()
    if not gt:
        return 0.0 if not pred else 1.0
    dist = levenshtein_distance(gt, pred)
    return dist / len(gt)


def run_evaluation(
    data_dir: str = "/home/ub/Desktop/COMBINATION/ai-cv-assessment/task3/data",
    output_dir: str = "/home/ub/Desktop/COMBINATION/ai-cv-assessment/task3/outputs"
):
    csv_file = os.path.join(data_dir, "labels_ground_truth.csv")
    raw_images_dir = os.path.join(data_dir, "raw_images")
    preprocessed_dir = os.path.join(output_dir, "preprocessed_images")
    os.makedirs(preprocessed_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Read ground truth
    ground_truth = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ground_truth.append(row)

    pipeline = NameplateOCRPipeline()

    fields_to_eval = ["equipment_id", "serial_number", "model", "voltage", "manufacture_date"]
    
    results = {
        "raw": {"field_correct": {f: 0 for f in fields_to_eval}, "total_samples": len(ground_truth), "cer_list": []},
        "preprocessed": {"field_correct": {f: 0 for f in fields_to_eval}, "total_samples": len(ground_truth), "cer_list": []}
    }

    labels_output = {}

    print(f"\n========================================================")
    print(f"  Evaluating Task 3 OCR Pipeline ({len(ground_truth)} samples)")
    print(f"========================================================\n")

    for item in ground_truth:
        fname = item["filename"]
        img_path = os.path.join(raw_images_dir, fname)
        
        # 1. Without Preprocessing (Raw)
        raw_res = pipeline.process_image(img_path, apply_preprocessing=False)
        raw_fields = raw_res["fields"]
        
        # 2. With Preprocessing
        prep_res = pipeline.process_image(img_path, apply_preprocessing=True)
        prep_fields = prep_res["fields"]
        
        # Save preprocessed image
        save_path = os.path.join(preprocessed_dir, fname)
        cv2.imwrite(save_path, prep_res["image"])

        # Record structured outputs
        labels_output[fname] = {
            "ground_truth": {f: item[f] for f in fields_to_eval},
            "raw_predictions": raw_fields,
            "preprocessed_predictions": prep_fields
        }

        # Calculate metrics for Raw
        for f in fields_to_eval:
            if raw_fields.get(f, "").strip() == item[f].strip():
                results["raw"]["field_correct"][f] += 1
        raw_cer = compute_cer(item["serial_number"], raw_fields.get("serial_number", ""))
        results["raw"]["cer_list"].append(raw_cer)

        # Calculate metrics for Preprocessed
        for f in fields_to_eval:
            if prep_fields.get(f, "").strip() == item[f].strip():
                results["preprocessed"]["field_correct"][f] += 1
        prep_cer = compute_cer(item["serial_number"], prep_fields.get("serial_number", ""))
        results["preprocessed"]["cer_list"].append(prep_cer)

        print(f"[{fname}]")
        print(f"  GT Serial:   {item['serial_number']}")
        print(f"  Raw Pred:    {raw_fields['serial_number']} (CER: {raw_cer:.2f})")
        print(f"  Prep Pred:   {prep_fields['serial_number']} (CER: {prep_cer:.2f})")

    # Aggregate summaries
    total = len(ground_truth)
    accuracy_report = {
        "dataset_summary": {
            "total_images": total,
            "fields_evaluated": fields_to_eval
        },
        "performance_comparison": {
            "without_preprocessing_raw": {
                "field_accuracies": {
                    f: {
                        "correct": results["raw"]["field_correct"][f],
                        "total": total,
                        "accuracy_percent": round((results["raw"]["field_correct"][f] / total) * 100, 2)
                    } for f in fields_to_eval
                },
                "overall_field_accuracy_percent": round(
                    (sum(results["raw"]["field_correct"].values()) / (total * len(fields_to_eval))) * 100, 2
                ),
                "serial_number_mean_cer": round(float(np.mean(results["raw"]["cer_list"])), 4)
            },
            "with_preprocessing": {
                "field_accuracies": {
                    f: {
                        "correct": results["preprocessed"]["field_correct"][f],
                        "total": total,
                        "accuracy_percent": round((results["preprocessed"]["field_correct"][f] / total) * 100, 2)
                    } for f in fields_to_eval
                },
                "overall_field_accuracy_percent": round(
                    (sum(results["preprocessed"]["field_correct"].values()) / (total * len(fields_to_eval))) * 100, 2
                ),
                "serial_number_mean_cer": round(float(np.mean(results["preprocessed"]["cer_list"])), 4)
            }
        }
    }

    # Save outputs
    labels_file = os.path.join(output_dir, "labels_output.json")
    with open(labels_file, "w", encoding="utf-8") as f:
        json.dump(labels_output, f, indent=2)

    report_file = os.path.join(output_dir, "accuracy_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(accuracy_report, f, indent=2)

    print("\n========================================================")
    print("                EVALUATION SUMMARY")
    print("========================================================")
    print(f"Metric                  | Raw Baseline  | Preprocessed")
    print(f"------------------------+---------------+---------------")
    for f in fields_to_eval:
        raw_acc = accuracy_report["performance_comparison"]["without_preprocessing_raw"]["field_accuracies"][f]["accuracy_percent"]
        prep_acc = accuracy_report["performance_comparison"]["with_preprocessing"]["field_accuracies"][f]["accuracy_percent"]
        print(f"{f:<23} | {raw_acc:>12.1f}% | {prep_acc:>12.1f}%")
    print(f"------------------------+---------------+---------------")
    print(f"Overall Field Accuracy  | {accuracy_report['performance_comparison']['without_preprocessing_raw']['overall_field_accuracy_percent']:>12.1f}% | {accuracy_report['performance_comparison']['with_preprocessing']['overall_field_accuracy_percent']:>12.1f}%")
    print(f"Serial Number Mean CER  | {accuracy_report['performance_comparison']['without_preprocessing_raw']['serial_number_mean_cer']:>13.4f} | {accuracy_report['performance_comparison']['with_preprocessing']['serial_number_mean_cer']:>13.4f}")
    print("========================================================\n")
    print(f"Report saved to: {report_file}")
    print(f"Labels saved to: {labels_file}")
    print(f"Preprocessed images saved to: {preprocessed_dir}\n")


if __name__ == "__main__":
    run_evaluation()
