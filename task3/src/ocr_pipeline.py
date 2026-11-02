"""
OCR Extraction Pipeline for Industrial Equipment Nameplates.

Extracts:
1. equipment_id
2. serial_number
3. model
4. voltage
5. manufacture_date
"""

import re
from typing import Dict, Any, Optional
import cv2
import numpy as np
import pytesseract

from preprocess import preprocess_nameplate


class NameplateOCRPipeline:
    def __init__(self, tesseract_cmd: Optional[str] = "/usr/bin/tesseract"):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def extract_raw_text(self, img: np.ndarray, psm: int = 6) -> str:
        """Run Tesseract LSTM OCR on given image array."""
        custom_config = f"--oem 3 --psm {psm}"
        text = pytesseract.image_to_string(img, config=custom_config)
        return text

    def parse_fields(self, ocr_text: str) -> Dict[str, str]:
        """
        Extract structured fields from OCR text using multi-pattern regex
        and contextual keyword anchors.
        """
        lines = [line.strip() for line in ocr_text.split("\n") if line.strip()]
        
        extracted = {
            "equipment_id": "",
            "serial_number": "",
            "model": "",
            "voltage": "",
            "manufacture_date": ""
        }
        
        # 1. Equipment ID: EQ-[A-Z]{2,4}-[0-9]{3,5}
        eq_match = re.search(r"\b(EQ-[A-Z0-9]+-[0-9]+)\b", ocr_text)
        if eq_match:
            extracted["equipment_id"] = eq_match.group(1).strip()
        else:
            for line in lines:
                if "EQUIPMENT" in line.upper() or "EQUIP" in line.upper():
                    parts = re.split(r"[:=]\s*", line)
                    if len(parts) > 1:
                        val = re.sub(r"[^A-Z0-9-]", "", parts[1].strip())
                        if val:
                            extracted["equipment_id"] = val
                            break

        # 2. Serial Number: SN-[A-Z0-9-]+
        sn_match = re.search(r"\b(SN-[A-Z0-9]+-[A-Z0-9]+)\b", ocr_text)
        if sn_match:
            extracted["serial_number"] = sn_match.group(1).strip()
        else:
            for line in lines:
                if "SERIAL" in line.upper() or "S/N" in line.upper():
                    parts = re.split(r"[:=]\s*", line)
                    if len(parts) > 1:
                        val = re.sub(r"[^A-Z0-9-]", "", parts[1].strip())
                        if val:
                            extracted["serial_number"] = val
                            break

        # 3. Model: Look after MODEL / TYPE anchor or regex
        for line in lines:
            if "MODEL" in line.upper() or "TYPE" in line.upper():
                parts = re.split(r"[:=]\s*", line)
                if len(parts) > 1:
                    val = parts[1].strip()
                    val = re.sub(r"[|\[\]()]", "", val).strip()
                    # Clean trailing noise
                    val = val.split()[0] if val else ""
                    if val:
                        extracted["model"] = val
                        break
        if not extracted["model"]:
            m_match = re.search(r"\b([A-Z0-9]{2,8}-[A-Z0-9.-]{3,15})\b", ocr_text)
            if m_match and m_match.group(1) != extracted["equipment_id"] and m_match.group(1) != extracted["serial_number"]:
                extracted["model"] = m_match.group(1).strip()

        # 4. Voltage: e.g. 415V, 480V, 230V, 24V, 400V
        v_match = re.search(r"\b([0-9]{2,4}\s*V(?:AC|DC)?)\b", ocr_text)
        if v_match:
            # Normalize whitespace: '415 V' -> '415V'
            extracted["voltage"] = re.sub(r"\s+", "", v_match.group(1).strip())
        else:
            for line in lines:
                if "VOLT" in line.upper():
                    m = re.search(r"([0-9]{2,4}\s*V?)", line)
                    if m:
                        val = m.group(1).strip()
                        if not val.endswith("V"):
                            val += "V"
                        extracted["voltage"] = val
                        break

        # 5. Manufacture Date: YYYY-MM-DD or YYYY/MM/DD
        d_match = re.search(r"\b(20[12][0-9][-/.][0-1][0-9][-/.][0-3][0-9])\b", ocr_text)
        if d_match:
            # Normalize slashes to dashes
            extracted["manufacture_date"] = d_match.group(1).replace("/", "-").replace(".", "-").strip()
        else:
            for line in lines:
                if "MFG" in line.upper() or "DATE" in line.upper():
                    m = re.search(r"(20[12][0-9][-/.][0-1][0-9][-/.][0-3][0-9])", line)
                    if m:
                        extracted["manufacture_date"] = m.group(1).replace("/", "-").replace(".", "-").strip()
                        break

        return extracted

    def process_image(self, img_path: str, apply_preprocessing: bool = True) -> Dict[str, Any]:
        """Process image file and return extracted fields and raw OCR text."""
        raw_bgr = cv2.imread(img_path)
        if raw_bgr is None:
            raise FileNotFoundError(f"Cannot load image: {img_path}")
        
        if apply_preprocessing:
            processed_img = preprocess_nameplate(raw_bgr)
            ocr_text = self.extract_raw_text(processed_img)
            active_img = processed_img
        else:
            ocr_text = self.extract_raw_text(raw_bgr)
            active_img = raw_bgr
            
        fields = self.parse_fields(ocr_text)
        return {
            "fields": fields,
            "raw_text": ocr_text,
            "image": active_img
        }
