# src/analytics/ocr.py
#
# PURPOSE:
#   Extract text from shelf images (price tags, product labels, brand names)
#   using PaddleOCR. Results are paired with nearest detection bounding box.

import sys
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as cfg


@dataclass
class OCRResult:
    """Single OCR text detection."""
    text: str
    confidence: float
    bbox: List[int]         # [x1, y1, x2, y2]
    nearest_product: Optional[dict] = None  # nearest detection box

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 3),
            "bbox": self.bbox,
            "nearest_product": self.nearest_product,
        }


class ShelfOCR:
    """
    OCR reader for retail shelf images.

    Uses PaddleOCR for text extraction, then links detected text
    to the nearest product bounding box.
    """

    def __init__(self, languages: List[str] = None, gpu: bool = None):
        self.languages = languages or cfg.OCR_LANGUAGES
        self._engine = None
        self._gpu = gpu if gpu is not None else (cfg.DEVICE == "cuda")

    @property
    def engine(self):
        """Lazy-load PaddleOCR engine."""
        if self._engine is None:
            from src.analytics.paddle_ocr import PaddleOCREngine
            self._engine = PaddleOCREngine()
        return self._engine

    def read(
        self,
        image: np.ndarray,
        min_confidence: float = None,
    ) -> List[OCRResult]:
        """
        Extract text from an image.

        Args:
            image: RGB numpy array
            min_confidence: minimum confidence threshold

        Returns:
            List of OCRResult objects
        """
        min_confidence = min_confidence or cfg.OCR_CONFIDENCE

        raw_results = self.engine.read_crop(image)
        results = []

        for bbox_points, text, conf in raw_results:
            if conf < min_confidence:
                continue

            # Convert polygon to [x1, y1, x2, y2]
            if bbox_points is not None and isinstance(bbox_points, list) and len(bbox_points) >= 4:
                xs = [p[0] for p in bbox_points]
                ys = [p[1] for p in bbox_points]
                bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
            else:
                bbox = [0, 0, 0, 0]

            results.append(OCRResult(
                text=text.strip(),
                confidence=conf,
                bbox=bbox,
            ))

        return results

    def read_with_products(
        self,
        image: np.ndarray,
        detections: List[dict],
        min_confidence: float = None,
    ) -> List[OCRResult]:
        """
        Extract text and link each to the nearest product detection.

        Args:
            image:      RGB numpy array
            detections: list of dicts with x1/y1/x2/y2/class_name keys
            min_confidence: OCR confidence threshold

        Returns:
            List of OCRResult with nearest_product filled in
        """
        ocr_results = self.read(image, min_confidence)

        for ocr in ocr_results:
            ocr_cx = (ocr.bbox[0] + ocr.bbox[2]) / 2
            ocr_cy = (ocr.bbox[1] + ocr.bbox[3]) / 2

            best_dist = float("inf")
            best_det = None

            for det in detections:
                det_cx = (det["x1"] + det["x2"]) / 2
                det_cy = (det["y1"] + det["y2"]) / 2
                dist = ((ocr_cx - det_cx) ** 2 + (ocr_cy - det_cy) ** 2) ** 0.5

                if dist < best_dist:
                    best_dist = dist
                    best_det = det

            ocr.nearest_product = best_det

        return ocr_results

    def extract_prices(self, ocr_results: List[OCRResult]) -> List[dict]:
        """
        Filter OCR results to find price-like patterns.

        Returns list of {text, value, bbox, product}
        """
        import re
        prices = []
        price_pattern = re.compile(r'[£$€₹]?\s*\d+[.,]\d{2}')

        for ocr in ocr_results:
            matches = price_pattern.findall(ocr.text)
            for match in matches:
                # Extract numeric value
                numeric = re.sub(r'[^\d.,]', '', match).replace(',', '.')
                try:
                    value = float(numeric)
                    prices.append({
                        "text": match.strip(),
                        "value": value,
                        "bbox": ocr.bbox,
                        "product": ocr.nearest_product,
                    })
                except ValueError:
                    continue

        return prices
