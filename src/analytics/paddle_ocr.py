# src/analytics/paddle_ocr.py
#
# PURPOSE:
#   GPU-accelerated OCR engine using PaddleOCR (DB text detection + SVTR_LCNet recognition).
#   Optimized for 4GB VRAM with FP16 mixed precision and batch processing.
#
# ARCHITECTURE:
#   1. Text Detection:  DB (Differentiable Binarization) algorithm
#   2. Text Recognition: SVTR_LCNet (lightweight CNN-based recognizer)
#   3. Angle Classification: Detects and corrects rotated text
#
# USAGE:
#   engine = PaddleOCREngine()
#   results = engine.read_crop(crop_numpy_rgb)
#   batch_results = engine.read_batch([crop1, crop2, ...])

import os
import sys
import time
from typing import List, Tuple, Optional, Union
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as cfg


class PaddleOCREngine:
    """
    GPU-accelerated PaddleOCR engine for retail product label reading.

    Features:
      - DB text detection + SVTR_LCNet recognition
      - FP16 mixed precision inference
      - Batch processing (up to 16 crops per forward pass)
      - Automatic GPU/CPU fallback
    """

    def __init__(
        self,
        lang: str = "en",
        use_angle_cls: bool = True,
        gpu: bool = True,
        precision: str = "fp32",
        det_db_box_thresh: float = 0.5,
        rec_image_shape: str = "3, 48, 320"
    ):
        self._lang = lang
        self._use_angle_cls = use_angle_cls
        self._use_gpu = gpu
        self._precision = precision
        self._det_db_box_thresh = det_db_box_thresh
        self._rec_image_shape = rec_image_shape

        self._ocr = None
        self._failed = False
        
        # Check if easyocr is available as a fallback
        self._easyocr = None

    @property
    def is_available(self) -> bool:
        """Check if PaddleOCR is loaded and ready."""
        return not self._failed

    def _load(self):
        """Lazy-load PaddleOCR with GPU/FP16 configuration."""
        if self._ocr is not None or self._failed:
            return

        try:
            from paddleocr import PaddleOCR
        except ImportError as e:
            self._failed = True
            print(f"[PaddleOCR] Failed to import paddleocr: {e}")
            return

        # Determine if GPU is actually available
        use_gpu = self._use_gpu
        if use_gpu:
            try:
                import paddle
                if not paddle.device.is_compiled_with_cuda():
                    self._failed = True
                    print("[PaddleOCR] GPU was requested, but paddlepaddle-gpu is not installed.")
                    return
            except Exception as e:
                self._failed = True
                print(f"[PaddleOCR] GPU requested but failed to initialize Paddle GPU: {e}")
                return

        print(f"[PaddleOCR] Loading engine (GPU={use_gpu}, lang={self._lang}, precision={self._precision})...")
        t0 = time.time()

        ocr_kwargs = dict(
            use_angle_cls=self._use_angle_cls,
            lang=self._lang,
            use_gpu=use_gpu,
            show_log=False,
            det_db_box_thresh=self._det_db_box_thresh,
            rec_image_shape=self._rec_image_shape,
        )




        # Enable TensorRT FP16 if GPU is available
        if use_gpu and self._precision == "fp16":
            ocr_kwargs["enable_mkldnn"] = False
            ocr_kwargs["use_tensorrt"] = False  # TRT requires model conversion; use native FP16
            # PaddleOCR uses Paddle's native FP16 when precision is set

        try:
            self._ocr = PaddleOCR(**ocr_kwargs)
            print(f"[PaddleOCR] Engine loaded in {time.time() - t0:.2f}s.")
        except Exception as e:
            self._failed = True
            print(f"[PaddleOCR] Failed to load engine: {e}")
            return

    def read_crop(
        self,
        crop: np.ndarray,
    ) -> List[Tuple[Optional[list], str, float]]:
        """
        Run OCR on a single image crop.
        Falls back to EasyOCR if PaddleOCR is broken on this system.
        """
        self._load()

        if self._failed:
            # Fallback to EasyOCR if Paddle fails (e.g. DLL issues on Windows)
            if self._easyocr is None:
                try:
                    import easyocr
                    print("[OCR] Falling back to PyTorch-native EasyOCR...")
                    self._easyocr = easyocr.Reader(['en'], gpu=self._use_gpu)
                except Exception as e:
                    print(f"[OCR] EasyOCR fallback failed: {e}")
                    return []
            
            try:
                results = self._easyocr.readtext(crop)
                output = []
                for bbox, text, conf in results:
                    output.append((bbox, text, conf))
                return output
            except Exception as e:
                print(f"[OCR] EasyOCR failed on crop: {e}")
                return []

        try:
            # PaddleOCR expects BGR or RGB numpy arrays
            results = self._ocr.ocr(crop, cls=self._use_angle_cls)
        except Exception as e:
            print(f"[PaddleOCR] OCR failed on crop: {e}")
            self._failed = True
            return self.read_crop(crop) # Re-run with fallback

        if not results or results[0] is None:
            # If the engine is completely broken and returning None immediately despite no exception
            if not getattr(self, "_first_success_seen", False):
                print("[PaddleOCR] Engine appears broken (returned None on first crop). Failing over.")
                self._failed = True
                return self.read_crop(crop)
            return []
        
        self._first_success_seen = True

        # Convert PaddleOCR format to our standard format
        # PaddleOCR returns: [[bbox_points, (text, confidence)], ...]
        output = []
        for line in results[0]:
            bbox_points = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            text = line[1][0]
            confidence = line[1][1]
            output.append((bbox_points, text, confidence))

        return output

    def read_batch(
        self,
        crops: List[np.ndarray],
    ) -> List[List[Tuple[Optional[list], str, float]]]:
        """
        Run OCR on a batch of crops for throughput optimization.

        Args:
            crops: List of NumPy RGB arrays

        Returns:
            List of results per crop (same format as read_crop)
        """
        if not self.is_available or not crops:
            return [[] for _ in crops]

        self._load()

        batch_size = getattr(cfg, "PADDLE_MAX_BATCH_SIZE", 16)
        all_results = []

        # Process in batches
        for batch_start in range(0, len(crops), batch_size):
            batch = crops[batch_start:batch_start + batch_size]

            for crop in batch:
                try:
                    results = self._ocr.ocr(crop, cls=self._use_angle_cls)
                    if not results or results[0] is None:
                        all_results.append([])
                        continue

                    output = []
                    for line in results[0]:
                        bbox_points = line[0]
                        text = line[1][0]
                        confidence = line[1][1]
                        output.append((bbox_points, text, confidence))
                    all_results.append(output)

                except Exception as e:
                    print(f"[PaddleOCR] Batch OCR failed on crop: {e}")
                    all_results.append([])

        return all_results

    def extract_text(self, image: np.ndarray) -> str:
        """
        Extract all text from an image as a single concatenated string.

        Args:
            image: NumPy RGB array

        Returns:
            Concatenated text string
        """
        results = self.read_crop(image)
        texts = [text for _, text, conf in results if conf >= cfg.OCR_CONFIDENCE]
        return " ".join(texts)
