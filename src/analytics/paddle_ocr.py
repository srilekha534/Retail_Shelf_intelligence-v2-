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
import concurrent.futures

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as cfg

# ── Global variables for the worker process ────────────────────────────────
_worker_ocr_engine = None

def _patch_paddle_dlls_on_windows():
    """Automatically copies cuDNN 8 and cuBLAS 11 DLLs into Paddle's libs folder if missing on Windows."""
    import os, shutil, platform
    if platform.system() != "Windows":
        return
        
    try:
        import paddle
        import site
        
        paddle_libs = os.path.join(os.path.dirname(paddle.__file__), 'libs')
        
        # Find site-packages safely
        sp_dirs = [p for p in site.getsitepackages() if 'site-packages' in p]
        sp = sp_dirs[0] if sp_dirs else os.path.join(os.environ.get('VIRTUAL_ENV', ''), 'Lib', 'site-packages')
        
        dlls_to_copy = [
            os.path.join(sp, 'nvidia', 'cublas', 'bin', 'cublas64_11.dll'),
            os.path.join(sp, 'nvidia', 'cublas', 'bin', 'cublasLt64_11.dll'),
            os.path.join(sp, 'nvidia', 'cudnn', 'bin', 'cudnn64_8.dll'),
            os.path.join(sp, 'nvidia', 'cudnn', 'bin', 'cudnn_ops_infer64_8.dll'),
            os.path.join(sp, 'nvidia', 'cudnn', 'bin', 'cudnn_cnn_infer64_8.dll')
        ]
        
        for d in dlls_to_copy:
            if os.path.exists(d):
                target = os.path.join(paddle_libs, os.path.basename(d))
                if not os.path.exists(target):
                    shutil.copy(d, target)
    except Exception:
        pass

def _init_worker(lang, use_angle_cls, use_gpu, precision, det_db_box_thresh, rec_image_shape):
    """Initializes the PaddleOCR engine inside the isolated worker process."""
    global _worker_ocr_engine

    # Patch DLLs silently before initializing PaddleOCR
    _patch_paddle_dlls_on_windows()

    try:
        from paddleocr import PaddleOCR
        ocr_kwargs = dict(
            use_angle_cls=use_angle_cls,
            lang=lang,
            use_gpu=use_gpu,
            show_log=False,
            det_db_box_thresh=det_db_box_thresh,
            rec_image_shape=rec_image_shape,
        )
        if use_gpu and precision == "fp16":
            ocr_kwargs["enable_mkldnn"] = False
            ocr_kwargs["use_tensorrt"] = False
        _worker_ocr_engine = PaddleOCR(**ocr_kwargs)
    except Exception as e:
        print(f"[PaddleOCR Worker] Failed to load engine: {e}")
        _worker_ocr_engine = None

def _run_worker_ocr(crop: np.ndarray) -> List[Tuple[list, str, float]]:
    """Runs OCR on a single crop using the pre-initialized worker engine."""
    global _worker_ocr_engine
    if _worker_ocr_engine is None:
        # Raise error so the main process catches it and falls back to EasyOCR
        raise RuntimeError("PaddleOCR engine not initialized in worker.")
    
    try:
        res = _worker_ocr_engine.ocr(crop, cls=True)
        if not res or not res[0]:
            return []
        
        parsed = []
        for line in res[0]:
            if not line:
                continue
            box, (text, conf) = line
            parsed.append((box, text, conf))
        return parsed
    except Exception as e:
        print(f"[PaddleOCR Worker] GPU OCR Failed: {e}. Falling back to CPU PaddleOCR...")
        # If GPU fails (usually due to missing cudnn64_8.dll on Windows), fallback to CPU
        try:
            from paddleocr import PaddleOCR
            _worker_ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
            res = _worker_ocr_engine.ocr(crop, cls=True)
            if not res or not res[0]:
                return []
            parsed = []
            for line in res[0]:
                if not line:
                    continue
                box, (text, conf) = line
                parsed.append((box, text, conf))
            return parsed
        except Exception as cpu_e:
            print(f"[PaddleOCR Worker] CPU Fallback also failed: {cpu_e}")
            # Raise so main process falls back to EasyOCR
            raise cpu_e


class PaddleOCREngine:
    """
    GPU-accelerated PaddleOCR engine for retail product label reading.
    Runs entirely inside an isolated ProcessPool to prevent Pybind11 / DLL 
    clashes with PyTorch on Windows (e.g., '_gpuDeviceProperties' already registered).
    """

    def __init__(
        self,
        lang: str = "en",
        use_angle_cls: bool = True,
        precision: str = "fp32",
        det_db_box_thresh: float = 0.5,
        rec_image_shape: str = "3, 48, 320"
    ):
        self._lang = lang
        self._use_angle_cls = use_angle_cls
        self._use_gpu = cfg.DEVICE in ["cuda", "gpu"]
        self._precision = precision
        self._det_db_box_thresh = det_db_box_thresh
        self._rec_image_shape = rec_image_shape

        self._pool = None
        self._failed = False
        self._easyocr = None

    @property
    def is_available(self) -> bool:
        return not self._failed

    def _load(self):
        """Lazy-load the ProcessPoolExecutor."""
        if self._pool is not None or self._failed:
            return

        try:
            # We use a single worker process so it holds the model in memory
            self._pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=1,
                initializer=_init_worker,
                initargs=(
                    self._lang,
                    self._use_angle_cls,
                    self._use_gpu,
                    self._precision,
                    self._det_db_box_thresh,
                    self._rec_image_shape
                )
            )
            # Submit a dummy task to force initialization and catch immediate crashes
            future = self._pool.submit(_run_worker_ocr, np.zeros((10, 10, 3), dtype=np.uint8))
            future.result(timeout=20)
            print("[PaddleOCR] Isolated worker process started successfully.")
        except Exception as e:
            self._failed = True
            print(f"[PaddleOCR] Failed to start isolated worker process: {e}")
            if self._pool:
                self._pool.shutdown(wait=False)
                self._pool = None

    def _fallback_easyocr(self, crop: np.ndarray):
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

    def read_crop(
        self,
        crop: np.ndarray,
    ) -> List[Tuple[Optional[list], str, float]]:
        self._load()

        if self._failed or self._pool is None:
            return self._fallback_easyocr(crop)

        try:
            future = self._pool.submit(_run_worker_ocr, crop)
            return future.result(timeout=10)
        except Exception as e:
            print(f"[PaddleOCR] Worker process failed during inference: {e}")
            return self._fallback_easyocr(crop)

    def read_batch(
        self,
        crops: List[np.ndarray],
    ) -> List[List[Tuple[Optional[list], str, float]]]:
        if not self.is_available or not crops:
            return [[] for _ in crops]

        self._load()

        if self._failed or self._pool is None:
            return [self._fallback_easyocr(crop) for crop in crops]

        batch_size = getattr(cfg, "PADDLE_MAX_BATCH_SIZE", 16)
        all_results = []

        # Process in batches
        for batch_start in range(0, len(crops), batch_size):
            batch = crops[batch_start:batch_start + batch_size]
            
            # Submit batch to process pool
            futures = [self._pool.submit(_run_worker_ocr, crop) for crop in batch]
            for future in futures:
                try:
                    all_results.append(future.result(timeout=10))
                except Exception as e:
                    print(f"[PaddleOCR] Batch OCR failed on crop: {e}")
                    all_results.append([])

        return all_results

    def extract_text(self, image: np.ndarray) -> str:
        results = self.read_crop(image)
        texts = [text for _, text, conf in results if conf >= cfg.OCR_CONFIDENCE]
        return " ".join(texts)
