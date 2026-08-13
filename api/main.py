# api/main.py
#
# PURPOSE:
#   REST API that wraps the detector, counter, and anomaly detector.
#   The Streamlit dashboard calls these endpoints.
#
# ENDPOINTS:
#   POST /detect              — run detection on an uploaded image
#   POST /detect/batch        — batch detection on multiple images
#   GET  /health              — sanity check
#   GET  /buffer/stats        — replay buffer statistics
#   POST /buffer/seed         — seed buffer from processed dataset
#   POST /continual/train     — trigger incremental fine-tuning
#   GET  /analytics/heatmap   — heatmap data for last detection
#   GET  /analytics/shelf-share — shelf share for last detection
#   POST /analytics/ocr       — extract text from shelf image
#   POST /anomaly/ml/train    — train ML anomaly model
#   POST /active-learning/query — query uncertain samples

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True) 

# USAGE:
#   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

import io
import os
import sys
import time
import uuid
import cv2
import base64
from pathlib import Path
from typing import List, Optional, Dict, Any
from collections import deque
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg

import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.detection.detector import ShelfDetector
from src.detection.counter import ProductCounter
from src.anomaly.rules import AnomalyDetector
from src.continual_learning.replay_buffer import ReplayBuffer
from src.continual_learning.trainer import IncrementalTrainer
from src.database import db

# Initialize the SQLite Database
db.init_db()


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title        = cfg.DASHBOARD_TITLE,
    description  = "Retail shelf monitoring API with continual learning",
    version      = "2.0.0",
)

# Mount history images directory
history_images_dir = Path("data/history_images")
history_images_dir.mkdir(parents=True, exist_ok=True)
app.mount("/history-images", StaticFiles(directory="data/history_images"), name="history_images")

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Lazy-initialise heavy components once ─────────────────────────────────────

_state: Dict[str, Any] = {}
_detection_history: deque = deque(maxlen=500)  # store recent results for ML training
_job_queue: Dict[str, dict] = {}               # async job tracking
_executor = ThreadPoolExecutor(max_workers=cfg.ASYNC_WORKERS)

def get_detector() -> ShelfDetector:
    if "detector" not in _state:
        weights = cfg.BEST_WEIGHTS if os.path.exists(cfg.BEST_WEIGHTS) else cfg.MODEL_NAME
        _state["detector"] = ShelfDetector(model_path=weights, device=cfg.DEVICE)
    return _state["detector"]

def get_counter() -> ProductCounter:
    if "counter" not in _state:
        _state["counter"] = ProductCounter()
    return _state["counter"]

def get_anomaly_detector() -> AnomalyDetector:
    if "anomaly" not in _state:
        _state["anomaly"] = AnomalyDetector()
    return _state["anomaly"]

def get_buffer() -> ReplayBuffer:
    if "buffer" not in _state:
        _state["buffer"] = ReplayBuffer()
    return _state["buffer"]

def get_dl_anomaly():
    if "dl_anomaly" not in _state:
        from src.anomaly.dl_detector import DLAnomalyDetector
        _state["dl_anomaly"] = DLAnomalyDetector()
    return _state["dl_anomaly"]

def get_product_identifier():
    if "identifier" not in _state or not hasattr(_state["identifier"], "brand_catalog"):
        from src.analytics.product_identifier import ProductIdentifier
        _state["identifier"] = ProductIdentifier()
    return _state["identifier"]

def get_planogram_checker():
    if "planogram" not in _state:
        from src.anomaly.planogram import PlanogramChecker
        _state["planogram"] = PlanogramChecker()
    return _state["planogram"]


# ── Response models ───────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
    product_name: Optional[str] = None

class ZoneStats(BaseModel):
    zone_id: int
    count: int

class AnomalyOut(BaseModel):
    type: str
    severity: str
    description: str
    zone_id: int

class DetectionResponse(BaseModel):
    image_width:    int
    image_height:   int
    total_products: int
    counts_by_class: dict
    avg_confidence:  float
    zones:          List[ZoneStats]
    anomalies:      List[AnomalyOut]
    detections:     List[BoundingBox]
    processing_time_ms: float
    hardware_device: Optional[str] = None
    product_inventory: Optional[dict] = None
    image_b64: Optional[str] = None
    image_anomaly_b64: Optional[str] = None
    original_image_path: Optional[str] = None
    processed_image_path: Optional[str] = None
    anomaly_image_path: Optional[str] = None

class AsyncJobResponse(BaseModel):
    job_id: str
    status: str
    message: str


# ── Core detection helper ─────────────────────────────────────────────────────

def _run_detection(img_array: np.ndarray, filename: str = "image.jpg", conf: float = 0.25, ocr_enabled: bool = True, detect_anomalies: bool = True) -> dict:
    """Run full detection pipeline and return response dict."""
    t0 = time.time()
    
    run_id = uuid.uuid4().hex
    original_img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    orig_path = f"data/history_images/{run_id}_original.jpg"
    cv2.imwrite(orig_path, original_img_bgr)

    detector = get_detector()
    result = detector.detect(img_array, conf=conf)
    result.image_path = filename

    counter = get_counter()
    stats = counter.count(result)

    anomaly_detector = get_anomaly_detector()
    anomalies = anomaly_detector.detect(stats)

    # DL Anomaly Detection (Autoencoder)
    dl_anomalies = []
    dl_detector = get_dl_anomaly()
    
    # We will score each detected product if the model is trained
    dl_scores = [0.0] * len(result.detections)
    if dl_detector.is_trained and len(result.detections) > 0:
        crops = []
        for d in result.detections:
            x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
            # Add small padding for crop
            h, w = img_array.shape[:2]
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(w, x2 + pad_x)
            cy2 = min(h, y2 + pad_y)
            crop = img_array[cy1:cy2, cx1:cx2]
            if crop.size > 0 and crop.shape[0] > 0 and crop.shape[1] > 0:
                crops.append(crop)
            else:
                crops.append(np.zeros((256, 256, 3), dtype=np.uint8))
            
        dl_scores = dl_detector.predict_batch(crops)
        
        # Check if any score is above threshold
        for i, score in enumerate(dl_scores):
            if score > dl_detector.threshold:
                # Flag this product as anomalous
                d = result.detections[i]
                dl_anomalies.append({
                    "type": "dl_anomaly",
                    "severity": "high",
                    "description": f"Visual anomaly detected (MSE score: {score:.3f})",
                    "confidence": round(score, 3),
                    "bbox": [d.x1, d.y1, d.x2, d.y2],
                    "zone_id": -1,
                })

    elapsed_ms = (time.time() - t0) * 1000

    det_list = [
        {
            "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2,
            "confidence": round(d.confidence, 4),
            "class_id": d.class_id,
            "class_name": d.class_name,
            "dl_score": round(dl_scores[i], 4) if dl_detector.is_trained else 0.0,
        } for i, d in enumerate(result.detections)
    ]

    # ── Product identification via OCR ────────────────────────────────────
    inv_dict = {"counts": {}, "unique_products": 0, "total_identified": 0, "total_unidentified": len(det_list)}
    
    if ocr_enabled:
        identifier = get_product_identifier()
        inventory = identifier.identify(img_array, det_list)
        inv_dict = inventory.to_dict()

        # Merge OCR-identified names back onto detections
        # Build a map from crop_index -> product name
        name_map = {p.crop_index: p.name for p in inventory.products}
        for i, det in enumerate(det_list):
            det["product_name"] = name_map.get(i, "Unknown")
            
        planogram_checker = get_planogram_checker()
        plano_anomalies = planogram_checker.check_compliance(inventory)
        anomalies.extend(plano_anomalies)
    else:
        for det in det_list:
            det["product_name"] = "Unknown"

    # ── Render bounding boxes ─────────────────────────────────────────────
    img_processed = img_array.copy()
    img_anomaly = img_array.copy()

    # 1. Processed Image: Only green bounding boxes for products
    for i, d in enumerate(result.detections):
        x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
        cv2.rectangle(img_processed, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{d.class_name} {d.confidence:.2f}"
        cv2.putText(img_processed, label, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 2. Anomaly Image: Only red bounding boxes for anomalies
    # Draw DL anomalies
    for i, d in enumerate(result.detections):
        if dl_detector.is_trained and dl_scores[i] > dl_detector.threshold:
            x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
            cv2.rectangle(img_anomaly, (x1, y1), (x2, y2), (255, 0, 0), 3)
            label = f"Damaged {dl_scores[i]:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_anomaly, (x1, y1 - h - 10), (x1 + w + 10, y1), (255, 0, 0), -1)
            cv2.putText(img_anomaly, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw rule-based anomalies (OOS gaps, missing tags, misplaced, fallen, low stock)
    global_y_offset = 30
    for a in anomalies:
        if hasattr(a, 'detection') and a.detection:
            # Localized anomaly (has bounding box)
            ax1, ay1, ax2, ay2 = int(a.detection.x1), int(a.detection.y1), int(a.detection.x2), int(a.detection.y2)
            cv2.rectangle(img_anomaly, (ax1, ay1), (ax2, ay2), (255, 0, 0), 3)
            label = str(a.anomaly_type.value).replace('_', ' ').upper()
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_anomaly, (ax1, ay1 - h - 10), (ax1 + w + 10, ay1), (255, 0, 0), -1)
            cv2.putText(img_anomaly, label, (ax1 + 5, ay1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            # Global anomaly (no bounding box, e.g. LOW_STOCK)
            label = f"ALERT: {str(a.anomaly_type.value).replace('_', ' ').upper()} - {a.description}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(img_anomaly, (10, global_y_offset - h - 10), (10 + w + 20, global_y_offset + 10), (0, 0, 255), -1)
            cv2.putText(img_anomaly, label, (20, global_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            global_y_offset += h + 30

    # Convert drawn images to base64
    img_processed_bgr = cv2.cvtColor(img_processed, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', img_processed_bgr)
    img_b64 = base64.b64encode(buffer).decode('utf-8')

    img_anomaly_bgr = cv2.cvtColor(img_anomaly, cv2.COLOR_RGB2BGR)
    _, buffer_anomaly = cv2.imencode('.jpg', img_anomaly_bgr)
    anomaly_b64 = base64.b64encode(buffer_anomaly).decode('utf-8')
    
    proc_path = f"data/history_images/{run_id}_processed.jpg"
    cv2.imwrite(proc_path, img_processed_bgr)


    anomaly_path = f"data/history_images/{run_id}_anomaly.jpg"
    cv2.imwrite(anomaly_path, img_anomaly_bgr)

    response_data = {
        "image_width": stats.image_width,
        "image_height": stats.image_height,
        "total_products": stats.total_products,
        "counts_by_class": stats.counts_by_class,
        "avg_confidence": round(stats.avg_confidence, 4),
        "zones": [{"zone_id": z.zone_id, "count": z.count} for z in stats.zones],
        "anomalies": [
            {
                "type": a.anomaly_type.value,
                "severity": a.severity,
                "description": a.description,
                "zone_id": a.zone_id,
            } for a in anomalies
        ] + dl_anomalies,
        "detections": det_list,
        "processing_time_ms": round(elapsed_ms, 2),
        "hardware_device": cfg.DEVICE,
        "product_inventory": {
            "counts_by_name": inv_dict["counts"],
            "unique_products": inv_dict["unique_products"],
            "total_identified": inv_dict["total_identified"],
            "total_unidentified": inv_dict["total_unidentified"],
            "products": inv_dict.get("products", []),
        },
        "image_b64": img_b64,
        "image_anomaly_b64": anomaly_b64,
        "original_image_path": f"/history-images/{run_id}_original.jpg",
        "processed_image_path": f"/history-images/{run_id}_processed.jpg",
        "anomaly_image_path": f"/history-images/{run_id}_anomaly.jpg",
    }

    # Store for ML anomaly training
    _detection_history.append(response_data)

    # Persist the detection to SQLite
    try:
        detection_id = db.log_detection(response_data, orig_path, proc_path)
        response_data["detection_id"] = detection_id
    except Exception as e:
        print(f"Failed to log detection to database: {e}")

    return response_data


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/history")
def get_history(limit: int = 50, offset: int = 0):
    """Retrieve historical detection runs."""
    try:
        history = db.get_history(limit=limit, offset=offset)
        return {"history": history, "total": len(history)}
    except Exception as e:
        print(f"Error fetching history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch history")

@app.delete("/history")
def clear_all_history():
    """Clear all historical detection runs."""
    try:
        db.clear_all_history()
        return {"status": "ok", "message": "All history cleared"}
    except Exception as e:
        print(f"Error clearing history: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear history")

@app.delete("/history/{record_id}")
def delete_history_record(record_id: int):
    """Delete a specific historical detection run."""
    try:
        db.delete_history_record(record_id)
        return {"status": "ok", "message": f"Record {record_id} deleted"}
    except Exception as e:
        print(f"Error deleting history record: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete history record")

@app.get("/health")
def health():
    """Quick sanity check."""
    return {
        "status":  "ok",
        "model":   cfg.MODEL_NAME,
        "weights": cfg.BEST_WEIGHTS if os.path.exists(cfg.BEST_WEIGHTS) else "pretrained",
        "device":  cfg.DEVICE,
        "detection_mode": cfg.DETECTION_MODE,
        "ocr_backend": getattr(cfg, "OCR_BACKEND", "paddleocr"),
        "paddle_gpu": getattr(cfg, "PADDLE_USE_GPU", True),
        "paddle_precision": getattr(cfg, "PADDLE_PRECISION", "fp16"),
        "history_size": len(_detection_history),
        "dl_anomaly_trained": get_dl_anomaly().is_trained,
    }


@app.post("/detect", response_model=DetectionResponse)
async def detect(
    file: UploadFile = File(...),
    confidence: float = Form(0.25),
    ocr_enabled: bool = Form(True),
    detect_anomalies: bool = Form(True)
):
    try:
        """Upload a shelf image, get back detections + anomaly alerts."""
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image (JPEG/PNG)")

        raw = await file.read()
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        img_array = np.array(pil_img)

        return _run_detection(img_array, file.filename or "image.jpg", conf=confidence, ocr_enabled=ocr_enabled, detect_anomalies=detect_anomalies)
    except Exception as e:
        import traceback
        with open("error_log.txt", "w") as f:
            f.write(traceback.format_exc())
        raise e


@app.post("/detect/batch")
async def detect_batch(files: List[UploadFile] = File(...)):
    """
    Batch detection on multiple images.
    Returns a list of detection results, one per image.
    """
    results = []
    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            results.append({"error": f"{file.filename or 'unknown'}: not an image"})
            continue

        raw = await file.read()
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        img_array = np.array(pil_img)

        data = _run_detection(img_array, file.filename or "image.jpg")
        data["filename"] = file.filename or "image.jpg"
        results.append(data)

    return {"results": results, "total_images": len(results)}


@app.post("/detect/async", response_model=AsyncJobResponse)
async def detect_async(file: UploadFile = File(...)):
    """
    Submit an image for async processing.
    Returns a job_id; poll /detect/async/{job_id} for results.
    """
    if len(_job_queue) >= cfg.MAX_QUEUE_SIZE:
        raise HTTPException(status_code=429, detail="Queue full. Try again later.")

    raw = await file.read()
    job_id = str(uuid.uuid4())[:8]

    _job_queue[job_id] = {"status": "processing", "result": None, "filename": file.filename or "image.jpg"}

    def _process():
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        img_array = np.array(pil_img)
        result = _run_detection(img_array, file.filename or "image.jpg")
        _job_queue[job_id] = {"status": "done", "result": result, "filename": file.filename or "image.jpg"}

    _executor.submit(_process)

    return AsyncJobResponse(
        job_id=job_id,
        status="processing",
        message="Job submitted. Poll /detect/async/{job_id} for results.",
    )


@app.get("/detect/async/{job_id}")
def get_async_result(job_id: str):
    """Get the result of an async detection job."""
    if job_id not in _job_queue:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_queue[job_id]


# ── Analytics endpoints ───────────────────────────────────────────────────────

@app.post("/analytics/heatmap")
async def heatmap(file: UploadFile = File(...)):
    """Generate a product density heatmap for an uploaded image."""
    from src.analytics.heatmap import generate_heatmap

    raw = await file.read()
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    img_array = np.array(pil_img)

    data = _run_detection(img_array, file.filename or "image.jpg")

    heat = generate_heatmap(
        data["detections"],
        data["image_width"],
        data["image_height"],
    )

    return {
        "heatmap": heat.tolist(),
        "image_width": data["image_width"],
        "image_height": data["image_height"],
        "total_products": data["total_products"],
    }


@app.post("/analytics/shelf-share")
async def shelf_share(file: UploadFile = File(...)):
    """Calculate shelf share analysis for an uploaded image."""
    from src.analytics.shelf_share import calculate_shelf_share

    raw = await file.read()
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    img_array = np.array(pil_img)

    data = _run_detection(img_array, file.filename or "image.jpg")

    share = calculate_shelf_share(
        data["detections"],
        data["image_width"],
        data["image_height"],
    )

    return {
        "occupancy_rate": round(share.occupancy_rate, 4),
        "occupied_area": share.occupied_area,
        "empty_area": share.empty_area,
        "share_by_class": share.share_by_class,
        "count_by_class": share.count_by_class,
    }


@app.post("/analytics/ocr")
async def ocr_extract(file: UploadFile = File(...)):
    """Extract text (prices, labels) from a shelf image using OCR."""
    from src.analytics.ocr import ShelfOCR

    raw = await file.read()
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    img_array = np.array(pil_img)

    data = _run_detection(img_array, file.filename or "image.jpg")

    ocr = ShelfOCR()
    ocr_results = ocr.read_with_products(img_array, data["detections"])
    prices = ocr.extract_prices(ocr_results)

    return {
        "texts": [r.to_dict() for r in ocr_results],
        "prices": prices,
        "total_texts": len(ocr_results),
        "total_prices": len(prices),
    }


@app.post("/analytics/identify-products")
async def identify_products(file: UploadFile = File(...)):
    """
    Identify products by name using OCR on each detected product region.
    Returns a grouped inventory with counts per unique product.
    """
    raw = await file.read()
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    img_array = np.array(pil_img)

    data = _run_detection(img_array, file.filename or "image.jpg")

    identifier = get_product_identifier()
    inventory = identifier.identify(img_array, data["detections"])

    return {
        "inventory": inventory.to_dict(),
        "total_detections": data["total_products"],
    }


# ── DL Anomaly endpoints (Autoencoder) ─────────────────────────────────────────

@app.post("/anomaly/dl/train")
def train_dl_anomaly(epochs: int = 50, lr: float = 1e-3):
    """Train DL anomaly model (Autoencoder) from normal image crops."""
    train_dir = os.path.join(cfg.DATA_DIR, "train_normal")
    if not os.path.exists(train_dir):
        raise HTTPException(
            status_code=400,
            detail=f"Training directory {train_dir} does not exist. Please place normal images there.",
        )
        
    # Read normal crops
    import cv2
    crops = []
    for filename in os.listdir(train_dir):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            filepath = os.path.join(train_dir, filename)
            img = cv2.imread(filepath)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                crops.append(img)
                
    if len(crops) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 10 normal images in {train_dir}, found {len(crops)}.",
        )

    dl_detector = get_dl_anomaly()
    dl_detector.train(crops, epochs=epochs, lr=lr)
    
    return {
        "status": "trained",
        "observations": len(crops),
        "message": "DL anomaly model trained. Future /detect calls will include visual anomaly scores.",
    }


# ── Active learning endpoint ─────────────────────────────────────────────────

class ActiveLearningRequest(BaseModel):
    images_dir: str
    method: Optional[str] = None
    query_size: Optional[int] = None

@app.post("/active-learning/query")
def active_learning_query(req: ActiveLearningRequest):
    """Find the most uncertain images for labeling."""
    from src.continual_learning.active_learning import ActiveLearner

    if not os.path.exists(req.images_dir):
        raise HTTPException(status_code=400, detail=f"Directory not found: {req.images_dir}")

    learner = ActiveLearner(
        method=req.method or cfg.AL_UNCERTAINTY_METHOD,
        query_size=req.query_size or cfg.AL_QUERY_SIZE,
    )
    detector = get_detector()
    samples = learner.query_from_directory(detector, req.images_dir)

    return {
        "samples": [s.to_dict() for s in samples],
        "total_queried": len(samples),
        "method": learner.method,
    }


# ── Buffer & training endpoints ───────────────────────────────────────────────

@app.get("/buffer/stats")
def buffer_stats():
    """Return replay buffer statistics."""
    return get_buffer().stats()


@app.post("/buffer/seed")
def seed_buffer(phase: int = 1, max_samples: int = 100):
    """Seed the replay buffer from the processed training dataset."""
    trainer = IncrementalTrainer(buffer=get_buffer())
    trainer.seed_buffer_from_phase(phase=phase, max_samples=max_samples)
    return {"status": "ok", "buffer_size": get_buffer().size}


class ContinualTrainRequest(BaseModel):
    new_images_dir: str
    new_labels_dir: str
    class_names:    List[str]
    phase:          int = 2
    epochs:         Optional[int] = None
    use_ewc:        bool = False


@app.post("/continual/train")
def continual_train(req: ContinualTrainRequest, background_tasks: BackgroundTasks):
    """
    Trigger incremental fine-tuning in the background.
    Optionally use EWC to prevent catastrophic forgetting.
    """
    if not os.path.exists(req.new_images_dir):
        raise HTTPException(status_code=400, detail=f"Images dir not found: {req.new_images_dir}")
    if not os.path.exists(req.new_labels_dir):
        raise HTTPException(status_code=400, detail=f"Labels dir not found: {req.new_labels_dir}")

    trainer = IncrementalTrainer(buffer=get_buffer())

    def run_training():
        trainer.train_new_phase(
            new_images_dir = req.new_images_dir,
            new_labels_dir = req.new_labels_dir,
            class_names    = req.class_names,
            phase          = req.phase,
            epochs         = req.epochs,
        )
        # Reload detector with updated weights
        _state["detector"] = ShelfDetector(
            model_path=cfg.BEST_WEIGHTS, device=cfg.DEVICE
        )
        print("[API] Detector reloaded with updated weights.")

    background_tasks.add_task(run_training)

    return {
        "status":  "training_started",
        "phase":   req.phase,
        "use_ewc": req.use_ewc,
        "message": "Fine-tuning running in background.",
    }
