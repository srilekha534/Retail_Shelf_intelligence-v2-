# Retail Shelf Intelligence - Comprehensive Project Manual

This document provides a highly detailed, component-by-component breakdown of the entire Retail Shelf Intelligence architecture, how data flows through the system, and how each module interacts.

---

## 1. System Architecture Overview

The system is designed as a closed-loop AI pipeline for retail environments. It encompasses data ingestion, object detection (YOLOv8), text recognition (OCR), statistical anomaly detection (Heuristics & Isolation Forests), continual learning (Replay Buffers), and MLOps tracking (SQLite & WandB).

The pipeline operates in two primary modes:
1. **Inference Mode**: Processing images via the API/Dashboard to return shelf analytics.
2. **Training Mode**: Fine-tuning the YOLOv8 model on new data incrementally.

---

## 2. Core Inference Pipeline (`api/main.py`)

When an image is sent to the `POST /detect` endpoint, the orchestrator (`api/main.py`) executes a synchronous pipeline of micro-services.

### A. Detection (`src/detection/detector.py`)
- **Model**: Ultralytics YOLOv8 Medium (`yolov8m.pt`).
- **Configuration**: 
  - `imgsz=1280` (High resolution for dense shelf products).
  - `half=True` (FP16 inference for GPU acceleration).
  - `augment=True` (Test-Time Augmentation for higher recall).
- **Process**: The image is passed to the YOLOv8 model which returns a list of bounding boxes (x1, y1, x2, y2, confidence, class_id).

### B. Spatial Counting (`src/detection/counter.py`)
- **Process**: Takes the bounding boxes and divides the image height into virtual "Shelf Zones" (e.g., Top Shelf, Middle Shelf, Bottom Shelf).
- **Output**: Returns global counts and per-zone counts to understand stock distribution vertically.

### C. Anomaly Detection (`src/anomaly/rules.py` & `ml.py`)
- **Rule-Based**: Checks for static heuristics (e.g., if total products < 10, trigger "Empty Shelf" alert; if a zone has 0 products, trigger "Zone Empty").
- **Machine Learning (Isolation Forest)**: Analyzes statistical features (total products, average confidence, spatial distribution) against historical distributions to flag subtle anomalies (e.g., highly unusual product density).

### D. Product Identification (`src/analytics/product_identifier.py`)
- **Model**: PaddleOCR (GPU accelerated, DB text detection + SVTR_LCNet recognition).
- **Preprocessing**: Crops every bounding box, then applies a full pipeline: Resize (short side 512) → Denoise (FastNlMeans) → Contrast Enhance (CLAHE) → Sharpen (Unsharp Mask) → Normalize.
- **OCR**: PaddleOCR runs text detection (DB algorithm) and recognition (SVTR_LCNet) with FP16 mixed precision on GPU.
- **Matching**: Matches the extracted text against a predefined known inventory list (e.g., "Coca-Cola", "Pepsi", "Colgate") using rapidfuzz for fast fuzzy matching.

### E. Persistence (`src/database/db.py`)
- **Database**: SQLite (`data/retail_intelligence.db`).
- **Process**: The final JSON payload containing bounding boxes, zone counts, identified products, and anomalies is permanently logged into the database. This allows for historical tracking of inventory over time.

---

## 3. The Continual Learning Pipeline (`src/continual_learning/`)

Traditional AI models suffer from "Catastrophic Forgetting" — if you train them on Phase 2 data, they forget Phase 1 data. This project solves this using a custom Continual Learning orchestrator.

### A. Replay Buffer (`replay_buffer.py`)
- A reservoir sampling system that securely holds a subset of previous training data (e.g., 2,000 images from Phase 1).
- When a new phase arrives, the buffer provides random historical samples to mix with the new data.

### B. The Unified Orchestrator (`scripts/run_continual_training.py`)
- Scans the `data/` directory for untrained phases (e.g., `data/phase2`, `data/phase3`).
- **Workflow**:
  1. Identifies `phaseN` is untrained.
  2. Seeds the Replay Buffer with Phase 1 data (if empty).
  3. Mixes `phaseN` data with the Buffer.
  4. Runs YOLOv8 fine-tuning on the mixed dataset.
  5. Injects `phaseN` data into the Buffer for future iterations.
  6. Saves `bestN.pt` to `models/checkpoints/phaseN/` and updates the global `best.pt`.

### C. MLOps Tracking (Weights & Biases)
- The pipeline natively passes the `WANDB_PROJECT` environment variable.
- All training metrics (loss, mAP, precision, recall) are beamed to the W&B cloud dashboard for tracking model degradation or improvement across phases.

---

## 4. Active Learning (`src/continual_learning/active_learning.py`)

To efficiently collect new data for Phase 2/3/4, the system uses an Active Learning query strategy.

- **Process**: Given a folder of unlabeled testing images, the Active Learner scores them based on model uncertainty.
- **Strategies**:
  - **Entropy**: High variance in confidence.
  - **Least Confident**: The model's maximum confidence is unusually low (e.g., 60-70%).
- **Outcome**: It returns the top `N` most confusing images. The human administrator can then manually annotate these difficult edge cases, drop them into a new Phase folder, and run the Continuous Learning script to make the model bulletproof.

---

## 5. Visual Analytics (`frontend/src/app/page.tsx` & `src/analytics/`)

The Next.js React dashboard is the primary UI for interacting with the backend.

- **Heatmaps**: Uses 2D Gaussian blur kernels over bounding box centers to generate red-hot density maps (`src/analytics/heatmap.py`).
- **Gap Maps**: Inverts the heatmap to highlight "cold" zones, indicating physical shelf gaps where products are missing.
- **Shelf Share**: Calculates the physical pixel area consumed by each product brand (via the OCR engine) to generate "Share of Shelf" pie charts.

---

## 6. Configuration Management (`config.py`)

The absolute source of truth for the entire pipeline.
- Defines hardware usage (`DEVICE = "cuda"` vs `"mps"`).
- Sets threshold tolerances (`CONFIDENCE_THRESHOLD = 0.25`, `OCR_CONFIDENCE = 0.4`).
- Defines file paths and directory structures for models, checkpoints, and databases.
- Sets hyper-parameters for EWC (Elastic Weight Consolidation) and Replay Buffer sizes.
