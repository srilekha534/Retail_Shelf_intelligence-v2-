# Retail Shelf Intelligence

AI-powered retail shelf monitoring with continual learning, real-time OCR product identification, zone-based anomaly alerts, and a Next.js dashboard.

---

## Features

| Feature | Description | Implementation |
|---------|-------------|----------------|
| **Product Detection** | Detects retail products on crowded shelves | YOLOv8m fine-tuned on SKU-110K |
| **Product Counting** | Counts products per-class and per-shelf-zone | Custom zoning & spatial counters |
| **OCR Product Identification**| Identifies specific brands/SKUs via text recognition | PaddleOCR (GPU, DB + SVTR_LCNet) + Brand catalog matcher |
| **Anomaly Detection** | Alerts on empty shelves, low stock, and misplaced products | Rule-based heuristics + ML Isolation Forest |
| **Continual Learning** | Learns new product phases without forgetting old ones | Experience replay buffer + generic N-phase orchestrator |
| **Database & History** | Persists metrics and inventory for historical analysis | SQLite Database |
| **MLOps Tracking** | Logs training curves and performance metrics | Weights & Biases (WandB) |
| **REST API** | Endpoints for detection, analytics, and training | FastAPI + background worker queue |
| **Dashboard** | Visualizes shelf share, product heatmaps, and metrics | Next.js (React) + CSS Modules |

---

## 📖 Complete Documentation
For a **very detailed** technical breakdown of every single module and how data flows through the system, please read the [PROJECT MANUAL](PROJECT_MANUAL.md).

---

## Tech Stack

*   **Model/Training:** YOLOv8 (Ultralytics) + PyTorch
*   **Image Processing:** OpenCV + Pillow
*   **Text Recognition:** PaddleOCR (GPU accelerated, DB + SVTR_LCNet)
*   **API Framework:** FastAPI + Uvicorn
*   **Interactive Dashboard:** Next.js App Router (React)
*   **Database:** SQLite3
*   **MLOps:** Weights & Biases (WandB)
*   **Testing & Verification:** pytest

---

## Directory Structure

```
retail-shelf-intelligence/
├── config.py                          # Central configuration for the entire project
├── requirements.txt                   # Project dependencies

│
├── data/
│   ├── raw/                           # Raw SKU-110K dataset (CSVs + images)
│   ├── processed/                     # Prepared YOLOv8 train/val splits
│   ├── phase2/                        # Phase 2 images & CSV annotations
│   └── replay_buffer/                 # Reservoir for experience replay
│
├── models/
│   ├── checkpoints/                   # Trained model checkpoints
│   │   ├── best.pt                    # Latest overall best checkpoint
│   │   ├── last.pt                    # Latest overall last checkpoint
│   │   ├── phase1/                    # Phase 1 checkpoints (best1.pt & last1.pt)
│   │   └── phase2/                    # Phase 2 checkpoints (best2.pt & last2.pt)
│   └── configs/
│       └── data_kaggle.yaml           # Dataset path configuration for YOLO
│
│
├── src/
│   ├── database/
│   │   └── db.py                      # SQLite connection & metric logging
│   ├── detection/
│   │   ├── detector.py                # YOLOv8 inference wrapper
│   │   ├── counter.py                 # Spatial product counter & zone-based statistics
│   │   └── train.py                   # Phase 1 base model training script
│   ├── anomaly/
│   │   ├── rules.py                   # Rule-based heuristics (low stock, empty shelf)
│   │   └── ml_detector.py             # Machine learning anomaly model
│   ├── continual_learning/
│   │   ├── replay_buffer.py           # Reservoir sampling replay storage
│   │   ├── trainer.py                 # Phase-based incremental fine-tuning
│   │   └── active_learning.py         # Pool-based uncertainty query strategy
│   └── utils/
│       ├── image_utils.py             # Shared OpenCV/Pillow drawing helpers
│       └── prepare_dataset.py         # SKU-110K CSV to YOLO txt converter
│
├── scripts/
│   └── run_continual_training.py      # Automated N-Phase Continual Learning Orchestrator
│
├── api/
│   └── main.py                        # FastAPI REST service & background tasks
├── frontend/                          # Next.js React Dashboard
└── tests/
    ├── test_detector.py               # Detector & zoning unit tests
    ├── test_api.py                    # API and OCR integration test
    ├── test_identifier.py             # Product name matching verification
    ├── test_model_accuracy.py         # Multi-checkpoint comparison tool
    └── evaluate.py                    # Catastrophic forgetting analysis script
```

---

## Quickstart

### 1. Install Dependencies
Set up a python virtual environment and install requirements:
```bash
python -m venv venv
# macOS / Linux:
source venv/bin/activate
# Windows (PowerShell):
.\venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Download and Extract SKU-110K
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/eg4000/sku110k-cvpr19) or the original [drive mirror](https://drive.google.com/file/d/1iq93lCdhaPUN0fWbLieMtzfB1850pKwd).
Extract so that the CSV annotations and raw images are in `data/raw/`:
```
data/raw/
    images/
        train/   # .jpg files
        val/     # .jpg files
    annotations/
        annotations_train.csv
        annotations_val.csv
```

### 3. Prepare Dataset
Convert CSV annotations into normalized YOLO format `.txt` files:
```bash
python -m src.utils.prepare_dataset
```
*Note: You can control the subset size in `config.py` using `SUBSET_SIZE` (default is 1500 for fast training/validation).*

### 4. Train the Base Model (Phase 1)
```bash
python -m src.detection.train
```
This automatically uses the best backend (`cuda` for NVIDIA GPU, `mps` for Apple Silicon, or `cpu`).
The weights are saved to `models/checkpoints/phase1/best1.pt` and copied as `best.pt` in the root checkpoints folder.

### 5. Run Validation and Verification
Start the REST API:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Run the Next.js dashboard interface:
```bash
cd frontend
npm install
npm run dev
```

---

## Continual Learning Workflow

Incremental fine-tuning allows the model to learn new shelf products (e.g. Phase 2) while mixing in reservoir samples of Phase 1 products to prevent catastrophic forgetting.

### Step 1: Seed Replay Buffer (Phase 1)
Populate the experience replay buffer with Phase 1 training samples:
```bash
python -c "
from src.continual_learning.trainer import IncrementalTrainer
t = IncrementalTrainer()
t.seed_buffer_from_phase(phase=1)
"
```

### Step 2: Incremental Training (Any Phase)
Place new phase images in `data/phaseN/images/` and annotation labels in `data/phaseN/labels/`.
Then run the automated continual learning orchestrator:
```powershell
# To train a specific phase:
python scripts/run_continual_training.py --phase 2

# To automatically scan and train all untrained phases sequentially:
python scripts/run_continual_training.py --all
```
*Note: The orchestrator automatically handles seeding the Replay Buffer, setting the WandB environment variables, and updating the global `best.pt`.*

---

## Active Learning Workflow

Active learning allows you to query a pool of raw, unlabeled images to find the ones that the model is most uncertain about. Annotating and training on these hard cases yields a much larger accuracy boost than randomly selecting images.

### Query Uncertain Images
Run the active learning script by specifying the pool folder and an optional output directory to copy the selected images:
```bash
python -m src.continual_learning.active_learning \
    --images_dir path/to/unlabeled_pool \
    --query_size 20 \
    --method entropy \
    --output_dir data/to_annotate
```
*Note: Available methods are `entropy` (default), `margin` (confused top-2 classes), `least_confident` (unsure about max class), and `detection_count` (abnormal counts).*

---

## Model Evaluation and Forgetting Analysis

Use the following scripts to evaluate and verify your checkpoints:

### 1. Measure Catastrophic Forgetting
Compare the baseline model (Phase 1) to the fine-tuned model (Phase 2) on the original validation split:
```powershell
# Windows (PowerShell):
python tests/evaluate.py `
    --before models/checkpoints/phase1/best1.pt `
    --after models/checkpoints/phase2/best2.pt

# Linux / macOS:
python tests/evaluate.py \
    --before models/checkpoints/phase1/best1.pt \
    --after models/checkpoints/phase2/best2.pt
```

### 2. Compare All Discovered Checkpoints
Discover and compare the accuracy and inference speed of all trained model variants:
```bash
python tests/test_model_accuracy.py --all
```
This outputs a clean comparison table:
```
MODEL ACCURACY EVALUATION RESULTS
Sorted by mAP50 (higher is better)
Note: '*' indicates the best performer in that column.
================================================================================
┌─────────────────┬──────────┬──────────┬───────────┬──────────┬─────────────────┬────────────┐
│ Model Name      │ mAP50    │ mAP50-95 │ Precision │ Recall   │ Inference Speed │ Status     │
├─────────────────┼──────────┼──────────┼───────────┼──────────┼─────────────────┼────────────┤
│ phase1/last1.pt │ 0.8339 * │ 0.4705   │ 0.8833 *  │ 0.7861 * │ 13.4 ms         │ Best Model │
│ phase1/best1.pt │ 0.8244   │ 0.4720 * │ 0.8816    │ 0.7778   │ 13.4 ms         │ Ready      │
│ best.pt         │ 0.1822   │ 0.0602   │ 0.3602    │ 0.2359   │ 11.1 ms         │ Ready      │
│ phase2/best2.pt │ 0.1822   │ 0.0602   │ 0.3602    │ 0.2359   │ 11.1 ms         │ Ready      │
│ last.pt         │ 0.0601   │ 0.0188   │ 0.1519    │ 0.1478   │ 12.7 ms         │ Ready      │
│ phase2/last2.pt │ 0.0601   │ 0.0188   │ 0.1519    │ 0.1478   │ 12.9 ms         │ Ready      │
└─────────────────┴──────────┴──────────┴───────────┴──────────┴─────────────────┴────────────┘
```

---

## Configuration Settings

All hyperparameter values, thresholds, and options are located centrally in `config.py`. Key settings include:
*   `SUBSET_SIZE`: Size of prepared image subset (e.g. `1500`).
*   `EPOCHS`: Initial base training epochs (e.g. `100`).
*   `BATCH_SIZE`: Training batch size (e.g. `4` for safe default on 4GB VRAM).
*   `REPLAY_BUFFER_MAX_SIZE`: Maximum number of items in experience replay reservoir.
*   `REPLAY_SAMPLE_SIZE`: Number of old phase samples mixed in during incremental training.
#   R e t a i l _ S h e l f _ i n t e l l i g e n c e - v 2 -  
 #   R e t a i l _ S h e l f _ i n t e l l i g e n c e - v 2 -  
 #   R e t a i l _ S h e l f _ i n t e l l i g e n c e - v 2 -  
 