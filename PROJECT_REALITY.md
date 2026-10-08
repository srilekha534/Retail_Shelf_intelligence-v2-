# PROJECT REALITY: Retail Shelf Intelligence v2

> **Document Status**: Verified against codebase and filesystem on October 7, 2026.  
> **Source of Truth**: Actual repository source files, configuration values, directory contents, and command execution results.  
> **Rule**: No marketing claims. No assumed features. Only what is proven by code and real execution.

---

## 1. What This Project Is

This project is a computer vision web system for retail shelf monitoring. It accepts shelf photos through a web dashboard or REST API, runs a YOLOv8 object detector to locate products, groups them into vertical zones, identifies product names and brands with OCR and fuzzy catalog matching, flags layout and visual anomalies, and logs runs to a SQLite database `(api/main.py, src/detection/detector.py, src/database/db.py)`. It also provides training scripts to fine-tune the detector on new data using experience replay, and an uncertainty query tool for active learning `(src/continual_learning/trainer.py, src/continual_learning/active_learning.py)`.

---

## 2. Real Directory Tree

*Generated from actual filesystem inspection:*

```
Retail_Shelf_intelligence-v2-/
├── .gitignore                          # Git file exclusion patterns
├── config.py                           # Central configuration for paths, device, and thresholds
├── import_trace.txt                    # Python import trace output file
├── PROJECT_MANUAL.md                   # Conceptual architecture document
├── README.md                           # Envisioned project overview and user documentation
├── requirements.txt                    # Python package dependencies
├── setup.py                            # Setuptools package setup file for retail_shelf_intelligence
├── yolo26n.pt                          # YOLO weight file in root (5.5 MB)
├── yolov8l.pt                          # YOLOv8 Large weight file in root (87.8 MB)
├── yolov8m.pt                          # YOLOv8 Medium weight file in root (52.1 MB)
├── yolov8n.pt                          # YOLOv8 Nano weight file in root (6.5 MB)
├── yolov8s.pt                          # YOLOv8 Small weight file in root (22.6 MB)
│
├── api/
│   ├── __init__.py                     # Module package marker
│   └── main.py                         # FastAPI server handling detection, analytics, and history
│
├── data/
│   ├── history_images/                 # Stored copies of original, processed, and anomaly images
│   ├── replay_buffer/                  # Buffer image and label directories (currently empty)
│   │   ├── images/                     # Stored replay image copies (0 files)
│   │   └── labels/                     # Stored replay YOLO label copies (0 files)
│   └── retail_intelligence.db          # SQLite database storing detections, inventory, and anomalies
│
├── frontend/
│   ├── eslint.config.mjs               # ESLint configuration
│   ├── next-env.d.ts                   # Next.js TypeScript declarations
│   ├── next.config.ts                  # Next.js reverse proxy rewrites to FastAPI (port 8000)
│   ├── package.json                    # Node dependencies and scripts (Next.js 16, React 19)
│   ├── package-lock.json               # Locked Node dependency tree
│   ├── tsconfig.json                   # TypeScript configuration
│   └── src/
│       ├── app/
│       │   ├── favicon.ico             # App icon
│       │   ├── globals.css             # Global CSS styles and CSS variables
│       │   ├── layout.tsx              # Root app layout with navigation header
│       │   ├── Layout.module.css       # CSS module for layout header and links
│       │   ├── page.tsx                # Main dashboard page (upload, webcam, visualization)
│       │   ├── page.module.css         # CSS module for main dashboard page
│       │   ├── settings-context.tsx    # React Context for confidence threshold and OCR toggle
│       │   ├── history/
│       │   │   └── page.tsx            # History viewer page querying SQLite detection runs
│       │   └── settings/
│       │       └── page.tsx            # Settings page for detection confidence and OCR toggle
│       └── components/
│           └── dashboard/
│               ├── anomalies-summary.tsx       # Summary card grid for anomaly counts
│               ├── anomalies-visualization.tsx # Visual alert card for anomalies
│               ├── anomaly-list-panel.tsx      # List card of individual anomaly descriptions
│               ├── anomaly-panel.tsx           # Anomaly list with severity color pills
│               ├── AnomalyPanel.module.css     # CSS module for anomaly panel
│               ├── identification-summary.tsx  # Product identification statistics card
│               ├── metrics-bar.tsx             # Top bar showing total items, OOS, and identified
│               ├── MetricsBar.module.css       # CSS module for metrics bar
│               ├── ocr-results-table.tsx       # Table of extracted OCR text strings and confidence
│               └── product-chart.tsx           # Bar chart visualizer for product counts (Recharts)
│
├── models/
│   ├── checkpoints/
│   │   ├── best.pt                     # Active best YOLO weights (155.5 MB, identical to Phase 1)
│   │   ├── best.pt.backup              # Exact backup copy of best.pt (155.5 MB)
│   │   ├── last.pt                     # Git LFS text pointer file (136 bytes, NOT a model)
│   │   ├── phase1/
│   │   │   ├── best1.pt                # Phase 1 best YOLO weights (155.5 MB)
│   │   │   ├── last1.pt                # Phase 1 last epoch YOLO weights (155.5 MB)
│   │   │   └── train/                  # Ultralytics Phase 1 training run folder
│   │   │       ├── args.yaml           # Training argument snapshot
│   │   │       ├── labels.jpg          # Training label distribution plot
│   │   │       ├── results.csv         # 40-epoch Phase 1 training metrics log
│   │   │       ├── train_batch*.jpg    # Mosaic batch inspection images
│   │   │       └── weights/
│   │   │           ├── best.pt         # Phase 1 best checkpoint weights (155.5 MB)
│   │   │           └── last.pt         # Phase 1 last checkpoint weights (155.5 MB)
│   │   └── phase2/
│   │       └── finetune/               # Ultralytics Phase 2 fine-tuning run folder
│   │           ├── args.yaml           # Fine-tuning argument snapshot
│   │           ├── labels.jpg          # Fine-tuning label distribution plot
│   │           ├── results.csv         # 33-epoch Phase 2 fine-tuning metrics log
│   │           ├── train_batch*.jpg    # Mosaic batch inspection images
│   │           └── weights/
│   │               ├── best.pt         # Phase 2 fine-tuned best weights (136.1 MB)
│   │               └── last.pt         # Phase 2 fine-tuned last weights (136.1 MB)
│   └── configs/
│       ├── data_kaggle.yaml            # Dataset YAML pointing to external data/processed path
│       └── train_config.yaml           # Hyperparameters YAML for YOLO training runs
│
├── scripts/
│   └── run_continual_training.py       # CLI runner to scan and train phase folders sequentially
│
├── src/
│   ├── __init__.py                     # Package marker
│   ├── analytics/
│   │   ├── __init__.py                 # Package marker
│   │   ├── heatmap.py                  # 2D Gaussian product density and gap heatmap functions
│   │   ├── ocr.py                      # ShelfOCR class extracting text and regex price patterns
│   │   ├── paddle_ocr.py               # ProcessPool worker for PaddleOCR with EasyOCR fallback
│   │   ├── product_identifier.py       # Product crop preprocessing, brand catalog, and fuzzy match
│   │   └── shelf_share.py              # Boolean mask occupancy and brand shelf-share calculator
│   ├── anomaly/
│   │   ├── __init__.py                 # Package marker
│   │   ├── autoencoder.py              # PyTorch 4-layer Conv Autoencoder definition (256x256)
│   │   ├── dl_detector.py              # DLAnomalyDetector class calculating reconstruction MSE
│   │   ├── feature_extractor.py        # EfficientNet-B0 1280-dim feature extractor (NOT INTEGRATED)
│   │   ├── planogram.py                # Synthetic demo planogram violation checker
│   │   └── rules.py                    # Spatial heuristics (empty shelf, low stock, misplaced, fallen)
│   ├── continual_learning/
│   │   ├── __init__.py                 # Package marker
│   │   ├── active_learning.py          # Uncertainty sampling (entropy, margin, confidence, count)
│   │   ├── ewc.py                      # Elastic Weight Consolidation Fisher calculator (NOT INTEGRATED)
│   │   ├── replay_buffer.py            # Reservoir sampling buffer saving historical images/labels
│   │   └── trainer.py                  # Incremental fine-tuning mixing buffer data and freezing layers
│   ├── database/
│   │   ├── __init__.py                 # Package marker
│   │   └── db.py                       # SQLite tables, connection, detection logger, history queries
│   ├── detection/
│   │   ├── __init__.py                 # Package marker
│   │   ├── counter.py                  # ProductCounter class dividing shelves into horizontal zones
│   │   ├── detector.py                 # ShelfDetector wrapper around Ultralytics YOLOv8
│   │   └── train.py                    # Base Phase 1 YOLOv8 training script
│   └── utils/
│       ├── __init__.py                 # Package marker
│       ├── image_utils.py              # OpenCV/PIL image loading, letterbox resizing, canvas padding
│       ├── prepare_dataset.py          # SKU-110K CSV annotation converter to YOLO format
│       └── visualizer.py               # Bounding box and anomaly zone drawing helpers
│
└── tests/
    ├── evaluate.py                     # CLI tool measuring mAP and catastrophic forgetting
    ├── test_api.py                     # Script testing /detect (FAILS: hardcodes missing val_14.jpg)
    ├── test_detector.py                # 29 Pytest unit tests for detector, counter, buffer (PASSES)
    ├── test_identifier.py              # Script testing ProductIdentifier (FAILS: missing val dir)
    └── test_model_accuracy.py          # CLI benchmark evaluating all discovered .pt models
```

---

## 3. Tech Stack

### Python Dependencies (requirements.txt)
*Exact packages specified in [requirements.txt](file:///d:/Projects/Retail_Shelf_intelligence-v2-/requirements.txt):*
* `ultralytics>=8.2.0`
* `torch>=2.0.0`
* `torchvision>=0.15.0`
* `torchaudio>=2.0.0`
* `opencv-python>=4.8.0`
* `fastapi>=0.110.0`
* `uvicorn>=0.29.0`
* `Pillow>=10.0.0`
* `numpy>=1.24.0`
* `pandas>=2.0.0`
* `PyYAML>=6.0`
* `requests>=2.31.0`
* `python-multipart>=0.0.9`
* `easyocr>=1.7.0`
* `paddlepaddle-gpu==2.6.2` *(Hardcoded exact version)*
* `nvidia-cudnn-cu11==8.9.5.29` *(Hardcoded exact version)*
* `paddleocr>=2.7.0`
* `rapidfuzz>=3.0.0`
* `scikit-learn>=1.3.0`
* `av>=10.0.0`
* `tqdm>=4.65.0`
* `pytest`
* `httpx`
* `wandb>=0.16.0`
* Extra pip index: `--extra-index-url https://download.pytorch.org/whl/cu118`

### Node.js Frontend Dependencies (frontend/package.json)
*Exact packages from [frontend/package.json](file:///d:/Projects/Retail_Shelf_intelligence-v2-/frontend/package.json):*
* `next`: `16.3.0`
* `react`: `19.2.8`
* `react-dom`: `19.2.8`
* `recharts`: `^3.10.1`
* `typescript`: `^5`
* `eslint`: `^9`
* `eslint-config-next`: `16.3.0`
* `@types/node`: `^20`
* `@types/react`: `^19`
* `@types/react-dom`: `^19`
* *Note*: Tailwind CSS is **NOT** listed in `package.json` and is **NOT** installed.

---

## 4. How the System Actually Works, Step by Step

*Tracing the real execution path in code:*

```
[User Browser]
      │
      │ 1. Multipart Form Upload (file, confidence, ocr_enabled)
      ▼
[FastAPI: POST /detect in api/main.py]
      │
      │ 2. Read bytes -> NumPy RGB array
      │    Save original image to data/history_images/{uuid}_original.jpg
      ▼
[ShelfDetector.detect() in src/detection/detector.py]
      │
      │ 3. YOLOv8 forward pass using models/checkpoints/best.pt
      │    Returns DetectionResult (bounding boxes, confidences, class_id)
      ▼
[ProductCounter.count() in src/detection/counter.py]
      │
      │ 4. Splits image into 4 horizontal zones (n_zones=4)
      │    Assigns products by box center Y coordinate
      │    Computes ShelfStats (zone counts, total products, average confidence)
      ▼
[AnomalyDetector.detect() in src/anomaly/rules.py]
      │
      │ 5. Rule heuristics evaluated on ShelfStats:
      │    - Empty shelf gaps (>1.5x product width or empty zone)
      │    - Fallen products (aspect ratio > 2.5x median)
      │    - Low stock (zone count <= 5)
      │    - Misplaced products (skipped for single-class "product")
      ▼
[DLAnomalyDetector.predict_batch() in src/anomaly/dl_detector.py]
      │
      │ 6. (Bypassed if models/autoencoder.pth is missing)
      │    If trained: crops products, resizes to 256x256, measures Autoencoder MSE
      │    Flags score > 0.5 as dl_anomaly
      ▼
[ProductIdentifier.identify() in src/analytics/product_identifier.py]
      │
      │ 7. (If ocr_enabled is True)
      │    Crops bounding boxes with 5% padding
      │    PaddleOCREngine (or EasyOCR fallback) extracts text strings
      │    Filters price tags and noise regexes
      │    Fuzzy matches against hardcoded brand catalog (threshold 0.55)
      │    Merges recognized brand names back onto detections
      ▼
[PlanogramChecker.check_compliance() in src/anomaly/planogram.py]
      │
      │ 8. Hardcodes a demo PLANOGRAM_VIOLATION: asserts top brand is missing 2 items
      ▼
[Image Annotation & Base64 Encoding in api/main.py]
      │
      │ 9. Draws green boxes for products (img_processed)
      │    Draws red boxes and alert text for anomalies (img_anomaly)
      │    Saves to data/history_images/ and encodes to Base64 strings
      ▼
[db.log_detection() in src/database/db.py]
      │
      │ 10. Persists run in SQLite tables (detections, inventory, anomalies)
      ▼
[JSON Response returned to Frontend]
      │
      │ 11. Dashboard renders MetricsBar, ProductChart, OCRResultsTable, and previews
```

---

## 5. Each Module Explained

| File Path | Purpose | Inputs | Outputs | Who Calls It | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `config.py` | Central configuration constants and hardware device detection | Environment variables, filesystem checks | Configuration variables | Imported by almost all modules | **WORKING** |
| `api/main.py` | REST API service exposing inference, analytics, and history | HTTP requests, image files, JSON payloads | JSON detection responses, Base64 images | Frontend dashboard, external clients | **WORKING** |
| `src/database/db.py` | SQLite database schema creation, logging, and history queries | Dictionaries of detection results, record IDs | SQLite rows, history records, insert IDs | `api/main.py` | **WORKING** |
| `src/detection/detector.py` | Wrapper around Ultralytics YOLOv8 inference | RGB NumPy image or file path, confidence threshold | `DetectionResult` object with bounding box list | `api/main.py`, `active_learning.py`, tests | **WORKING** |
| `src/detection/counter.py` | Divides shelf into horizontal zones and calculates product counts | `DetectionResult` | `ShelfStats` object with zone breakdown | `api/main.py`, tests | **WORKING** |
| `src/detection/train.py` | Orchestrates Phase 1 baseline YOLOv8 training | Dataset YAML, pretrained YOLO weights | Model checkpoints in `models/checkpoints/phase1/` | Invoked directly via CLI | **WORKING** |
| `src/anomaly/rules.py` | Heuristic checks for empty gaps, low stock, fallen, misplaced items | `ShelfStats` | List of `Anomaly` objects | `api/main.py`, tests | **WORKING** |
| `src/anomaly/autoencoder.py` | PyTorch 4-layer Convolutional Autoencoder architecture (256x256) | Image tensor `(B, 3, 256, 256)` | Reconstructed image tensor `(B, 3, 256, 256)` | `src/anomaly/dl_detector.py` | **WORKING** |
| `src/anomaly/dl_detector.py` | Trains Autoencoder and computes reconstruction MSE anomaly score | List of image crop arrays | MSE anomaly scores array, anomaly dictionaries | `api/main.py` | **PARTIAL** *(Autoencoder model file `autoencoder.pth` is missing from disk)* |
| `src/anomaly/planogram.py` | Synthesizes planogram violations based on identified inventory | `ProductInventory` object | List of `Anomaly` objects (`PLANOGRAM_VIOLATION`) | `api/main.py` | **WORKING** *(Generates synthetic demo violation)* |
| `src/anomaly/feature_extractor.py` | Extracts 1280-dim feature vector using EfficientNet-B0 | Image tensor `(B, 3, 256, 256)` | Feature tensor `(B, 1280)` | None | **NOT INTEGRATED** *(Never imported or called)* |
| `src/analytics/paddle_ocr.py` | Process pool worker for PaddleOCR with fallback to EasyOCR | Image crop NumPy array | List of `(bbox, text, conf)` tuples | `product_identifier.py`, `ocr.py` | **WORKING** |
| `src/analytics/ocr.py` | Extracts text and parses currency prices via regex | RGB NumPy array, detection boxes | List of `OCRResult` objects, price dictionaries | `api/main.py` (`/analytics/ocr` endpoint) | **WORKING** |
| `src/analytics/product_identifier.py` | Preprocesses crops, runs OCR, matches against brand catalog | RGB NumPy image, detection bounding box list | `ProductInventory` object with identified items | `api/main.py` | **WORKING** |
| `src/analytics/shelf_share.py` | Calculates pixel occupancy rate and brand space percentages | Detection dictionaries, image dimensions | `ShelfShareResult` object | `api/main.py` (`/analytics/shelf-share`) | **WORKING** |
| `src/analytics/heatmap.py` | Generates 2D Gaussian product density heatmaps and gap maps | Detection dictionaries, image dimensions | Float32 NumPy heatmap array (0.0 to 1.0) | `api/main.py` (`/analytics/heatmap`) | **WORKING** |
| `src/continual_learning/replay_buffer.py` | Reservoir sampling buffer storing past images and labels | Image paths, label paths, phase numbers | Exported training sample directories | `trainer.py`, `api/main.py`, tests | **WORKING** |
| `src/continual_learning/trainer.py` | Incremental fine-tuning mixing new data with buffer and layer freezing | New image/label folders, class list, phase number | Updated YOLO weights in `phase{N}/` and `best.pt` | `scripts/run_continual_training.py`, `api/main.py` | **WORKING** |
| `src/continual_learning/ewc.py` | Computes Fisher Information matrix for weight penalties | PyTorch model, DataLoader, lambda weight | EWC scalar loss penalty tensor | None | **NOT INTEGRATED** *(Never called by `trainer.py`)* |
| `src/continual_learning/active_learning.py` | Queries unlabeled images with high detector uncertainty | Image directory path, `ShelfDetector` | Top-k `UncertaintySample` objects | CLI runner, `api/main.py` | **WORKING** |
| `src/utils/image_utils.py` | Helper functions for loading, resizing, letterbox padding, saving | File paths, NumPy arrays, PIL images | Processed NumPy arrays or saved files | `src/detection/detector.py`, tests | **WORKING** |
| `src/utils/prepare_dataset.py` | Parses SKU-110K CSV files and writes YOLO formatted text files | Raw CSV annotation files and raw images | Normalized YOLO txt files in `data/processed/` | Invoked directly via CLI | **WORKING** |
| `src/utils/visualizer.py` | Helper functions to draw detection boxes and anomaly overlays | Images, bounding box coordinates, anomaly dicts | Annotated image arrays | `detector.detect_and_draw()`, tests | **PARTIAL** *(`draw_anomaly_zones` is never called)* |
| `scripts/run_continual_training.py` | CLI orchestrator discovering untrained `phaseN` folders | CLI arguments (`--phase`, `--all`) | Executes `IncrementalTrainer` per phase | User CLI command | **WORKING** |
| `tests/test_detector.py` | Pytest suite for detector, counter, buffer, and image utils | Synthetic image fixtures, mock results | 29 passing test assertions | `pytest` | **WORKING** *(Anomaly tests contain dummy `assert True`)* |
| `tests/test_api.py` | Standalone script testing POST `/detect` with val_14.jpg | File `data/processed/images/val/val_14.jpg` | Prints status and detected inventory | Standalone python script | **PARTIAL** *(Fails when val_14.jpg is missing)* |
| `tests/test_identifier.py` | Standalone script testing `ProductIdentifier` on 3 images | Folder `data/processed/images/val` | Prints OCR identification time and brands | Standalone python script | **PARTIAL** *(Fails when val folder is missing)* |
| `tests/evaluate.py` | Evaluates checkpoint mAP and measures accuracy drop | Checkpoint `.pt` paths, dataset YAML | Prints mAP50, mAP50-95, forgetting score | Standalone python script | **PARTIAL** *(Fails if dataset path in YAML is missing)* |
| `tests/test_model_accuracy.py` | Finds all `.pt` checkpoints and evaluates validation metrics | Discovered checkpoint paths, dataset YAML | Prints formatted ASCII table of mAP scores | Standalone python script | **PARTIAL** *(Fails if dataset path in YAML is missing)* |
| `frontend/src/app/page.tsx` | Main Next.js dashboard UI | User file uploads, webcam frames | Telemetry cards, charts, overlay visuals | User browser navigation | **WORKING** |
| `frontend/src/app/history/page.tsx` | Next.js history page displaying past runs from SQLite | None (calls `/api/history` via fetch) | Rendered list of past runs, delete actions | User browser navigation | **WORKING** |
| `frontend/src/app/settings/page.tsx` | Next.js settings page with confidence and OCR controls | User slider and checkbox interactions | Updates React context values | User browser navigation | **WORKING** |

---

## 6. Anomaly Detection

### Real Anomaly Types in rules.py
The rule-based detector `(src/anomaly/rules.py:L31-L37)` defines five anomaly types in `AnomalyType`:
1. `EMPTY_SHELF`:
   - A shelf zone contains zero detections `(rules.py:L95)`.
   - OR two adjacent products on the same shelf have a horizontal gap exceeding `max(1.5 * avg_product_width, 50px)` `(rules.py:L152)`.
2. `FALLEN_PRODUCT`:
   - A product has a width-to-height aspect ratio greater than `2.5 * median_shelf_aspect_ratio` and greater than `1.2` `(rules.py:L137)`.
3. `LOW_STOCK`:
   - A zone has product count greater than `empty_threshold` (default 2) and less than or equal to `low_stock_threshold` (default 5) `(rules.py:L170)`.
4. `MISPLACED`:
   - A product is located farther than `60% of image width` from its class cluster centroid `(rules.py:L201)`.
   - *Active constraint*: If the model detects only one class (such as "product"), this check automatically aborts and returns empty list `(rules.py:L198)`.
5. `PLANOGRAM_VIOLATION`:
   - Defined in the enum, but generated by `src/anomaly/planogram.py:L34`. It inspects recognized OCR products and synthesizes a demonstration alert asserting the top product is missing 2 facings `(planogram.py:L29-L38)`.

*Note on Missing Price Tags*: There is **no code** in `rules.py` checking for price tags or missing price labels.

### Real Autoencoder Design (src/anomaly/autoencoder.py & dl_detector.py)
* **Input Resolution**: `256x256` RGB image crops `(ANOMALY_ROI_SIZE = 256 in config.py:L102)`.
* **Architecture (`LightweightAutoencoder`)**:
  * **Encoder**: 4 ConvBlocks:
    * Conv2d(k=3, s=2, p=1) -> BatchNorm2d -> ReLU: 3 -> 32 `(128x128)`
    * Conv2d(k=3, s=2, p=1) -> BatchNorm2d -> ReLU: 32 -> 64 `(64x64)`
    * Conv2d(k=3, s=2, p=1) -> BatchNorm2d -> ReLU: 64 -> 128 `(32x32)`
    * Conv2d(k=3, s=2, p=1) -> BatchNorm2d -> ReLU: 128 -> 128 `(16x16 Bottleneck)`
  * **Decoder**: 4 ConvTransposeBlocks:
    * ConvTranspose2d(k=3, s=2, p=1, out_pad=1) -> BatchNorm2d -> ReLU: 128 -> 128 `(32x32)`
    * ConvTranspose2d(k=3, s=2, p=1, out_pad=1) -> BatchNorm2d -> ReLU: 128 -> 64 `(64x64)`
    * ConvTranspose2d(k=3, s=2, p=1, out_pad=1) -> BatchNorm2d -> ReLU: 64 -> 32 `(128x128)`
    * ConvTranspose2d(k=3, s=2, p=1, out_pad=1) -> Sigmoid: 32 -> 3 `(256x256)`
* **Anomaly Score Calculation**: Mean Squared Error (MSE) computed per pixel and averaged across channels, height, and width `(autoencoder.py:L83-L85)`.
* **Threshold**: `ANOMALY_THRESHOLD = 0.5` in `config.py:L104`.
* **Model Checkpoint Path**: `models/autoencoder.pth` `(dl_detector.py:L80)`.
* *Disk Status*: `models/autoencoder.pth` does **not** exist in the repository. The detector initializes with `_is_trained = False` and skips deep learning inference until trained via `POST /anomaly/dl/train`.

### How Both Are Combined in the API (api/main.py:L192-234, L261-264)
1. `_run_detection()` runs `AnomalyDetector.detect(stats)` to gather heuristic spatial anomalies.
2. If `dl_detector.is_trained` is True, crops each detected product bounding box with 10% padding, evaluates MSE reconstruction scores, and appends any score > 0.5 as a `dl_anomaly`.
3. If OCR is enabled, runs `PlanogramChecker.check_compliance(inventory)` and appends planogram anomalies.
4. All anomaly dictionaries are merged into a single `anomalies` list returned in the JSON response and drawn on `img_anomaly`.

---

## 7. OCR and Product Identification

### Engine and Execution (src/analytics/paddle_ocr.py)
* **Primary Engine**: PaddleOCR using DB (Differentiable Binarization) text detection and SVTR_LCNet text recognition `(paddle_ocr.py:L8-L10)`.
* **Process Isolation**: Runs inside a single-worker `ProcessPoolExecutor` `(paddle_ocr.py:L166)` to prevent C++ runtime and Pybind11 symbol conflicts with PyTorch on Windows `(paddle_ocr.py:L132)`.
* **Windows cuDNN Patch**: `_patch_paddle_dlls_on_windows()` automatically copies `cublas64_11.dll` and `cudnn64_8.dll` from the virtual environment into Paddle's `libs` directory `(paddle_ocr.py:L31-L61)`.
* **Fallback**: If PaddleOCR fails to initialize or crashes, it automatically falls back to PyTorch-native `EasyOCR` (`easyocr.Reader(['en'], gpu=self._use_gpu)`) `(paddle_ocr.py:L189-L208)`.

### Crop Preprocessing (src/analytics/product_identifier.py:L245-L273)
Each detected bounding box crop passes through:
1. Short side upscaling to at least 256 pixels using cubic interpolation if smaller.
2. Mild unsharp masking via Gaussian blur (`sigmaX=1.5`, weighted blending 1.5x original - 0.5x blurred).

### Catalog Matching Mechanism (src/analytics/product_identifier.py:L168-L233)
* **Text Noise Filters**: Excludes text strings matching regexes for price tags (`$3.99`, bare decimals), shelf codes (`5G8`, `COS 4A`), pure numbers, and 1-2 character fragments `(product_identifier.py:L113-L137)`.
* **Brand Catalog**: An in-memory Python dictionary containing 33 product brands and alias lists:
  * Drinks: Coca-Cola, Fanta, Sprite, Pepsi, Dr Pepper, Minute Maid, Tim Hortons.
  * Pharmacy & Personal Care: Profissimo, Pure, Calggy, Advil, Balea, Colgate, Dial, Equate, Irish Spring, Ivory, Ibuprofen, Olay, Raid, Tylenol, Fixodent, Crest, Sensodyne, Dove, Nivea, Gillette.
  * Deodorants: Old Spice, Degree, Speed Stick, Secret, Axe, Suave, Right Guard, Mitchum, Brut, Arrid, Dry Idea, Ban, Sure.
* **Matching Logic**:
  1. Checks if any keyword alias is a direct substring of the lowercase OCR text.
  2. If no exact substring matches, calculates similarity ratio using `rapidfuzz.fuzz.ratio` (or `difflib.SequenceMatcher.ratio` fallback).
  3. If similarity ratio >= `0.55` (55%), assigns the brand name `(product_identifier.py:L230)`.
  4. If unmatched, labels the product as `"Unknown"`.

---

## 8. Continual Learning

### What trainer.py Really Does (src/continual_learning/trainer.py)
* **Buffer Sampling**: Pulls up to `REPLAY_SAMPLE_SIZE` (default 2000, `config.py:L133`) historical image and label files from `ReplayBuffer` `(trainer.py:L223)`.
* **Label Conversion**: If `new_labels_dir` contains CSV files, converts them on-the-fly to normalized YOLO bounding box text files in a temporary directory `(trainer.py:L134-L204)`.
* **Data Mixing**: Copies new images/labels and replay samples into a temporary dataset directory with a generated `cl_dataset.yaml` `(trainer.py:L206-L248)`.
* **Model Training**: Loads YOLO from `self.weights_path` (default `models/checkpoints/best.pt`).
* **Training Parameters**:
  * `freeze = 10`: Freezes the first 10 backbone layers to preserve generic visual features `(trainer.py:L271)`.
  * `lr0 = 0.001`, `lrf = 0.01`: Lower learning rates to prevent drastic parameter disruption `(trainer.py:L269-L270)`.
  * `epochs = cfg.CL_EPOCHS`: Default 100 epochs `(trainer.py:L117)`.
  * `batch = cfg.BATCH_SIZE`: Default 4 `(trainer.py:L260)`.
  * `imgsz = cfg.IMG_SIZE`: Default 640 `(trainer.py:L261)`.
  * Augmentations: `mosaic = 1.0`, `fliplr = 0.5`, `degrees = 10.0`, `scale = 0.5` `(trainer.py:L275-L280)`.
* **Weight Saving**: Copies `finetune/weights/best.pt` to `phase{phase}/best{phase}.pt` and overwrites `models/checkpoints/best.pt` `(trainer.py:L299-L302)`.
* **Buffer Ingestion**: Adds new training samples into `ReplayBuffer` `(trainer.py:L314-L318)`.

### Elastic Weight Consolidation (ewc.py) Status
* **`ewc.py` is NOT used.**
* `src/continual_learning/trainer.py` does not import or call `ewc.py`.
* `scripts/run_continual_training.py` does not use `ewc.py`.
* In `api/main.py:L690`, the Pydantic schema accepts `use_ewc: bool = False`, but the background training worker **never passes `use_ewc`** to `trainer.train_new_phase()`.

---

## 9. Active Learning

### What the Code Really Does (src/continual_learning/active_learning.py)
* **Class**: `ActiveLearner`.
* **Input Pool**: Reads up to `pool_size` (default 200, `config.py:L145`) image paths from a target directory.
* **Inference**: Executes `detector.detect(img_path)` for each image and gathers bounding box confidences.
* **Scoring Strategies (`score_uncertainty`)**:
  * `entropy`: Computes Shannon entropy on confidence probabilities normalized to sum to 1 `(active_learning.py:L83-L88)`.
  * `margin`: Computes `1.0 - (top1_conf - top2_conf)` `(active_learning.py:L95-L96)`.
  * `least_confident`: Computes `1.0 - max(confidences)` `(active_learning.py:L100)`.
  * `detection_count`: If count < 5, returns 1.0; if count > 300, returns 0.8; otherwise `1.0 - mean(confidences)` `(active_learning.py:L105-L110)`.
* **Selection**: Sorts images descending by uncertainty score and returns the top `query_size` (default 20, `config.py:L146`) samples as `UncertaintySample` objects.
* **Invocation**: Can be run via CLI (`python -m src.continual_learning.active_learning --images_dir <path>`) or via API (`POST /active-learning/query`).

---

## 10. Real API Endpoints

*All endpoints from [api/main.py](file:///d:/Projects/Retail_Shelf_intelligence-v2-/api/main.py):*

| Path | Method | Request Parameters / Body | Response Fields | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `/health` | `GET` | None | `status`, `model`, `weights`, `device`, `detection_mode`, `ocr_backend`, `paddle_gpu`, `paddle_precision`, `history_size`, `dl_anomaly_trained` | System sanity and hardware status check |
| `/detect` | `POST` | `multipart/form-data`: `file` (UploadFile, required), `confidence` (float, default 0.25), `ocr_enabled` (bool, default True), `detect_anomalies` (bool, default True) | `image_width`, `image_height`, `total_products`, `counts_by_class`, `avg_confidence`, `zones`, `anomalies`, `detections`, `processing_time_ms`, `hardware_device`, `product_inventory`, `image_b64`, `image_anomaly_b64`, `original_image_path`, `processed_image_path`, `anomaly_image_path`, `detection_id` | Core shelf detection and analysis pipeline |
| `/detect/batch` | `POST` | `multipart/form-data`: `files` (List[UploadFile], required) | `results` (list of detection responses), `total_images` | Sequential batch detection |
| `/detect/async` | `POST` | `multipart/form-data`: `file` (UploadFile, required) | `job_id`, `status`, `message` | Submits detection to background thread pool |
| `/detect/async/{job_id}` | `GET` | Path param: `job_id` (str) | `status`, `result`, `filename` | Polls result of an async detection job |
| `/history` | `GET` | Query params: `limit` (int, default 50), `offset` (int, default 0) | `history` (list of records with detections, anomalies, inventory), `total` | Retrieves past detection runs from SQLite |
| `/history` | `DELETE` | None | `status`, `message` | Clears all records from SQLite |
| `/history/{record_id}` | `DELETE` | Path param: `record_id` (int) | `status`, `message` | Deletes a single detection record and children |
| `/analytics/heatmap` | `POST` | `multipart/form-data`: `file` (UploadFile, required) | `heatmap` (2D float array), `image_width`, `image_height`, `total_products` | Generates 2D Gaussian density heatmap array |
| `/analytics/shelf-share` | `POST` | `multipart/form-data`: `file` (UploadFile, required) | `occupancy_rate`, `occupied_area`, `empty_area`, `share_by_class`, `count_by_class` | Calculates pixel occupancy and brand share |
| `/analytics/ocr` | `POST` | `multipart/form-data`: `file` (UploadFile, required) | `texts`, `prices`, `total_texts`, `total_prices` | Extracts raw OCR texts and regex prices |
| `/analytics/identify-products` | `POST` | `multipart/form-data`: `file` (UploadFile, required) | `inventory`, `total_detections` | Extracts recognized brand inventory |
| `/anomaly/dl/train` | `POST` | Query params: `epochs` (int, default 50), `lr` (float, default 0.001) | `status`, `observations`, `message` | Trains Autoencoder on crops in `data/train_normal` |
| `/active-learning/query` | `POST` | JSON: `images_dir` (str, required), `method` (str, optional), `query_size` (int, optional) | `samples`, `total_queried`, `method` | Scores uncertain images in a target folder |
| `/buffer/stats` | `GET` | None | `size`, `max_size`, `total_seen`, `phases`, `classes` | Replay buffer storage statistics |
| `/buffer/seed` | `POST` | Query params: `phase` (int, default 1), `max_samples` (int, default 100) | `status`, `buffer_size` | Seeds replay buffer from processed data folder |
| `/continual/train` | `POST` | JSON: `new_images_dir` (str, required), `new_labels_dir` (str, required), `class_names` (List[str], required), `phase` (int, default 2), `epochs` (int, optional), `use_ewc` (bool, default False - unused) | `status`, `phase`, `use_ewc`, `message` | Launches incremental fine-tuning in background |

### Next.js Proxy Rewrites (frontend/next.config.ts)
* `source: "/api/:path*"` -> `destination: "http://localhost:8000/:path*"`: Proxies requests beginning with `/api/` on port 3000 to port 8000 without the `/api/` prefix.
* `source: "/history-images/:path*"` -> `destination: "http://localhost:8000/history-images/:path*"`: Proxies static history image requests.
* *Direct Call Note*: If a client calls FastAPI directly on port 8000, endpoints like `/api/health` or `/api/detect` do **not** exist and return `404 Not Found`. FastAPI only registers `/health` and `/detect`.

---

## 11. Database

*SQLite database file: `data/retail_intelligence.db` (src/database/db.py:L11-L65):*

### Table: `detections`
* `id`: `INTEGER PRIMARY KEY AUTOINCREMENT`
* `timestamp`: `DATETIME DEFAULT CURRENT_TIMESTAMP`
* `total_products`: `INTEGER`
* `avg_confidence`: `REAL`
* `processing_time_ms`: `REAL`
* `original_image_path`: `TEXT`
* `processed_image_path`: `TEXT`
* `total_identified`: `INTEGER DEFAULT 0`

### Table: `inventory`
* `id`: `INTEGER PRIMARY KEY AUTOINCREMENT`
* `detection_id`: `INTEGER` *(FOREIGN KEY references detections(id))*
* `product_name`: `TEXT`
* `count`: `INTEGER`

### Table: `anomalies`
* `id`: `INTEGER PRIMARY KEY AUTOINCREMENT`
* `detection_id`: `INTEGER` *(FOREIGN KEY references detections(id))*
* `type`: `TEXT`
* `severity`: `TEXT`
* `description`: `TEXT`
* `zone_id`: `TEXT`

---

## 12. Frontend

### Pages
* **`frontend/src/app/page.tsx` (Dashboard)**:
  * Single image upload (drag-and-drop or file picker).
  * Batch upload (sequential processing with progress indicator).
  * Live camera feed (`navigator.mediaDevices.getUserMedia`).
  * Preview tabs: Original Input, AI Detection Output (green boxes), Anomaly Overlay (red boxes).
  * Hardware badge: Displays GPU/CPU status fetched from `/health`.
* **`frontend/src/app/history/page.tsx` (History Viewer)**:
  * Fetches historical detection runs from `/api/history`.
  * Expandable accordion cards displaying run timestamp, thumbnail, total products, identification rate, and anomaly breakdowns.
  * Delete actions (delete single record, clear all history).
* **`frontend/src/app/settings/page.tsx` (Settings)**:
  * Slider for Detection Confidence Threshold (range 0.1 to 0.95, step 0.05).
  * Checkbox toggle for "Enable OCR Brand Recognition".
  * State persisted in React Context (`settings-context.tsx`).

### Components
* `MetricsBar`: Header metric cards showing Total Products, Identified, OOS Gaps, and Planogram Anomalies.
* `ProductChart`: Horizontal bar chart rendered with Recharts showing count per brand.
* `OCRResultsTable`: Scrollable table showing recognized product name, OCR confidence, and bounding box coordinates.
* `AnomalyPanel` / `AnomalyListPanel`: Cards rendering detected anomaly descriptions with color-coded severity pills.

### API Calling Method & Styling
* **API Calls**:
  * `frontend/src/app/page.tsx:L72, L173` hardcodes absolute URLs: `fetch("http://localhost:8000/health")` and `fetch("http://localhost:8000/detect", ...)`.
  * `frontend/src/app/history/page.tsx:L31` uses relative proxy URL: `fetch("/api/history")`.
* **Styling Method**: Pure CSS Modules (`page.module.css`, `Layout.module.css`, `AnomalyPanel.module.css`, `MetricsBar.module.css`) and global CSS variables (`globals.css`). No Tailwind CSS.

---

## 13. Configuration Reference

*Exact parameters and defaults from [config.py](file:///d:/Projects/Retail_Shelf_intelligence-v2-/config.py):*

| Variable | Actual Default Value | Meaning / Purpose |
| :--- | :--- | :--- |
| `ROOT_DIR` | Absolute path to project root | Base directory anchor |
| `DATA_DIR` | `.../data` | Root data folder |
| `RAW_DIR` | `.../data/raw` | Raw SKU-110K images and annotations |
| `PROCESSED_DIR` | `.../data/processed` | Prepared YOLO training splits |
| `REPLAY_BUFFER_DIR` | `.../data/replay_buffer` | Experience replay buffer directory |
| `DB_PATH` | `.../data/retail_intelligence.db` | SQLite database file location |
| `MODELS_DIR` | `.../models` | Base directory for models and configs |
| `CHECKPOINTS_DIR` | `.../models/checkpoints` | Saved weight checkpoints directory |
| `CONFIGS_DIR` | `.../models/configs` | Model and training YAML configs |
| `DATASET_YAML` | `.../models/configs/data_kaggle.yaml` | Active dataset YAML path |
| `SUBSET_SIZE` | `None` | Max images for fast debugging (None = all) |
| `TRAIN_RATIO` | `0.80` | Train split proportion (80% train, 20% val) |
| `DEVICE` | Auto-detected (`cuda`, `mps`, or `cpu`) | Compute device (checks `nvidia-smi` first) |
| `MODEL_NAME` | `"yolov8m.pt"` | Base pretrained YOLO model weights |
| `BEST_WEIGHTS` | `.../models/checkpoints/best.pt` | Active best fine-tuned weights |
| `DETECTION_MODE` | `"detect"` | Mode: `"detect"` or `"segment"` |
| `CLASS_NAMES` | `["product"]` | List of target detection classes |
| `EPOCHS` | `100` | Training epochs for Phase 1 base training |
| `BATCH_SIZE` | `4` | Training and inference batch size |
| `IMG_SIZE` | `640` | Input image resolution for YOLO |
| `WORKERS` | `4` | DataLoader subprocess worker threads |
| `CONFIDENCE_THRESHOLD` | `0.25` | Default detection confidence cutoff |
| `IOU_THRESHOLD` | `0.45` | Non-Maximum Suppression IoU threshold |
| `EMPTY_SHELF_MAX_PRODUCTS`| `2` | Max products before a zone is empty |
| `LOW_STOCK_MAX_PRODUCTS` | `5` | Max products before a zone is low stock |
| `MISPLACED_IOU_THRESHOLD` | `0.1` | Threshold for misplaced clustering |
| `ANOMALY_ROI_SIZE` | `256` | Crop input resolution for Autoencoder |
| `ANOMALY_BATCH_SIZE` | `16` | Autoencoder batch inference size |
| `ANOMALY_THRESHOLD` | `0.5` | MSE cutoff for Autoencoder anomaly flag |
| `ANOMALY_PRECISION` | `"fp16"` | Mixed precision for Autoencoder inference |
| `ANOMALY_DEVICE` | Follows `DEVICE` | Device used for Autoencoder inference |
| `OCR_ENABLED` | `True` | Global master toggle for OCR processing |
| `OCR_BACKEND` | `"paddleocr"` | OCR engine backend name |
| `OCR_LANGUAGES` | `["en"]` | Language list for OCR |
| `OCR_CONFIDENCE` | `0.15` | Minimum OCR text acceptance score |
| `MAX_OCR_PRODUCTS` | `900` | Max product crops processed per image |
| `PADDLE_USE_GPU` | `True` | GPU flag for PaddleOCR |
| `PADDLE_USE_ANGLE_CLS` | `False` | Angle classification flag for PaddleOCR |
| `PADDLE_LANG` | `"en"` | Language code for PaddleOCR |
| `PADDLE_PRECISION` | `"fp16"` | Mixed precision flag for PaddleOCR |
| `PADDLE_MAX_BATCH_SIZE` | `16` | Max batch size for PaddleOCR forward passes |
| `PADDLE_DET_DB_BOX_SCORE`| `0.5` | DB text box detection score threshold |
| `PADDLE_REC_IMAGE_SHAPE` | `"3, 48, 320"` | Input tensor shape for text recognition |
| `HEATMAP_RADIUS` | `40` | Gaussian blur kernel radius for heatmaps |
| `HEATMAP_INTENSITY` | `0.6` | Overlay opacity for heatmap blending |
| `CAMERA_INTERVAL_SEC` | `5` | Capture interval for webcam auto-mode |
| `REPLAY_BUFFER_MAX_SIZE` | `4000` | Max sample storage for replay buffer |
| `REPLAY_SAMPLE_SIZE` | `2000` | Samples mixed per continual training phase |
| `CL_EPOCHS` | `100` | Epochs per incremental continual learning phase |
| `EWC_LAMBDA` | `10000` | Importance weight for EWC loss penalty |
| `EWC_N_SAMPLES` | `200` | Samples used for Fisher matrix estimation |
| `WANDB_PROJECT` | `"retail-shelf-intelligence"` | Project name for Weights & Biases logging |
| `AL_UNCERTAINTY_METHOD` | `"entropy"` | Default active learning scoring method |
| `AL_POOL_SIZE` | `200` | Max images scored in active learning pool |
| `AL_QUERY_SIZE` | `20` | Images selected per active learning round |
| `MAX_QUEUE_SIZE` | `50` | Max queue depth for async detection jobs |
| `ASYNC_WORKERS` | `2` | Background thread pool size for async jobs |
| `API_HOST` | `"0.0.0.0"` | FastAPI bind host |
| `API_PORT` | `8000` | FastAPI bind port |
| `DASHBOARD_TITLE` | `"Retail Shelf Intelligence"` | UI title string |

---

## 14. How to Set Up and Run

### Step 1: Python Environment
```powershell
# Windows
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```
*(Note: If `paddlepaddle-gpu==2.6.2` fails to compile on non-CUDA systems or Python 3.12+, remove it from `requirements.txt`; the system will automatically fall back to `EasyOCR`)*.

### Step 2: Dataset Preparation (SKU-110K)
The raw dataset is not included in the repository. To prepare data:
1. Place raw SKU-110K images in `data/raw/images/train/` and `data/raw/images/val/`.
2. Place annotations in `data/raw/annotations/annotations_train.csv` and `annotations_val.csv`.
3. Run the preparation script:
```powershell
python -m src.utils.prepare_dataset
```
This converts CSV labels into YOLO format and writes `data/processed/`.

### Step 3: Start the Backend API
```powershell
.\venv\Scripts\activate
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
API runs on `http://localhost:8000`.

### Step 4: Start the Frontend Dashboard
```powershell
cd frontend
npm install
npm run dev
```
Dashboard runs on `http://localhost:3000`.

### Step 5: Base Training (Phase 1)
```powershell
python -m src.detection.train
```

### Step 6: Continual Learning (Phase 2+)
Place new data in `data/phase2/images/` and `data/phase2/labels/`:
```powershell
# Train Phase 2 explicitly:
python scripts/run_continual_training.py --phase 2

# Train all discovered phase folders sequentially:
python scripts/run_continual_training.py --all
```

---

## 15. Models and Checkpoints on Disk

### Files and Sizes on Disk
* **Pretrained Weights in Root**:
  * `yolo26n.pt`: 5,544,453 bytes (5.29 MB)
  * `yolov8n.pt`: 6,534,387 bytes (6.23 MB)
  * `yolov8s.pt`: 22,588,772 bytes (21.54 MB)
  * `yolov8m.pt`: 52,136,884 bytes (49.72 MB)
  * `yolov8l.pt`: 87,792,836 bytes (83.73 MB)
* **Checkpoints in `models/checkpoints/`**:
  * `best.pt`: 155,497,956 bytes (148.29 MB, SHA256: `fb26d22fab08bf1e...`)
  * `best.pt.backup`: 155,497,956 bytes (148.29 MB, SHA256: `fb26d22fab08bf1e...`)
  * `last.pt`: 136 bytes *(Git LFS text pointer file, not valid weights)*
  * `phase1/best1.pt`: 155,497,956 bytes (148.29 MB, SHA256: `fb26d22fab08bf1e...`)
  * `phase1/last1.pt`: 155,499,364 bytes (148.29 MB)
  * `phase1/train/weights/best.pt`: 155,497,956 bytes (148.29 MB, SHA256: `fb26d22fab08bf1e...`)
  * `phase1/train/weights/last.pt`: 155,499,364 bytes
  * `phase2/finetune/weights/best.pt`: 136,148,348 bytes (129.84 MB, SHA256: `6d52a54a180a91dc...`)
  * `phase2/finetune/weights/last.pt`: 136,148,348 bytes (129.84 MB)
  * `phase2/best2.pt`: **MISSING ON DISK**. Does not exist directly in `models/checkpoints/phase2/`.

### Real Recorded Metrics from Disk Logs
*(Source: `models/checkpoints/phase1/train/results.csv` and `models/checkpoints/phase2/finetune/results.csv`)*

* **Phase 1 Baseline Training (`models/checkpoints/phase1/train/results.csv`)**:
  * Epochs run: **40** epochs.
  * Starting metrics (Epoch 1): `mAP50 = 0.85180`, `Precision = 0.87883`, `Recall = 0.79978`.
  * Final metrics (Epoch 40):
    * `Precision`: `0.91263` (91.3%)
    * `Recall`: `0.86712` (86.7%)
    * `mAP50`: `0.90599` (90.6%)
    * `mAP50-95`: `0.54508` (54.5%)
    * `Box Loss (val)`: `1.35446`
    * `Class Loss (val)`: `0.51526`
  * Trend: **Steadily improving**. Precision and recall steadily increased across epochs, and validation loss decreased consistently.
* **Phase 2 Incremental Fine-Tuning (`models/checkpoints/phase2/finetune/results.csv`)**:
  * Epochs run: **33** epochs.
  * Starting metrics (Epoch 1): `mAP50 = 0.09102`, `Precision = 0.22731`, `Recall = 0.16103`.
  * Final metrics (Epoch 33):
    * `Precision`: `0.45651` (45.7%)
    * `Recall`: `0.27075` (27.1%)
    * `mAP50`: `0.25361` (25.4%)
    * `mAP50-95`: `0.10043` (10.0%)
    * `Box Loss (val)`: `2.64576`
    * `Class Loss (val)`: `1.65524`
  * Trend: **Erratic and severely degraded**. After starting fine-tuning, mAP50 dropped from 0.906 to 0.091, oscillated wildly, and finished at 0.254 (a net drop of 65.2 percentage points).

---

## 16. Tests

### Test Inventory
1. `tests/test_detector.py`: Pytest suite testing `TestDetection`, `TestCounter`, `TestAnomalyDetection`, `TestReplayBuffer`, and `TestImageUtils`.
2. `tests/test_api.py`: Procedural script calling `POST /detect` using `TestClient`.
3. `tests/test_identifier.py`: Procedural script evaluating `ProductIdentifier` on 3 images.
4. `tests/evaluate.py`: Standalone CLI script evaluating mAP and forgetting score between two models.
5. `tests/test_model_accuracy.py`: Standalone CLI script discovering `.pt` models and evaluating mAP.

### What Really Runs
* Running `pytest tests/test_detector.py`:
  * **Result**: **29 passed** in 8.74 seconds.
  * **Caveat**: All 7 test functions inside `TestAnomalyDetection` (`test_no_anomaly_healthy_shelf`, `test_empty_shelf_detected`, `test_low_stock_detected`, etc.) contain only `assert True` `(test_detector.py:L146-L170)` and perform no real assertions.

### What Fails and Why
* Running `pytest tests/`:
  * **Result**: **Interrupted: 2 Collection Errors**.
  * **Error 1**: `tests/test_api.py:L10` executes at import time:
    ```python
    files={"file": ("val_14.jpg", open("data/processed/images/val/val_14.jpg", "rb"), "image/jpeg")}
    ```
    Crashes with `FileNotFoundError: 'data/processed/images/val/val_14.jpg'`.
  * **Error 2**: `tests/test_identifier.py:L17` executes at import time:
    ```python
    images = [f for f in os.listdir(VAL_DIR) if f.endswith(".jpg")][:3]
    ```
    Crashes with `FileNotFoundError: 'D:\Projects\Retail_Shelf_intelligence-v2-\data\processed\images\val'`.
* Running `tests/test_model_accuracy.py` or `tests/evaluate.py`:
  * Fails unless dataset YAML is updated because `models/configs/data_kaggle.yaml:L4` points to non-existent path `D:\retail_shelf_intelligence\data\processed`.

---

## 17. Mismatches (README vs Code)

| Topic | README Says | Code Really Does | Source File |
| :--- | :--- | :--- | :--- |
| **EWC Continual Learning** | Continual training combines Experience Replay with EWC loss penalties `(README:L32, 97, 125, 330)` | `ewc.py` is never called by `trainer.py` or `run_continual_training.py`. `use_ewc` in API request is ignored | [src/continual_learning/trainer.py](file:///d:/Projects/Retail_Shelf_intelligence-v2-/src/continual_learning/trainer.py), [api/main.py:L690](file:///d:/Projects/Retail_Shelf_intelligence-v2-/api/main.py#L690) |
| **API Endpoints** | Direct endpoints `GET /api/health`, `POST /api/detect`, `GET /api/history` `(README:L476, 489, 536)` | Endpoints are `/health`, `/detect`, `/history`. Calling `/api/...` directly on FastAPI returns 404 | [api/main.py:L402, 419, 372](file:///d:/Projects/Retail_Shelf_intelligence-v2-/api/main.py#L402) |
| **Missing API Endpoints** | `GET /api/anomalies` filters anomalies; `GET /api/analytics` aggregates counts `(README:L537, 538)` | Neither endpoint exists in `api/main.py` or `db.py` | [api/main.py](file:///d:/Projects/Retail_Shelf_intelligence-v2-/api/main.py), [src/database/db.py](file:///d:/Projects/Retail_Shelf_intelligence-v2-/src/database/db.py) |
| **Continual Train Endpoint** | `POST /train/incremental` with payload `{"phase": 2, "new_images_dir": "...", ...}` `(README:L523)` | Endpoint is `POST /continual/train` and requires `class_names: List[str]` in payload | [api/main.py:L684, 693](file:///d:/Projects/Retail_Shelf_intelligence-v2-/api/main.py#L684) |
| **Price Tag Anomaly Check** | Rule-based engine checks "Missing Price Tags" `(README:L52, 423, 437)` | `rules.py` only checks empty shelf, low stock, misplaced, and fallen items. No price tag check exists | [src/anomaly/rules.py:L31-L37](file:///d:/Projects/Retail_Shelf_intelligence-v2-/src/anomaly/rules.py#L31) |
| **Autoencoder Crop Size** | "4-layer Encoder-Decoder operating on 64x64 cropped patches" `(README:L440)` | Architecture takes `256x256` patches (`ANOMALY_ROI_SIZE = 256`) | [src/anomaly/autoencoder.py:L6](file:///d:/Projects/Retail_Shelf_intelligence-v2-/src/anomaly/autoencoder.py#L6), [config.py:L102](file:///d:/Projects/Retail_Shelf_intelligence-v2-/config.py#L102) |
| **Autoencoder Threshold** | `AUTOENCODER_THRESHOLD = 0.015` `(README:L442, 649)` | Variable is named `ANOMALY_THRESHOLD = 0.5` | [config.py:L104](file:///d:/Projects/Retail_Shelf_intelligence-v2-/config.py#L104) |
| **Autoencoder Model Path** | `models/checkpoints/autoencoder.pt` `(README:L163)` | Model path is `models/autoencoder.pth` (and file does not exist on disk) | [src/anomaly/dl_detector.py:L80](file:///d:/Projects/Retail_Shelf_intelligence-v2-/src/anomaly/dl_detector.py#L80) |
| **Replay Buffer Size Config** | `REPLAY_BUFFER_MAX_SIZE = 500` `(README:L642)` | `REPLAY_BUFFER_MAX_SIZE = 4000` | [config.py:L132](file:///d:/Projects/Retail_Shelf_intelligence-v2-/config.py#L132) |
| **Replay Sample Size Config** | `REPLAY_SAMPLE_SIZE = 50` `(README:L643)` | `REPLAY_SAMPLE_SIZE = 2000` | [config.py:L133](file:///d:/Projects/Retail_Shelf_intelligence-v2-/config.py#L133) |
| **CL Epochs Config** | `CL_EPOCHS = 10` `(README:L644)` | `CL_EPOCHS = 100` | [config.py:L134](file:///d:/Projects/Retail_Shelf_intelligence-v2-/config.py#L134) |
| **EWC Lambda Config** | `EWC_LAMBDA = 400.0` `(README:L645)` | `EWC_LAMBDA = 10000` | [config.py:L137](file:///d:/Projects/Retail_Shelf_intelligence-v2-/config.py#L137) |
| **OCR Confidence Config** | `OCR_CONFIDENCE = 0.40` `(README:L648)` | `OCR_CONFIDENCE = 0.15` | [config.py:L112](file:///d:/Projects/Retail_Shelf_intelligence-v2-/config.py#L112) |
| **Empty Shelf Gap Config** | `EMPTY_SHELF_GAP_THRESH = 150` `(README:L650)` | Setting does not exist in `config.py`. Uses `EMPTY_SHELF_MAX_PRODUCTS = 2` | [config.py:L97](file:///d:/Projects/Retail_Shelf_intelligence-v2-/config.py#L97) |
| **Frontend Styling** | Built with "Next.js 16 (App Router) and Tailwind/CSS Modules" `(README:L544)` | No Tailwind CSS installed or configured. Built purely with Vanilla CSS Modules | [frontend/package.json](file:///d:/Projects/Retail_Shelf_intelligence-v2-/frontend/package.json) |
| **Missing File: pytest.ini** | `pytest.ini` listed in tree `(README:L146)` | Does not exist | Root directory |
| **Missing File: ml_detector.py** | `src/anomaly/ml_detector.py` listed in tree `(README:L182)` | Does not exist | `src/anomaly/` |
| **Missing File: dataset.yaml** | `models/configs/dataset.yaml` listed in tree `(README:L167)` | Does not exist (only `data_kaggle.yaml` exists) | `models/configs/` |
| **Undocumented: planogram.py** | Not listed in tree | Exists and is called by `api/main.py` | [src/anomaly/planogram.py](file:///d:/Projects/Retail_Shelf_intelligence-v2-/src/anomaly/planogram.py) |
| **Undocumented: feature_extractor.py** | Not listed in tree | Exists (EfficientNet-B0) | [src/anomaly/feature_extractor.py](file:///d:/Projects/Retail_Shelf_intelligence-v2-/src/anomaly/feature_extractor.py) |

---

## 18. Known Problems and Risks

1. **Hardcoded External Path in `data_kaggle.yaml`**:
   `models/configs/data_kaggle.yaml:L4` specifies `path: D:\retail_shelf_intelligence\data\processed`. This folder does not exist on this machine, causing any YOLO evaluation (`val`, `test_model_accuracy.py`, `evaluate.py`) that relies on `cfg.DATASET_YAML` to fail unless manually changed.
2. **Broken Out-of-the-Box Test Suite (`pytest tests/`)**:
   Running `pytest tests/` fails with collection errors because `tests/test_api.py` and `tests/test_identifier.py` are written as procedural scripts rather than pytest test functions, and both open missing sample image files on import.
3. **Unused / Dead Code Modules**:
   - `src/continual_learning/ewc.py`: Completely disconnected from training loops.
   - `src/anomaly/feature_extractor.py`: Never imported or invoked.
   - `draw_anomaly_zones` in `src/utils/visualizer.py`: Never called by the API.
4. **Hardcoded API URL in Frontend**:
   `frontend/src/app/page.tsx:L72, L173` hardcodes `http://localhost:8000/health` and `http://localhost:8000/detect`. If the backend runs on a different port or host, the dashboard fails to communicate, bypassing the Next.js reverse proxy rewrites.
5. **Autoencoder Anomaly Model Missing from Disk**:
   `models/autoencoder.pth` does not exist on disk. `dl_detector.py` initializes with `_is_trained = False`, meaning visual defect detection is permanently inactive during `/detect` calls until a model is trained via `/anomaly/dl/train`.
6. **Git LFS Pointer Checkpoint**:
   `models/checkpoints/last.pt` is a 136-byte Git LFS text pointer file, not a binary PyTorch checkpoint. Attempting to load it with `torch.load()` or `YOLO()` throws an unpickling error.
7. **Synthetic Planogram Checker Logic**:
   `src/anomaly/planogram.py:L26-L38` hardcodes an artificial demonstration anomaly: whenever known products are identified, it asserts the most common product is missing 2 items to trigger an alert badge.
8. **Dependency Fragility in `requirements.txt`**:
   `paddlepaddle-gpu==2.6.2` and `nvidia-cudnn-cu11==8.9.5.29` are pinned to exact versions in `requirements.txt`. Installing on machines without CUDA 11 or on newer Python releases (3.12+) will fail during `pip install -r requirements.txt`. In addition, PaddleOCR worker process on CPU throws `Unknown argument: rec_image_shape` and falls back to EasyOCR.

---

## 19. Verification Log and System Audit

### Part A: Empirical Execution & Command Log

#### A.1 Forgetting Test Execution (`tests/evaluate.py`)
* **Command**:
  ```powershell
  .\venv\Scripts\python.exe tests/evaluate.py --before models/checkpoints/phase1/best1.pt --after models/checkpoints/phase2/best2.pt
  ```
* **Real Command Output**:
  ```
  ============================================================
  Model Evaluation
  ============================================================

  [Eval] Evaluating: models/checkpoints/phase1/best1.pt
  [Eval] Dataset:    D:\Projects\Retail_Shelf_intelligence-v2-\models\configs\data_kaggle.yaml
  Ultralytics 8.4.118  Python-3.14.4 torch-2.13.0+cpu CPU (13th Gen Intel Core i5-13420H)
  Model summary (fused): 93 layers, 25,840,339 parameters, 0 gradients, 78.7 GFLOPs

  Traceback (most recent call last):
    File "D:\Projects\Retail_Shelf_intelligence-v2-\tests\evaluate.py", line 227, in <module>
      main()
    File "D:\Projects\Retail_Shelf_intelligence-v2-\tests\evaluate.py", line 194, in main
      before_metrics = evaluate_model(args.before, args.dataset)
    File "D:\Projects\Retail_Shelf_intelligence-v2-\tests\evaluate.py", line 64, in evaluate_model
      results = model.val(
          data    = dataset_yaml,
          verbose = False,
      )
    File "D:\Projects\Retail_Shelf_intelligence-v2-\venv\Lib\site-packages\ultralytics\engine\validator.py", line 226, in __call__
      self.data = check_det_dataset(self.args.data, split=self.args.split)
    File "D:\Projects\Retail_Shelf_intelligence-v2-\venv\Lib\site-packages\ultralytics\data\utils.py", line 570, in check_det_dataset
      raise FileNotFoundError(m)
  FileNotFoundError: Dataset 'D://Projects/Retail_Shelf_intelligence-v2-/models/configs/data_kaggle.yaml' images not found, missing path 'D:\retail_shelf_intelligence\data\processed\images\val'
  ```
* **Why It Failed**:
  1. `models/configs/data_kaggle.yaml:L4` points to `D:\retail_shelf_intelligence\data\processed`, which is an external machine path not present on disk.
  2. The target `--after` weight file `models/checkpoints/phase2/best2.pt` does not exist on disk (only `models/checkpoints/phase2/finetune/weights/best.pt` exists).
* **Real Recorded Metrics Before and After (from actual training logs on disk)**:
  *(Source: `models/checkpoints/phase1/train/results.csv` vs `models/checkpoints/phase2/finetune/results.csv`)*

  | Metric | Phase 1 Before (`best1.pt`) | Phase 2 After (`phase2 best.pt`) | Net Drop / Change |
  | :--- | :--- | :--- | :--- |
  | **mAP50** | **0.90599** (90.60%) | **0.25361** (25.36%) | **-0.65238 (-65.24% drop)** |
  | **Precision** | **0.91263** (91.26%) | **0.45651** (45.65%) | **-0.45612 (-45.61% drop)** |
  | **Recall** | **0.86712** (86.71%) | **0.27075** (27.08%) | **-0.59637 (-59.64% drop)** |
  | **mAP50-95** | **0.54508** (54.51%) | **0.10043** (10.04%) | **-0.44465 (-44.47% drop)** |
  | **Validation Box Loss** | **1.35446** | **2.64576** | **+1.29130 (Worse fit)** |
  | **Validation Class Loss** | **0.51526** | **1.65524** | **+1.13998 (Worse fit)** |

---

#### A.2 Benchmark Evaluation (`tests/test_model_accuracy.py --all`)
* **Command**:
  ```powershell
  .\venv\Scripts\python.exe tests/test_model_accuracy.py --all
  ```
* **Real Command Output**:
  ```
  Found 9 models to evaluate.

  [1/9] 
  Evaluating: yolo26n.pt
  Path:       D:\Projects\Retail_Shelf_intelligence-v2-\yolo26n.pt
  Device:     cpu
  --------------------------------------------------
  Ultralytics 8.4.118  Python-3.14.4 torch-2.13.0+cpu CPU (13th Gen Intel Core i5-13420H)
  YOLO26n summary (fused): 122 layers, 2,408,932 parameters, 0 gradients, 5.5 GFLOPs
  Error evaluating yolo26n.pt: Dataset 'D://Projects/Retail_Shelf_intelligence-v2-/models/configs/data_kaggle.yaml' images not found, missing path 'D:\retail_shelf_intelligence\data\processed\images\val'

  [2/9] 
  Evaluating: yolov8l.pt
  Path:       D:\Projects\Retail_Shelf_intelligence-v2-\yolov8l.pt
  Device:     cpu
  --------------------------------------------------
  Error evaluating yolov8l.pt: Dataset '.../data_kaggle.yaml' images not found, missing path 'D:\retail_shelf_intelligence\data\processed\images\val'

  [3/9] 
  Evaluating: yolov8m.pt
  Path:       D:\Projects\Retail_Shelf_intelligence-v2-\yolov8m.pt
  Error evaluating yolov8m.pt: Dataset '.../data_kaggle.yaml' images not found, missing path 'D:\retail_shelf_intelligence\data\processed\images\val'

  [4/9] 
  Evaluating: yolov8n.pt
  Path:       D:\Projects\Retail_Shelf_intelligence-v2-\yolov8n.pt
  Error evaluating yolov8n.pt: Dataset '.../data_kaggle.yaml' images not found, missing path 'D:\retail_shelf_intelligence\data\processed\images\val'

  [5/9] 
  Evaluating: yolov8s.pt
  Path:       D:\Projects\Retail_Shelf_intelligence-v2-\yolov8s.pt
  Error evaluating yolov8s.pt: Dataset '.../data_kaggle.yaml' images not found, missing path 'D:\retail_shelf_intelligence\data\processed\images\val'

  [6/9] 
  Evaluating: best.pt
  Path:       D:\Projects\Retail_Shelf_intelligence-v2-\models\checkpoints\best.pt
  Error evaluating best.pt: Dataset '.../data_kaggle.yaml' images not found, missing path 'D:\retail_shelf_intelligence\data\processed\images\val'

  [7/9] 
  Evaluating: last.pt
  Path:       D:\Projects\Retail_Shelf_intelligence-v2-\models\checkpoints\last.pt
  Device:     cpu
  --------------------------------------------------
  Error evaluating last.pt: ERROR  D:\Projects\Retail_Shelf_intelligence-v2-\models\checkpoints\last.pt is not a loadable checkpoint  the file is empty, truncated or corrupted (UnpicklingError: invalid load key, 'v'.).

  [8/9] 
  Evaluating: phase1/best1.pt
  Path:       D:\Projects\Retail_Shelf_intelligence-v2-\models\checkpoints\phase1\best1.pt
  Error evaluating phase1/best1.pt: Dataset '.../data_kaggle.yaml' images not found, missing path 'D:\retail_shelf_intelligence\data\processed\images\val'

  [9/9] 
  Evaluating: phase1/last1.pt
  Path:       D:\Projects\Retail_Shelf_intelligence-v2-\models\checkpoints\phase1\last1.pt
  Error evaluating phase1/last1.pt: Dataset '.../data_kaggle.yaml' images not found, missing path 'D:\retail_shelf_intelligence\data\processed\images\val'

  No models were successfully evaluated.
  ```
* **Summary Table of Evaluated Checkpoints**:

  | Model File | Location | File Status | Evaluation Result | Root Cause |
  | :--- | :--- | :--- | :--- | :--- |
  | `yolo26n.pt` | Root | Valid weights (5.5 MB) | **FAILED** | Missing dataset path `D:\retail_shelf_intelligence\...` |
  | `yolov8l.pt` | Root | Valid weights (87.8 MB) | **FAILED** | Missing dataset path in `data_kaggle.yaml` |
  | `yolov8m.pt` | Root | Valid weights (52.1 MB) | **FAILED** | Missing dataset path in `data_kaggle.yaml` |
  | `yolov8n.pt` | Root | Valid weights (6.5 MB) | **FAILED** | Missing dataset path in `data_kaggle.yaml` |
  | `yolov8s.pt` | Root | Valid weights (22.6 MB) | **FAILED** | Missing dataset path in `data_kaggle.yaml` |
  | `best.pt` | `models/checkpoints/` | Valid weights (155.5 MB) | **FAILED** | Missing dataset path in `data_kaggle.yaml` |
  | `last.pt` | `models/checkpoints/` | **Corrupted (136 bytes)** | **FAILED** | Git LFS text pointer file, not a binary checkpoint |
  | `phase1/best1.pt` | `models/checkpoints/` | Valid weights (155.5 MB) | **FAILED** | Missing dataset path in `data_kaggle.yaml` |
  | `phase1/last1.pt` | `models/checkpoints/` | Valid weights (155.5 MB) | **FAILED** | Missing dataset path in `data_kaggle.yaml` |

---

#### A.3 Checkpoint Comparison and API Startup File Verification
* **Why `best.pt` and Phase 1 have the exact same numbers**:
  Computing SHA256 hashes and byte lengths on all checkpoint files proves the active production checkpoint `best.pt` was **never updated with Phase 2 weights**. It is an exact duplicate of Phase 1:

  | Checkpoint Path | File Size (Bytes) | SHA256 Hash | Relationship |
  | :--- | :--- | :--- | :--- |
  | `models/checkpoints/best.pt` | 155,497,956 | `fb26d22fab08bf1e3e60ddc0b11e9a7e6ea8e63a13a0fc545a96aa50085ba1e4` | **Identical to Phase 1** |
  | `models/checkpoints/best.pt.backup` | 155,497,956 | `fb26d22fab08bf1e3e60ddc0b11e9a7e6ea8e63a13a0fc545a96aa50085ba1e4` | Exact backup copy of Phase 1 |
  | `models/checkpoints/phase1/best1.pt` | 155,497,956 | `fb26d22fab08bf1e3e60ddc0b11e9a7e6ea8e63a13a0fc545a96aa50085ba1e4` | Original Phase 1 best checkpoint |
  | `models/checkpoints/phase1/train/weights/best.pt` | 155,497,956 | `fb26d22fab08bf1e3e60ddc0b11e9a7e6ea8e63a13a0fc545a96aa50085ba1e4` | Original Phase 1 training run best |
  | `models/checkpoints/phase2/finetune/weights/best.pt` | 136,148,348 | `6d52a54a180a91dc982753a81284d720c9ec652b7ad4f711200c3c6f44e39130` | **Distinct Phase 2 weights (never deployed)** |
  | `models/checkpoints/phase2/best2.pt` | **0 (Missing)** | *N/A* | **Does not exist on disk** |

* **Which File the API Loads at Startup**:
  1. In `config.py:L70`: `BEST_WEIGHTS = os.path.join(CHECKPOINTS_DIR, "best.pt")`.
  2. In `api/main.py:L78-L80`:
     ```python
     _weights = cfg.BEST_WEIGHTS if os.path.exists(cfg.BEST_WEIGHTS) else cfg.MODEL_NAME
     detector = ShelfDetector(weights_path=_weights, device=cfg.DEVICE)
     ```
  3. **Conclusion**: At startup, the API loads `models/checkpoints/best.pt`. Because `best.pt` has SHA256 `fb26d22f...`, the API runs **100% Phase 1 weights**. Phase 2 fine-tuning weights are completely inactive.

---

#### A.4 Live API Execution and Endpoint Test
* **Startup**: Started Uvicorn server via `uvicorn api.main:app --port 8000`.
  - Process started: PID 15504 on CPU device.
  - Server startup warning logged: `[DL Anomaly] No trained Autoencoder found. Needs training.`

* **Test Request 1: `GET /health`**:
  - **Status Code**: `200 OK`
  - **JSON Output**:
    ```json
    {
      "status": "ok",
      "model": "yolov8m.pt",
      "weights": "D:\\Projects\\Retail_Shelf_intelligence-v2-\\models\\checkpoints\\best.pt",
      "device": "cpu",
      "detection_mode": "detect",
      "ocr_backend": "paddleocr",
      "paddle_gpu": true,
      "paddle_precision": "fp16",
      "history_size": 5,
      "dl_anomaly_trained": false
    }
    ```

* **Test Request 2: `POST /detect` (Real Shelf Image, OCR Disabled)**:
  - Input: `data/history_images/e238904f-94ad-4e08-9df5-64d84f938d62_original.jpg`
  - Form parameters: `confidence=0.25`, `ocr_enabled=False`, `detect_anomalies=True`
  - **Status Code**: `200 OK`
  - **Total Products Detected**: `300` (hit YOLO default max detection limit)
  - **Average Confidence**: `0.7842`
  - **Processing Time**: `9115.28 ms` (9.12 seconds on CPU)
  - **Anomalies Returned**: **19 total**
    - `empty_shelf`: 7 anomalies (large horizontal gaps between items)
    - `fallen_product`: 12 anomalies (aspect ratio > 2.5x median)
  - **Errors / Warnings in Log**:
    `[DL Anomaly] No trained Autoencoder found. Needs training.` *(Skipped visual surface anomaly forward pass)*.

* **Test Request 3: `POST /detect` (Real Shelf Image, OCR Enabled)**:
  - Input: `data/history_images/dcc39f52-b4c6-482f-8a03-7cb3c829e24a_original.jpg`
  - Form parameters: `confidence=0.25`, `ocr_enabled=True`, `detect_anomalies=True`
  - **Status Code**: `200 OK`
  - **Errors / Warnings Logged**:
    ```
    [PaddleOCR Worker] Failed to load engine: Unknown argument: rec_image_shape
    [OCR] Falling back to PyTorch-native EasyOCR...
    ```
  - **Total Products Detected**: `300`
  - **Anomalies Returned**: **4 total** (`fallen_product`: 4)
  - **Processing Time**: Detection forward pass took `4999.68 ms`. Total HTTP request duration was `55.29 seconds` due to CPU EasyOCR scanning 300 individual product crops.

* **Test Request 4: `GET /history?limit=2`**:
  - **Status Code**: `200 OK`
  - **JSON Output**: Successfully retrieved 2 historical runs with IDs 6 and 7, timestamps, image paths, and anomaly counts inserted during the `/detect` calls.

---

#### A.5 Training Curves Analysis (`results.csv`)
* **Phase 1 Training Run (`models/checkpoints/phase1/train/results.csv`)**:
  - **Number of Epochs**: **40** epochs recorded.
  - **Initial Metrics (Epoch 1)**: `mAP50 = 0.85180`, `Precision = 0.87883`, `Recall = 0.79978`.
  - **Final Metrics (Epoch 40)**:
    - `Precision`: `0.91263` (91.3%)
    - `Recall`: `0.86712` (86.7%)
    - `mAP50`: `0.90599` (90.6%)
    - `mAP50-95`: `0.54508` (54.5%)
    - `Validation Box Loss`: `1.35446`
    - `Validation Class Loss`: `0.51526`
  - **Trend**: **Steadily improving**. Precision grew from 0.879 to 0.913, recall grew from 0.800 to 0.867, and validation losses decreased monotonically.

* **Phase 2 Fine-Tuning Run (`models/checkpoints/phase2/finetune/results.csv`)**:
  - **Number of Epochs**: **33** epochs recorded.
  - **Initial Metrics (Epoch 1)**: `mAP50 = 0.09102`, `Precision = 0.22731`, `Recall = 0.16103`.
  - **Final Metrics (Epoch 33)**:
    - `Precision`: `0.45651` (45.7%)
    - `Recall`: `0.27075` (27.1%)
    - `mAP50`: `0.25361` (25.4%)
    - `mAP50-95`: `0.10043` (10.0%)
    - `Validation Box Loss`: `2.64576`
    - `Validation Class Loss`: `1.65524`
  - **Trend**: **Erratic and severely degraded**. Fine-tuning began with an immediate collapse to 0.091 mAP50, oscillated between 0.15 and 0.26 across 33 epochs, and concluded at 0.254 mAP50. Both box loss and class loss doubled compared to Phase 1.

---

### Part B: Architectural Audit & Source Code Proofs

#### B.1 Datasets, Images, and Classes in Phase 1 vs Phase 2
* **Which Datasets Were Really Used**:
  * **Phase 1**: Only **SKU-110K** was used.
    * *Proof*: `models/checkpoints/phase1/train/args.yaml:L4` specifies `data: D:\retail_shelf_intelligence\models\configs\data_kaggle.yaml`.
    * In `models/configs/data_kaggle.yaml:L1-L3`: `names: ['product']`, `nc: 1`.
  * **Phase 2**: A temporary incremental dataset subset created from SKU-110K.
    * *Proof*: `models/checkpoints/phase2/finetune/args.yaml:L4` specifies `data: C:\Users\SRILEK~1\AppData\Local\Temp\cl_train_wf34n5kz\cl_dataset.yaml`.
  * **RPC or GroZi-120 Datasets**: **NEVER USED**. There is zero trace of RPC or GroZi-120 in code, configs, or checkpoint files.
* **Number of Classes in Each Phase**:
  * Phase 1: **1 class only** (`product`). Verified via `YOLO('models/checkpoints/phase1/best1.pt').names` -> `{0: 'product'}`.
  * Phase 2: **1 class only** (`product`). Verified via `YOLO('models/checkpoints/phase2/finetune/weights/best.pt').names` -> `{0: 'product'}`.
* **Number of Images**:
  * Standard SKU-110K contains 8,233 training images and 588 validation images.
  * In Phase 1 `args.yaml:L29`, `fraction: 1.0` (trained on all available images in `data_kaggle.yaml`).

---

#### B.2 Exact Code Path for Adding Products/Classes in Phase 2
The codebase provides two invocation entry points that converge on a single training method:

```
[CLI: scripts/run_continual_training.py]           [API: POST /continual/train in api/main.py]
         │                                                            │
         │ L64: train_phase(phase)                                    │ L700: BackgroundTask _train_job
         ▼                                                            ▼
    IncrementalTrainer.train_new_phase() in src/continual_learning/trainer.py (L113)
         │
         ├── 1. CSV to YOLO conversion (if needed):
         │      _convert_csv_labels() (trainer.py:L138-L204)
         │
         ├── 2. Build temporary dataset directory:
         │      tempfile.TemporaryDirectory(prefix="cl_train_") (trainer.py:L206)
         │      _copy_split() copies new images/labels (trainer.py:L218)
         │
         ├── 3. Export experience replay buffer:
         │      ReplayBuffer.export_for_training() (trainer.py:L223, replay_buffer.py:L142)
         │      _copy_split() mixes buffer samples into training folder (trainer.py:L226)
         │
         ├── 4. Generate dataset YAML configuration:
         │      cl_dataset.yaml with nc=len(class_names), names=class_names (trainer.py:L240-L248)
         │
         ├── 5. Ultralytics YOLOv8 Fine-Tuning:
         │      model = YOLO(self.weights_path) (trainer.py:L252)
         │      model.train(data=yaml_path, epochs=epochs, freeze=10, lr0=0.001, ...) (trainer.py:L257)
         │
         ├── 6. Checkpoint Persistence:
         │      shutil.copy2() saves best.pt to phase{phase}/best{phase}.pt and models/checkpoints/best.pt (trainer.py:L298-L302)
         │
         └── 7. Buffer Ingestion:
                ReplayBuffer.add_from_directory() adds new samples to buffer (trainer.py:L314-L318)
```

---

#### B.3 What is Inside the Replay Buffer on Disk
* **Replay Buffer Directory Paths**:
  * `data/replay_buffer/images/`
  * `data/replay_buffer/labels/`
* **Real File Count on Disk**:
  * Direct filesystem inspection confirmed **0 files** in `data/replay_buffer/images/` and **0 files** in `data/replay_buffer/labels/`.
* **Sample Count**: **0 samples**.
* **Which Phase**: **None**. Neither Phase 1 nor Phase 2 data is currently stored in the replay buffer.

---

#### B.4 Is There Any Script That Trains the Autoencoder?
* **Answer**: **NO.** There is no standalone script to train the autoencoder.
* **Where Training Code Exists**:
  * Training logic is implemented exclusively as an internal method `DLAnomalyDetector.train()` in [src/anomaly/dl_detector.py:L139-L215](file:///d:/Projects/Retail_Shelf_intelligence-v2-/src/anomaly/dl_detector.py#L139-L215).
  * It is exposed only through the FastAPI HTTP route `POST /anomaly/dl/train` in [api/main.py:L640-L670](file:///d:/Projects/Retail_Shelf_intelligence-v2-/api/main.py#L640-L670).
  * There is no CLI script in `scripts/`, `src/anomaly/`, or the repository root to trigger autoencoder training from a terminal.

---

#### B.5 Does Anything in the Frontend Work Without Autoencoder Weights?
* **Answer**: **YES, almost the ENTIRE dashboard works completely without the autoencoder weights.**
* The Autoencoder is only used to compute visual surface reconstruction anomalies on cropped patches. All other features rely on separate subsystems:

| Dashboard Feature | Backend Subsystem Dependency | Works Without Autoencoder? | Proof / Source File |
| :--- | :--- | :--- | :--- |
| **Product Detection Bounding Boxes** | `ShelfDetector` (`YOLOv8`) | **YES** | [src/detection/detector.py](file:///d:/Projects/Retail_Shelf_intelligence-v2-/src/detection/detector.py) |
| **Zone Counts & Densities** | `ProductCounter` (Geometry) | **YES** | [src/detection/counter.py](file:///d:/Projects/Retail_Shelf_intelligence-v2-/src/detection/counter.py) |
| **Metrics Bar (Total, Identified, OOS)** | `rules.py`, `ProductCounter` | **YES** | [frontend/src/components/dashboard/metrics-bar.tsx](file:///d:/Projects/Retail_Shelf_intelligence-v2-/frontend/src/components/dashboard/metrics-bar.tsx) |
| **Empty Shelf Gap Alerts** | `AnomalyDetector.detect()` | **YES** | [src/anomaly/rules.py:L95](file:///d:/Projects/Retail_Shelf_intelligence-v2-/src/anomaly/rules.py#L95) |
| **Fallen Product Alerts** | `AnomalyDetector.detect()` | **YES** | [src/anomaly/rules.py:L137](file:///d:/Projects/Retail_Shelf_intelligence-v2-/src/anomaly/rules.py#L137) |
| **Low Stock Warnings** | `AnomalyDetector.detect()` | **YES** | [src/anomaly/rules.py:L170](file:///d:/Projects/Retail_Shelf_intelligence-v2-/src/anomaly/rules.py#L170) |
| **Brand Distribution Bar Chart** | `ProductIdentifier` + RapidFuzz | **YES** | [frontend/src/components/dashboard/product-chart.tsx](file:///d:/Projects/Retail_Shelf_intelligence-v2-/frontend/src/components/dashboard/product-chart.tsx) |
| **OCR Results Table** | `ProductIdentifier` (EasyOCR) | **YES** | [frontend/src/components/dashboard/ocr-results-table.tsx](file:///d:/Projects/Retail_Shelf_intelligence-v2-/frontend/src/components/dashboard/ocr-results-table.tsx) |
| **Planogram Compliance Card** | `PlanogramChecker` | **YES** | [src/anomaly/planogram.py](file:///d:/Projects/Retail_Shelf_intelligence-v2-/src/anomaly/planogram.py) |
| **History Run Browser** | SQLite Database | **YES** | [frontend/src/app/history/page.tsx](file:///d:/Projects/Retail_Shelf_intelligence-v2-/frontend/src/app/history/page.tsx) |
| **Settings Controls** | React Context (Client-side) | **YES** | [frontend/src/app/settings/page.tsx](file:///d:/Projects/Retail_Shelf_intelligence-v2-/frontend/src/app/settings/page.tsx) |
| **Visual Surface Defect Detection** | `DLAnomalyDetector` (`autoencoder.pth`) | **NO** | Skipped in [api/main.py:L197](file:///d:/Projects/Retail_Shelf_intelligence-v2-/api/main.py#L197) when untrained |

---

---

## 20. Problem Statement Mapping

The stated project goal is:
> *"A continual learning system that adapts retail shelf monitoring models to new products and layouts without catastrophic forgetting."*

### Deliverable 1: Continual Learning Vision Architecture
* **What Code Implements**:
  * Baseline YOLOv8 model initialized with pretrained weights `(src/detection/detector.py:L90)`.
  * Experience replay buffer storing images and labels via reservoir sampling `(src/continual_learning/replay_buffer.py:L43-L120)`.
  * Incremental fine-tuning runner freezing the first 10 backbone layers (`freeze=10`) and using reduced learning rate `lr0=0.001` `(src/continual_learning/trainer.py:L269-L271)`.
  * *Missing*: Elastic Weight Consolidation `(src/continual_learning/ewc.py)` is unintegrated dead code.
* **Status**: **PARTIAL**
* **Proof**:
  * [src/continual_learning/trainer.py](file:///d:/Projects/Retail_Shelf_intelligence-v2-/src/continual_learning/trainer.py#L269-L271): Shows `freeze=10` and `lr0=0.001`.
  * Verification Log #8: Confirms `ewc.py` is never called.

### Deliverable 2: Autonomous Shelf Anomaly Detection System
* **What Code Implements**:
  * Spatial heuristics evaluating vertical zones for empty shelf gaps, fallen products, and low stock `(src/anomaly/rules.py:L74-L160)`.
  * 4-layer Convolutional Autoencoder architecture for 256x256 image patches `(src/anomaly/autoencoder.py:L44-L74)`.
  * Synthetic planogram checker creating demo mismatch alerts `(src/anomaly/planogram.py:L16-L40)`.
  * *Missing*: Autoencoder weights file is absent from disk, disabling visual defect detection. No price tag checking logic exists.
* **Status**: **PARTIAL**
* **Proof**:
  * Verification Log #10: Live `/detect` request flagged 19 heuristic anomalies on real image.
  * Verification Log #10: API server logged `[DL Anomaly] No trained Autoencoder found. Needs training.`

### Deliverable 3: Incremental Product Adaptation Framework
* **What Code Implements**:
  * Automated orchestrator scanning sequential `data/phaseN/` directories `(scripts/run_continual_training.py:L14-L44)`.
  * Dynamic conversion of incoming CSV annotations into normalized YOLO bounding box text files `(src/continual_learning/trainer.py:L138-L204)`.
  * Active learning query module scoring unlabeled images by uncertainty (entropy, margin, confidence, count) `(src/continual_learning/active_learning.py:L60-L114)`.
  * *Limitation*: Phase 1 and Phase 2 training on disk both trained on a single generic class (`product`); no multi-class product expansion occurred.
* **Status**: **PARTIAL**
* **Proof**:
  * Verification Log #9: `YOLO('models/checkpoints/phase1/best1.pt').names` and `YOLO('models/checkpoints/phase2/finetune/weights/best.pt').names` both returned `{0: 'product'}`.
  * [src/continual_learning/active_learning.py](file:///d:/Projects/Retail_Shelf_intelligence-v2-/src/continual_learning/active_learning.py#L60).

### Deliverable 4: Retail Shelf Analytics Dashboard
* **What Code Implements**:
  * Full Next.js 16 web dashboard supporting drag-and-drop image uploads, batch uploads, and live webcam feed `(frontend/src/app/page.tsx)`.
  * Metrics bar showing total counts, out-of-stock gaps, and identified products `(frontend/src/components/dashboard/metrics-bar.tsx)`.
  * Brand distribution bar chart using Recharts `(frontend/src/components/dashboard/product-chart.tsx)`.
  * OCR results table and anomaly panel `(frontend/src/components/dashboard/ocr-results-table.tsx)`.
  * Historical runs browser with persistent SQLite storage `(frontend/src/app/history/page.tsx)`.
* **Status**: **WORKING**
* **Proof**:
  * Verification Log #10: Full API pipeline processed live image, returned bounding boxes, and logged to SQLite table `detections` (ID 6, 7).
  * [frontend/package.json](file:///d:/Projects/Retail_Shelf_intelligence-v2-/frontend/package.json), [frontend/src/app/page.tsx](file:///d:/Projects/Retail_Shelf_intelligence-v2-/frontend/src/app/page.tsx).

---

## 21. Final Verdict on Forgetting

**Did the project show that catastrophic forgetting was prevented?**

### Answer: **NO.**

### Explanation in Simple Words:
1. **Severe Accuracy Collapse**: In the real recorded Phase 1 training run `(models/checkpoints/phase1/train/results.csv)`, the model achieved **0.906 mAP50** (90.6% accuracy) and **0.913 precision**. When the Phase 2 incremental fine-tuning run took place `(models/checkpoints/phase2/finetune/results.csv)`, the accuracy immediately crashed to **0.091 mAP50** on epoch 1 and finished at **0.254 mAP50** on epoch 33. This represents a massive **65.2 percentage point drop** in detection accuracy on the validation data.
2. **Phase 2 Weights Were Never Used**: The model file actively deployed by the API `(models/checkpoints/best.pt)` is byte-for-byte identical to Phase 1 `(models/checkpoints/phase1/best1.pt)`. The degraded Phase 2 model was never placed into production.
3. **No Multi-Class Retention Was Tested**: Both Phase 1 and Phase 2 models were trained on the exact same single class (`product`). No new product classes were ever retained because no new classes were ever introduced into the neural network.

---

## 22. Safe to Claim vs Do NOT Claim

### Feature Status Table

| Feature | Status | Proof |
| :--- | :--- | :--- |
| **YOLOv8 Shelf Detection** | **WORKING** | Runs in API and detected 300 products in live test `(Verification Log #10)` |
| **Zone Counting (4 zones)** | **WORKING** | `ProductCounter` splits shelf and computes densities `(src/detection/counter.py)` |
| **Empty Shelf Gap Detection** | **WORKING** | Flagged 7 out-of-stock gaps on real test image `(Verification Log #10)` |
| **Fallen Product Detection** | **WORKING** | Flagged 12 fallen products by aspect ratio `(Verification Log #10)` |
| **Low Stock Warning** | **WORKING** | Threshold heuristic implemented in `src/anomaly/rules.py` |
| **PaddleOCR / EasyOCR Fallback** | **WORKING** | Automatically fell back to EasyOCR when worker failed `(Verification Log #10)` |
| **Fuzzy Catalog Matching** | **WORKING** | 33-brand dictionary matcher in `src/analytics/product_identifier.py` |
| **SQLite Persistence** | **WORKING** | Successfully inserted detection runs, inventory, and anomalies `(Verification Log #10)` |
| **Next.js Web Dashboard** | **WORKING** | Complete UI with upload, webcam, charts, and history `(frontend/src/app/page.tsx)` |
| **Active Learning Querying** | **WORKING** | Entropy and margin uncertainty scoring in `src/continual_learning/active_learning.py` |
| **Experience Replay Pipeline** | **PARTIAL** | Code implemented in `trainer.py`, but replay buffer on disk is empty `(Verification Log #1)` |
| **Layer Freezing (`freeze=10`)** | **PARTIAL** | Configured in `trainer.py` and recorded in `phase2/finetune/args.yaml:L31` |
| **DL Autoencoder Anomalies** | **PARTIAL** | Architecture coded, but weights file `autoencoder.pth` is missing `(Verification Log #10)` |
| **Planogram Verification** | **PARTIAL** | Hardcoded demo check; does not compare real floor plans `(src/anomaly/planogram.py)` |
| **Automated Test Suite** | **PARTIAL** | 29 unit tests pass, but `pytest tests/` fails on import `(Verification Log #2)` |
| **Elastic Weight Consolidation** | **NOT INTEGRATED** | `ewc.py` is dead code; never called during training `(Verification Log #8)` |
| **Multi-Class Product Expansion** | **NOT INTEGRATED** | Both Phase 1 and Phase 2 detect only 1 class (`product`) `(Verification Log #9)` |
| **Price Tag Missing Detection** | **NOT INTEGRATED** | No rule or anomaly logic exists for price tags in `src/anomaly/rules.py` |
| **Tailwind CSS Dashboard** | **NOT INTEGRATED** | Not installed in `frontend/package.json`; uses pure CSS Modules |

### What MUST NOT Be Presented as Working
* ❌ **Do NOT claim that Elastic Weight Consolidation (EWC) is preventing forgetting.** (The training loop never calls it).
* ❌ **Do NOT claim that catastrophic forgetting was successfully prevented in Phase 2.** (Phase 2 mAP50 dropped by 65.2 percentage points).
* ❌ **Do NOT claim the model detects distinct product classes.** (The YOLO model detects only a single class: `product`).
* ❌ **Do NOT claim deep learning autoencoder anomaly detection is active out-of-the-box.** (`autoencoder.pth` is missing from disk).
* ❌ **Do NOT claim rule-based anomalies detect missing price tags.** (There is no price tag check in `rules.py`).
* ❌ **Do NOT claim that the test suite passes out-of-the-box.** (Running `pytest tests/` crashes with `FileNotFoundError`).
* ❌ **Do NOT claim the frontend uses Tailwind CSS.** (It uses standard CSS Modules).

---

## 23. Likely Examiner Questions and Honest Answers

1. **Q: How does your system prevent catastrophic forgetting during continual learning?**  
   *A*: The code implements Experience Replay by buffering historical samples and mixes them into fine-tuning passes while freezing the first 10 backbone layers of YOLOv8. However, Elastic Weight Consolidation (`ewc.py`) was not integrated into the Ultralytics training loop.

2. **Q: Why did Phase 2 mAP50 drop from 0.906 to 0.254 in your results.csv?**  
   *A*: The Phase 2 fine-tuning run suffered performance degradation during incremental training on the small subset, showing that experience replay alone with layer freezing did not maintain Phase 1 accuracy under the training setup used.

3. **Q: Did you introduce new product classes in Phase 2?**  
   *A*: No. Inspecting the trained weights for both Phase 1 and Phase 2 confirms that both models only contain a single class (`product`). Product-level differentiation is handled downstream by the OCR catalog matcher.

4. **Q: Why are `best.pt` and `phase1/best1.pt` completely identical?**  
   *A*: Because Phase 2 fine-tuning degraded accuracy, the active checkpoint `best.pt` in the repository was retained from Phase 1 (`best1.pt`) to keep the detection pipeline functional.

5. **Q: Where is the trained Autoencoder model file for anomaly detection?**  
   *A*: `models/autoencoder.pth` is not currently present on disk. The Autoencoder architecture is implemented in `autoencoder.py` and can be trained via `POST /anomaly/dl/train`, but the weights file is absent.

6. **Q: Is there a standalone script to train the Autoencoder?**  
   *A*: No standalone command-line script exists. Training is implemented as a method inside `DLAnomalyDetector.train()` and exposed via the FastAPI endpoint `/anomaly/dl/train`.

7. **Q: Does your rule-based engine check for missing price tags?**  
   *A*: No. While price regex extraction exists in `src/analytics/ocr.py`, `AnomalyDetector.detect()` in `src/anomaly/rules.py` only checks for empty shelf gaps, fallen products, low stock, and misplaced products.

8. **Q: How does planogram compliance checking work?**  
   *A*: In the current code (`src/anomaly/planogram.py`), it is a demonstration function. It counts identified products and synthesizes an anomaly asserting the most frequent product is missing 2 facings.

9. **Q: Why do all anomaly tests in `test_detector.py` have `assert True`?**  
   *A*: The test methods in `TestAnomalyDetection` were written as stubs and not yet populated with assertions against `ShelfStats`.

10. **Q: Why does running `pytest tests/` fail with FileNotFoundError?**  
    *A*: `tests/test_api.py` and `tests/test_identifier.py` are written as top-level procedural scripts that attempt to load validation images from `data/processed/images/val` during pytest's test collection phase.

11. **Q: Why does `tests/test_model_accuracy.py --all` fail to evaluate models?**  
    *A*: `models/configs/data_kaggle.yaml` contains a hardcoded absolute path (`D:\retail_shelf_intelligence\data\processed`) that does not match the active project directory.

12. **Q: What is currently inside the replay buffer on disk?**  
    *A*: `data/replay_buffer/images/` and `labels/` are currently empty (0 samples). The buffer must be seeded via `POST /buffer/seed` or `trainer.seed_buffer_from_phase()`.

13. **Q: Why did PaddleOCR fall back to EasyOCR during the live API test?**  
    *A*: The installed version of PaddleOCR threw an `Unknown argument: rec_image_shape` error when initialized by the worker process. The built-in exception handler caught the failure and switched to EasyOCR.

14. **Q: Does the frontend dashboard use Tailwind CSS as stated in the README?**  
    *A*: No. The frontend is built using Vanilla CSS Modules (`*.module.css`) and global CSS custom properties. Tailwind CSS is not in `package.json`.

15. **Q: Which datasets were actually used to train the models?**  
    *A*: Both Phase 1 and Phase 2 training configurations (`args.yaml`) reference the SKU-110K dataset. No other public retail datasets (such as RPC or GroZi-120) were used.

---

## 24. What Can Realistically Be Fixed Before the Review

### Quick Fixes (Under 1 Hour Each)
1. **Fix `data_kaggle.yaml` Dataset Path (10 minutes)**:
   Change `path: D:\retail_shelf_intelligence\data\processed` in `models/configs/data_kaggle.yaml` to the relative path `data/processed` so that `test_model_accuracy.py` and `evaluate.py` can find validation images once data is placed there.
2. **Fix `pytest tests/` Import Crashes (20 minutes)**:
   Wrap `tests/test_api.py` and `tests/test_identifier.py` inside standard `def test_*` functions and guard missing file paths with `pytest.mark.skipif`, allowing `pytest tests/` to pass cleanly.
3. **Replace Stub Tests in `TestAnomalyDetection` (25 minutes)**:
   Replace the 7 `assert True` statements in `tests/test_detector.py` with real assertions testing `AnomalyDetector.detect()` on mock `ShelfStats`.
4. **Fix Hardcoded API URL in Frontend (10 minutes)**:
   Change `http://localhost:8000` in `frontend/src/app/page.tsx` to relative paths (`/api/health`, `/api/detect`) so all calls flow through Next.js proxy rewrites.
5. **Fix PaddleOCR Worker Argument (15 minutes)**:
   Remove `rec_image_shape` and `use_gpu` from `PaddleOCR(**ocr_kwargs)` in `src/analytics/paddle_ocr.py` so the native worker initializes without falling back to EasyOCR.
6. **Create a Standalone Autoencoder Training Script (30 minutes)**:
   Add a simple CLI script `scripts/train_autoencoder.py` that takes a folder of image crops, calls `DLAnomalyDetector.train()`, and writes `models/autoencoder.pth`.

### Risky Changes (Do NOT Attempt Right Before Review)
1. **Modifying Ultralytics Loss for EWC**:
   Attempting to inject Fisher information penalties into Ultralytics YOLO's closed C++/PyTorch loss functions requires overriding internal trainer classes and risks breaking detection training entirely.
2. **Re-training Multi-Class Models**:
   Training YOLO on multiple SKU classes from scratch requires full multi-class bounding box annotations and several hours of GPU compute time.
3. **Rewriting Frontend to Tailwind CSS**:
   The dashboard is already fully styled with CSS Modules. Migrating to Tailwind adds zero functionality and risks breaking component styling.
