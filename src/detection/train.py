# src/detection/train.py
#
# PURPOSE:
#   Train YOLOv8n on the prepared SKU-110K dataset.
#   On CPU this will be slow — 30 epochs on 500 images takes ~2-4 hours.
#   Use EPOCHS=5 in config.py for a quick smoke test first.
#
# USAGE:
#   python -m src.detection.train
#
# OUTPUT:
#   models/checkpoints/best.pt   ← use this for inference
#   models/checkpoints/last.pt   ← last epoch weights
#   runs/detect/train*/          ← training logs, plots (created by ultralytics)

import os
import sys
from pathlib import Path

# Make sure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ultralytics import YOLO
import torch
import config as cfg


def train():
    # ACCURACY TIPS:
    # 1. RESOLUTION: Increase IMG_SIZE in config.py to 640 or 1024. Shelf images have 
    #    tiny products, and higher resolutions directly improve mAP for small bboxes.
    # 2. EPOCHS: Train for 100-150 epochs. YOLOv8 uses early stopping, so it stops 
    #    automatically when learning plateaus.
    # 3. ACTIVE LEARNING: Use active_learning.py to retrieve highly uncertain shelf 
    #    images, annotate them, and add them back to your training dataset.
    print("=" * 60)
    print("YOLOv8 Training — Retail Shelf Detection")
    print("=" * 60)
    print(f"  Model:      {cfg.MODEL_NAME}")
    print(f"  Dataset:    {cfg.DATASET_YAML}")
    print(f"  Epochs:     {cfg.EPOCHS}")
    print(f"  Batch size: {cfg.BATCH_SIZE}")
    print(f"  Image size: {cfg.IMG_SIZE}")
    print(f"  Config device: {cfg.DEVICE}")

    cuda_available = torch.cuda.is_available()
    mps_available = getattr(
        torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    device = cfg.DEVICE
    if device == "cuda" and not cuda_available:
        print("  [WARNING] CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    if device == "mps" and not mps_available:
        print("  [WARNING] MPS requested but not available. Falling back to CPU.")
        device = "cpu"
    if device == "cuda":
        print(f"  CUDA available: {torch.cuda.device_count()} device(s)")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    elif device == "mps":
        print("  MPS backend available. Using Apple GPU.")
    else:
        if cuda_available or mps_available:
            print(
                "  A GPU backend is available, but using CPU because cfg.DEVICE is set to 'cpu'.")
        else:
            print("  No GPU backend is available. Using CPU.")
    print(f"  Device:     {device}")
    print()

    # Validate dataset YAML exists
    if not os.path.exists(cfg.DATASET_YAML):
        print(f"[ERROR] Dataset YAML not found: {cfg.DATASET_YAML}")
        print("Run dataset preparation first:")
        print("  python -m src.utils.prepare_dataset")
        return

    # Check if resuming from checkpoint
    last_checkpoint = os.path.join(
        cfg.CHECKPOINTS_DIR, "phase1", "train", "weights", "last.pt")
    resume = False
    if os.path.exists(last_checkpoint):
        print(f"  Resuming from checkpoint: {last_checkpoint}")
        model = YOLO(last_checkpoint)
        resume = True
    else:
        print(f"  Starting fresh training")
        # YOLOv8n pretrained on COCO — we fine-tune it on retail products.
        # This is transfer learning: the backbone already knows how to detect
        # objects; we just teach it what retail shelf objects look like.
        model = YOLO(cfg.MODEL_NAME)

    # Train
    results = None
    try:
        results = model.train(
            data=cfg.DATASET_YAML,
            epochs=cfg.EPOCHS,
            batch=cfg.BATCH_SIZE,
            imgsz=cfg.IMG_SIZE,
            device=device,
            workers=cfg.WORKERS,
            project=os.path.join(cfg.CHECKPOINTS_DIR, "phase1"),
            name="train",
            exist_ok=True,       # overwrite previous run folder
            # only use pretrained weights if starting fresh
            pretrained=True if not resume else False,
            resume=resume,     # resume from checkpoint if available
            patience=10,         # stop early if val loss doesn't improve
            save=True,
            plots=True,       # saves confusion matrix, loss curves etc.
            verbose=True,

            # Enable mixed precision on GPU for better throughput
            amp=(device == "cuda"),
            # Faster data transfer to GPU
            half=(device == "cuda"),
            # Caching disabled to prevent host CPU MemoryError during spawning
            cache=False,

            # Tuned data augmentation — highly effective for dense retail shelf datasets
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,        # mosaic augmentation (4 images combined) - improves small object detection
            degrees=10.0,       # small rotation for hand-held camera tilt
            translate=0.1,
            scale=0.5,         # increased scale variation for distance differences
        )
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] Training interrupted by exception: {e}")
    finally:
        # Copy best/last weights to their phase folders and standard locations
        best_src = os.path.join(cfg.CHECKPOINTS_DIR, "phase1", "train", "weights", "best.pt")
        last_src = os.path.join(cfg.CHECKPOINTS_DIR, "phase1", "train", "weights", "last.pt")
        
        import shutil
        
        # Save to phase1/ directory as best1.pt and last1.pt
        phase1_dir = os.path.join(cfg.CHECKPOINTS_DIR, "phase1")
        os.makedirs(phase1_dir, exist_ok=True)
        
        if os.path.exists(best_src):
            shutil.copy2(best_src, os.path.join(phase1_dir, "best1.pt"))
            shutil.copy2(best_src, cfg.BEST_WEIGHTS)
            print(f"\nPhase 1 best weights saved -> {os.path.join(phase1_dir, 'best1.pt')}")
            print(f"Latest best weights updated -> {cfg.BEST_WEIGHTS}")
        else:
            print(f"\n[WARNING] best.pt not found at expected path: {best_src}")
            print("Check the 'runs/' directory for your weights.")
            
        if os.path.exists(last_src):
            shutil.copy2(last_src, os.path.join(phase1_dir, "last1.pt"))
            last_dest = os.path.join(cfg.CHECKPOINTS_DIR, "last.pt")
            shutil.copy2(last_src, last_dest)
            print(f"Phase 1 last weights saved -> {os.path.join(phase1_dir, 'last1.pt')}")
            print(f"Latest last weights updated -> {last_dest}")

    # Print summary
    if results is not None:
        print("\n" + "=" * 60)
        print("Training complete.")
        print(
            f"Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
        print(f"Weights -> {cfg.BEST_WEIGHTS}")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Training ended prematurely (interrupted or error).")
        print(f"Partial weights saved to -> {cfg.BEST_WEIGHTS}")
        print("=" * 60)
    print("\nNext step: test inference with:")
    print("  python -m src.detection.detector --image path/to/shelf.jpg")


if __name__ == "__main__":
    train()
