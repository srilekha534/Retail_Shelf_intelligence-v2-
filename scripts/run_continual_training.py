import os
import sys
import argparse
from pathlib import Path
import re

# Ensure root directory is on the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import config as cfg
from src.continual_learning.trainer import IncrementalTrainer

def get_available_phases():
    """Scan data directory for all phaseN folders, returning sorted phase numbers."""
    phases = []
    if not os.path.exists(cfg.DATA_DIR):
        return phases
        
    for d in os.listdir(cfg.DATA_DIR):
        match = re.match(r"^phase(\d+)$", d)
        if match:
            phase_num = int(match.group(1))
            
            # Phase 1 is the base training (handled by src/detection/train.py)
            # Continual learning starts at Phase 2
            if phase_num == 1:
                continue
                
            # Check that it has images to process
            img_dir = os.path.join(cfg.DATA_DIR, d, "images")
            if os.path.exists(img_dir) and len(os.listdir(img_dir)) > 0:
                phases.append(phase_num)
    return sorted(phases)

def get_untrained_phases(phases):
    """Filter to only phases that haven't been trained yet."""
    untrained = []
    for p in phases:
        best_pt_path = os.path.join(cfg.CHECKPOINTS_DIR, f"phase{p}", f"best{p}.pt")
        if not os.path.exists(best_pt_path):
            untrained.append(p)
    return untrained

def train_phase(phase: int):
    print("\n" + "="*60)
    print(f"Starting Phase {phase} Fine-Tuning Setup...")
    print("="*60)
    
    # Initialize trainer with current best weights
    weights = cfg.BEST_WEIGHTS if os.path.exists(cfg.BEST_WEIGHTS) else cfg.MODEL_NAME
    trainer = IncrementalTrainer(weights_path=weights)
    
    # Seed buffer if it's currently empty (e.g. if the user didn't train phase 2 yet, 
    # we need the buffer seeded from phase 1 to avoid catastrophic forgetting)
    if trainer.buffer.size == 0:
        print("Replay buffer is empty. Seeding from Phase 1 data...")
        trainer.seed_buffer_from_phase(phase=1)
        
    phase_images = os.path.join(cfg.DATA_DIR, f"phase{phase}", "images")
    phase_labels = os.path.join(cfg.DATA_DIR, f"phase{phase}", "labels")
    
    # Run the incremental trainer. It automatically updates best.pt and the replay buffer!
    trainer.train_new_phase(
        new_images_dir=phase_images,
        new_labels_dir=phase_labels,
        class_names=cfg.CLASS_NAMES,
        phase=phase,
        epochs=cfg.CL_EPOCHS
    )
    print(f"Phase {phase} training job completed successfully.")

def main():
    parser = argparse.ArgumentParser(description="Run Continual Learning Phase(s)")
    parser.add_argument("--phase", type=int, help="Run training for a specific phase number")
    parser.add_argument("--all", action="store_true", help="Automatically find and train all untrained phases sequentially")
    args = parser.parse_args()

    if args.all:
        print("Scanning for available phases...")
        available = get_available_phases()
        untrained = get_untrained_phases(available)
        
        if not untrained:
            print("No untrained phases found. Everything is up to date!")
            return
            
        print(f"Found {len(untrained)} untrained phases: {untrained}")
        for p in untrained:
            train_phase(p)
            
    elif args.phase is not None:
        train_phase(args.phase)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
