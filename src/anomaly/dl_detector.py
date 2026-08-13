# src/anomaly/dl_detector.py
#
# PURPOSE:
#   Deep Learning Anomaly Detector using Autoencoder.
#   Trains on normal crops, infers anomalies on new crops by measuring reconstruction MSE.
#
# USAGE:
#   detector = DLAnomalyDetector()
#   scores = detector.predict_batch(crops)

import os
import sys
from pathlib import Path
from typing import List, Optional
import time

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as cfg
from src.anomaly.autoencoder import LightweightAutoencoder


class CropDataset(Dataset):
    """Simple dataset for a list of NumPy image crops."""
    def __init__(self, crops: List[np.ndarray], target_size: int = 256):
        self.crops = crops
        self.target_size = target_size

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        crop = self.crops[idx]
        
        # 1. Denoise (FastNlMeans)
        denoised = cv2.fastNlMeansDenoisingColored(crop, None, h=6, hColor=6, templateWindowSize=7, searchWindowSize=21)
        
        # 2. Contrast Enhance (CLAHE on L channel in LAB color space)
        lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        
        # 3. Resize to target size (e.g. 256x256)
        img = cv2.resize(enhanced, (self.target_size, self.target_size))
        
        # 4. Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # 5. HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        
        return torch.from_numpy(img)


class DLAnomalyDetector:
    def __init__(
        self,
        model_path: str = None,
        roi_size: int = None,
        batch_size: int = None,
        threshold: float = None,
        device: str = None,
        precision: str = None,
    ):
        self.roi_size = roi_size or getattr(cfg, "ANOMALY_ROI_SIZE", 256)
        self.batch_size = batch_size or getattr(cfg, "ANOMALY_BATCH_SIZE", 16)
        self.threshold = threshold or getattr(cfg, "ANOMALY_THRESHOLD", 0.5)
        self.device = device or getattr(cfg, "ANOMALY_DEVICE", "cuda")
        self.precision = precision or getattr(cfg, "ANOMALY_PRECISION", "fp16")
        
        self.model_path = model_path or os.path.join(cfg.MODELS_DIR, "autoencoder.pth")
        
        self.model = LightweightAutoencoder().to(self.device)
        self._is_trained = False
        
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
            self.model.eval()
            self._is_trained = True
            print(f"[DL Anomaly] Loaded Autoencoder from {self.model_path}")
        else:
            print("[DL Anomaly] No trained Autoencoder found. Needs training.")

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        print(f"[DL Anomaly] Saved Autoencoder to {self.model_path}")

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def train(
        self, 
        crops: List[np.ndarray], 
        epochs: int = 50, 
        lr: float = 1e-3
    ):
        """
        Train the Autoencoder on normal image crops.
        """
        if len(crops) == 0:
            print("[DL Anomaly] No crops provided for training.")
            return

        dataset = CropDataset(crops, target_size=self.roi_size)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        self.model.train()
        print(f"[DL Anomaly] Starting training on {len(crops)} crops for {epochs} epochs...")
        
        use_amp = (self.precision == "fp16" and "cuda" in self.device)
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for batch in dataloader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        reconstructed = self.model(batch)
                        loss = criterion(reconstructed, batch)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    reconstructed = self.model(batch)
                    loss = criterion(reconstructed, batch)
                    loss.backward()
                    optimizer.step()
                    
                total_loss += loss.item() * batch.size(0)
                
            avg_loss = total_loss / len(dataset)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"[DL Anomaly] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
                
        self._is_trained = True
        self.save_model()
        self.model.eval()

    def predict_batch(self, crops: List[np.ndarray]) -> List[float]:
        """
        Score a batch of crops. Returns MSE anomaly score for each crop.
        Scores > threshold are anomalies.
        """
        if not self._is_trained or len(crops) == 0:
            return [0.0] * len(crops)
            
        dataset = CropDataset(crops, target_size=self.roi_size)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        self.model.eval()
        scores = []
        
        use_amp = (self.precision == "fp16" and "cuda" in self.device)
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        reconstructed = self.model(batch)
                        batch_scores = self.model.compute_anomaly_score(batch, reconstructed)
                else:
                    reconstructed = self.model(batch)
                    batch_scores = self.model.compute_anomaly_score(batch, reconstructed)
                    
                scores.extend(batch_scores.cpu().numpy().tolist())
                
        # Normalize scores to [0, 1] for visualization
        # In reality, MSE can be > 1 if data is not normalized, but here data is [0,1]
        # We cap it at 1.0 for the UI
        scores = [min(s, 1.0) for s in scores]
                
        return scores
