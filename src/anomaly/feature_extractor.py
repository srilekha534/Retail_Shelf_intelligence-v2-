# src/anomaly/feature_extractor.py
#
# PURPOSE:
#   Lightweight Backbone (EfficientNet-B0) for Feature Extraction.
#   Extracts high-level features from each ROI / image patch.
#   Output: 1280-dim Feature Vector.
#   Used as an alternative for PatchCore / OC-SVM anomaly detection.

import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetFeatureExtractor(nn.Module):
    """
    Extracts 1280-dim feature vector from EfficientNet-B0.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        # Load pretrained EfficientNet-B0
        if pretrained:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            self.model = models.efficientnet_b0(weights=weights)
        else:
            self.model = models.efficientnet_b0(weights=None)
            
        # We only want the features, not the classifier head
        self.features = self.model.features
        self.pooling = nn.AdaptiveAvgPool2d(1)
        
        # Freeze weights if used only for feature extraction
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # x: (B, 3, 256, 256)
        x = self.features(x)         # (B, 1280, 8, 8)
        x = self.pooling(x)          # (B, 1280, 1, 1)
        x = torch.flatten(x, 1)      # (B, 1280)
        return x
