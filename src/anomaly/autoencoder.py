# src/anomaly/autoencoder.py
#
# PURPOSE:
#   Lightweight Convolutional Autoencoder for Anomaly Detection.
#   Designed to run on 4GB VRAM GPUs with FP16 mixed precision.
#   Input: 256x256 RGB ROI (Product crop)
#   Bottleneck: 128 dim
#   Output: 256x256 RGB Reconstruction

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_last=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        ]
        if not is_last:
            layers.extend([
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
        else:
            layers.append(nn.Sigmoid()) # output in [0, 1]
            
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.block(x)


class LightweightAutoencoder(nn.Module):
    """
    Encoder: Conv Block x 4 (compresses 256x256 -> 16x16)
    Bottleneck: Conv to 128 channels
    Decoder: ConvTranspose x 4 (reconstructs 16x16 -> 256x256)
    """
    def __init__(self, in_channels=3, base_filters=32, bottleneck_dim=128):
        super().__init__()
        
        # Encoder (256x256 -> 128x128 -> 64x64 -> 32x32 -> 16x16)
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, base_filters),            # 128x128, 32
            ConvBlock(base_filters, base_filters * 2),       # 64x64, 64
            ConvBlock(base_filters * 2, base_filters * 4),   # 32x32, 128
            ConvBlock(base_filters * 4, bottleneck_dim),     # 16x16, 128 (Bottleneck)
        )
        
        # Decoder (16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256)
        self.decoder = nn.Sequential(
            ConvTransposeBlock(bottleneck_dim, base_filters * 4),
            ConvTransposeBlock(base_filters * 4, base_filters * 2),
            ConvTransposeBlock(base_filters * 2, base_filters),
            ConvTransposeBlock(base_filters, in_channels, is_last=True),
        )

    def forward(self, x):
        # x: (B, 3, 256, 256)
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def compute_anomaly_score(self, original, reconstructed):
        """
        Compute MSE loss per image in the batch as the anomaly score.
        original: (B, 3, H, W)
        reconstructed: (B, 3, H, W)
        Returns: (B,) tensor of MSE scores
        """
        # MSE per pixel
        mse = (original - reconstructed) ** 2
        # Average over channels, height, width -> (B,)
        score = mse.view(mse.size(0), -1).mean(dim=1)
        return score
