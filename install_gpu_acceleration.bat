@echo off
echo =======================================================
echo Upgrading AI Models for NVIDIA RTX 2050 (CUDA 11.8)
echo =======================================================

echo.
echo [1/3] Uninstalling CPU versions...
pip uninstall -y torch torchvision torchaudio paddlepaddle

echo.
echo [2/3] Installing PyTorch with CUDA 11.8 support...
echo (Note: This is a ~2.8 GB download and may take several minutes)
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo [3/3] Installing PaddlePaddle with GPU support...
echo (Note: This is a ~450 MB download)
pip install --no-cache-dir paddlepaddle-gpu

echo.
echo =======================================================
echo GPU Upgrade Complete! Please restart the API server.
echo =======================================================
pause
