# CheatCam Hardware Acceleration Guide

You now have two optimized versions of the CheatCam application designed to leverage your **RTX 3060 12GB**.

## 1. The Two Versions

### 🚀 Version A: `app_auto_hw.py` (Auto Detection)
- **Goal**: Reliable, cross-platform performance.
- **Behavior**: It checks if `torch` can see a CUDA device. If yes, it uses the GPU for YOLOv8. If not, it falls back to CPU automatically.
- **MediaPipe**: Uses the default delegate (usually CPU on Windows, but very efficient).
- **Use Case**: Best for general use or if you want the code to work correctly even if CUDA drivers have issues.

### 🔥 Version B: `app_nvidia_cuda.py` (NVIDIA/CUDA Optimized)
- **Goal**: Maximum performance for NVIDIA GPUs.
- **Behavior**:
  - **YOLOv8**: Explicitly forced to `device='cuda'`.
  - **MediaPipe**: Configured to use `Delegate.GPU`. This offloads face and hand landmarking to your RTX 3060, significantly reducing CPU load.
  - **Diagnostics**: Prints your GPU name and available VRAM (12GB) on startup.
- **Use Case**: Best for production or when monitoring many students simultaneously.

---

## 🛠️ Setup for RTX 3060

To actually use the GPU, standard `pip install` is often not enough because it installs the CPU version of PyTorch by default. Follow these steps:

### Step 1: Install CUDA-Enabled PyTorch
Run this in your terminal:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
*(The `cu121` refers to CUDA 11.8 or 12.1 is common, here we suggest 12.1).*

### Step 2: Install Remaining Dependencies
```bash
pip install -r requirements_gpu.txt
```

---

## 🏃 How to Run

### To run the Auto-Detect version:
```bash
python app_auto_hw.py
```
*(Runs on port `5001`)*

### To run the NVIDIA Optimized version:
```bash
python app_nvidia_cuda.py
```
*(Runs on port `5000`)*

> [!TIP]
> With your **12GB VRAM**, you can easily run both or even upgrade to a larger YOLO model (like `yolov8m.pt` or `yolov8l.pt`) in the future by changing the filename in the code!
