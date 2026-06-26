# ATM Security System — OWL-ViT Face Coverage Detector

A computer vision security system that detects whether a person's face is concealed by **sunglasses**, **face masks**, or **hats** before allowing an ATM transaction. Built with [OWL-ViT](https://huggingface.co/google/owlvit-base-patch32) (zero-shot object detection) and OpenCV.

---

## 🔍 How It Works

The system runs a **3-layer detection pipeline**:

| Layer | Method | Detects |
|-------|--------|---------|
| **1** | OWL-ViT (primary prompts) | `human face`, `sunglasses`, `face mask`, `hat` |
| **2** | OWL-ViT (alt prompts) | `surgical mask`, `medical face mask`, `baseball cap`, etc. |
| **3** | OpenCV CV fallback | HSV color mask detection + face visibility ratio |

### Security Decision Logic

```
Is a face detected?
  NO  → ❌ DENIED  (identity cannot be verified)
  YES → Does a cover (glasses/mask/hat) overlap with the face? (IoU ≥ 0.10)
          YES → ❌ DENIED  (face is concealed)
          NO  → Is the visible face area < 40% of the image? (hat pulled low)
                  YES → ❌ DENIED  (face partially hidden)
                  NO  → ✅ ALLOWED (transaction proceeds)
```

### Test Results

| Input | Condition | Result |
|-------|-----------|--------|
| `input.jpg` | Clear face | ✅ **ALLOWED** |
| `input2.png` | Sunglasses | ❌ **DENIED** — `sunglasses` IoU=0.26 |
| `input3.png` | Face mask | ❌ **DENIED** — `face mask` IoU=0.33 |
| `input4.jpeg` | Hat (face looking down) | ❌ **DENIED** — visibility=0.09 |

---

## Requirements

- Python 3.10+
- PyTorch 2.0+ (CUDA **or** ROCm)
- ~1 GB disk space (OWL-ViT model download on first run)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Wild3005/owlVit_deepLearning.git
cd owlVit_deepLearning
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate      # Linux / macOS
# venv\Scripts\activate       # Windows
```

### 3. Install PyTorch

Install PyTorch matching your hardware. Visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for the exact command.

**NVIDIA GPU (CUDA 12.1):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**AMD GPU (ROCm 6.1):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1
```

**CPU only:**
```bash
pip install torch torchvision
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

> **First run:** The OWL-ViT model (~1 GB) will be automatically downloaded from Hugging Face and cached in `~/.cache/huggingface/`.

---

## Usage

### Static image

```bash
# Run on a single image (opens result window)
python3 atm_security.py --image images/input2.png

# Headless mode — save result without opening a window
python3 atm_security.py --image images/input2.png --no-display
```

### Live webcam

```bash
python3 atm_security.py --camera
# Press 'q' or ESC to quit
```

### All arguments

```
usage: atm_security.py [-h] (--image IMAGE | --camera) [--no-display]

  --image IMAGE   Path to image file
  --camera        Use webcam (real-time)
  --no-display    Save result image without opening GUI window
```

Output images are saved automatically to `save_images/atm_result_<timestamp>.jpg`.

---

## Project Structure

```
owlVit_deepLearning/
│
├── atm_security.py     # Main ATM security system (OWL-ViT + OpenCV)
├── test_img.py         # Simple single-image OWL-ViT demo
├── test.py             # Simple webcam OWL-ViT demo
│
├── images/             # Input test images
│   ├── input.jpg       # Clear face (→ ALLOWED)
│   ├── input2.png      # Sunglasses (→ DENIED)
│   ├── input3.png      # Face mask  (→ DENIED)
│   └── input4.jpeg     # Hat        (→ DENIED)
│
├── save_images/        # Detection output images (auto-generated)
├── requirements.txt
└── README.md
```

---

## Configuration

Key thresholds can be tuned at the top of `atm_security.py`:

```python
DETECTION_THRESHOLD   = 0.13   # OWL-ViT minimum confidence
IOU_OVERLAP_THRESHOLD = 0.10   # Minimum face-cover overlap to trigger DENIED
FACE_COVER_RATIO      = 0.40   # Minimum face visibility ratio (vs image size)
```

---

## Model

- **Model:** [`google/owlvit-base-patch32`](https://huggingface.co/google/owlvit-base-patch32)
- **Type:** Open-vocabulary zero-shot object detection
- **Framework:** HuggingFace Transformers 4.52.4
- **Fallback:** OpenCV HSV color segmentation for mask/hat cases where OWL-ViT confidence is low

---

## License

This project is for educational and research purposes.
