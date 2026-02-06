# ========================= backend/config.py =========================

import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "backend", "models", "mobilenetv3_best.pth")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

IMG_SIZE = 224
CONF_HIGH = 0.85
CONF_LOW = 0.65

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"