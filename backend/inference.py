# backend/inference.py

import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from collections import defaultdict

from backend.config import MODEL_PATH, IMG_SIZE, DEVICE

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

device = DEVICE

checkpoint = torch.load(MODEL_PATH, map_location=device)

CLASS_NAMES = checkpoint["class_names"]
NUM_CLASSES = len(CLASS_NAMES)

model = models.mobilenet_v3_large(weights=None)
model.classifier[3] = torch.nn.Linear(
    model.classifier[3].in_features,
    NUM_CLASSES
)

model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])


def predict(image_path: str, top_k: int = 5) -> dict:
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)[0]

    # ---------- TOP CLASS ----------
    confidence, idx = torch.max(probs, dim=0)
    predicted_class = CLASS_NAMES[idx.item()]

    if "_" in predicted_class:
        predicted_crop, predicted_disease = predicted_class.split("_", 1)
    else:
        predicted_crop = predicted_class
        predicted_disease = "Unknown"

    # ---------- CROP CONFIDENCE GATE ----------
    crop_scores = defaultdict(float)

    for prob, cls in zip(probs, CLASS_NAMES):
        crop = cls.split("_")[0]
        crop_scores[crop] += prob.item()

    sorted_crops = sorted(
        crop_scores.items(), key=lambda x: x[1], reverse=True
    )

    top_crop, top_crop_conf = sorted_crops[0]
    second_crop_conf = sorted_crops[1][1] if len(sorted_crops) > 1 else 0.0

    crop_uncertain = (top_crop_conf - second_crop_conf) < 0.15

    # ---------- TOP-K ----------
    top_probs, top_idxs = torch.topk(probs, top_k)
    top_k_results = [
        {
            "class": CLASS_NAMES[i.item()],
            "prob": round(p.item(), 4)
        }
        for p, i in zip(top_probs, top_idxs)
    ]

    return {
        "predicted_class": predicted_class,
        "predicted_crop": predicted_crop,
        "predicted_disease": predicted_disease,
        "confidence": round(confidence.item(), 4),
        "crop_confidence": round(top_crop_conf, 4),
        "crop_uncertain": crop_uncertain,
        "top_k": top_k_results
    }