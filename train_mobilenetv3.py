import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
from sklearn.metrics import accuracy_score

def main():

    # =========================================================
    # PERFORMANCE / STABILITY
    # =========================================================
    torch.backends.cudnn.benchmark = True

    # =========================================================
    # CONFIGURATION
    # =========================================================
    DATA_DIR = "D:\Projects\AI_ML_DL\Plant_Disease_project\Dataset"
    IMG_SIZE = 224
    BATCH_SIZE = 32          # reduce to 24 if OOM
    NUM_WORKERS = 6

    EPOCHS_HEAD = 10
    EPOCHS_FINE = 35
    EARLY_STOPPING_PATIENCE = 7

    LR_HEAD = 1e-3
    LR_FINE = 1e-4

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # =========================================================
    # IMAGENET NORMALIZATION
    # =========================================================
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    # =========================================================
    # TRANSFORMS (REAL-WORLD SIMULATION)
    # =========================================================
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.55, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),

        transforms.ColorJitter(
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.1
        ),

        transforms.RandomPerspective(distortion_scale=0.35, p=0.4),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),

        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    # =========================================================
    # DATASETS
    # =========================================================
    train_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "train"),
        transform=train_transforms
    )

    val_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "valid"),
        transform=val_transforms
    )

    CLASS_NAMES = train_dataset.classes
    NUM_CLASSES = len(CLASS_NAMES)

    print(f"Classes detected: {NUM_CLASSES}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # =========================================================
    # MODEL
    # =========================================================
    model = models.mobilenet_v3_large(weights="IMAGENET1K_V1")

    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        NUM_CLASSES
    )

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    # =========================================================
    # TRAIN / VALIDATE FUNCTIONS
    # =========================================================
    def train_one_epoch(model, loader, optimizer):
        model.train()
        running_loss = 0.0

        for imgs, labels in tqdm(loader, leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        return running_loss / len(loader)

    def validate(model, loader):
        model.eval()
        preds, targets = [], []

        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(DEVICE)
                outputs = model(imgs)
                predicted = torch.argmax(outputs, dim=1)

                preds.extend(predicted.cpu().numpy())
                targets.extend(labels.numpy())

        return accuracy_score(targets, preds)

    # =========================================================
    # STAGE 1 — TRAIN CLASSIFIER HEAD
    # =========================================================
    print("\n🚀 Stage 1: Training classifier head")

    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(
        model.classifier.parameters(),
        lr=LR_HEAD,
        weight_decay=1e-4
    )

    for epoch in range(EPOCHS_HEAD):
        loss = train_one_epoch(model, train_loader, optimizer)
        val_acc = validate(model, val_loader)

        print(
            f"[HEAD] Epoch {epoch+1}/{EPOCHS_HEAD} | "
            f"Loss: {loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    # =========================================================
    # STAGE 2 — PARTIAL FINE-TUNING
    # =========================================================
    print("\n🔥 Stage 2: Partial fine-tuning")

    for layer in model.features[-4:]:
        for param in layer.parameters():
            param.requires_grad = True

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_FINE,
        weight_decay=1e-4
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        
    )

    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(EPOCHS_FINE):
        loss = train_one_epoch(model, train_loader, optimizer)
        val_acc = validate(model, val_loader)

        scheduler.step(val_acc)

        print(
            f"[FINE] Epoch {epoch+1}/{EPOCHS_FINE} | "
            f"Loss: {loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0

            torch.save({
                "model_state": model.state_dict(),
                "class_names": CLASS_NAMES
            }, "mobilenetv3_best.pth")

        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print("⏹️ Early stopping triggered")
            break

    print("\n✅ Training completed")
    print(f"🏆 Best Validation Accuracy: {best_val_acc:.4f}")
    print("📦 Model saved as mobilenetv3_best.pth")

if __name__ == "__main__":
    mp.freeze_support()   # REQUIRED on Windows
    main()
