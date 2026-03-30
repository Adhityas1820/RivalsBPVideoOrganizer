"""
train_model.py
--------------
Loads frames from dataset_frames/<MapName>/, splits them into
train/val/test at the frame level, fine-tunes a pretrained ResNet18,
and saves the best checkpoint to models/map_classifier.pth.

Run AFTER process_data.py.

Usage:
    python train_model.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, models, transforms
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_FRAMES_DIR = "dataset_frames"
MODELS_DIR         = "models"
MODEL_PATH         = os.path.join(MODELS_DIR, "map_classifier.pth")

TRAIN_RATIO   = 0.75
VAL_RATIO     = 0.15
# TEST_RATIO  = remainder (0.10)

BATCH_SIZE    = 32
NUM_EPOCHS    = 15
LEARNING_RATE = 1e-4
NUM_WORKERS   = 0      # 0 is safest on Windows; raise to 4 if you have many cores
IMG_SIZE      = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
# ---------------------------------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class CachedDataset(torch.utils.data.Dataset):
    """Loads all images into RAM upfront so epochs never touch disk."""
    def __init__(self, image_folder):
        self.classes   = image_folder.classes
        self.class_to_idx = image_folder.class_to_idx
        print(f"Loading {len(image_folder)} frames into RAM...", flush=True)
        self.images = []
        self.labels = []
        for i, (path, label) in enumerate(image_folder.samples):
            self.images.append(image_folder.loader(path))
            self.labels.append(label)
            if (i + 1) % 500 == 0 or (i + 1) == len(image_folder.samples):
                print(f"\r  {i + 1}/{len(image_folder.samples)} loaded", end="", flush=True)
        print()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class TransformSubset(torch.utils.data.Dataset):
    """Wraps a Subset so each split can have its own transform."""
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train(training)
    running_loss = 0.0
    correct = 0
    total   = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    num_batches = len(loader)
    with ctx:
        for batch_idx, (inputs, labels) in enumerate(loader, 1):
            inputs, labels = inputs.to(device), labels.to(device)

            if training:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss    = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted  = outputs.max(1)
            correct       += predicted.eq(labels).sum().item()
            total         += labels.size(0)

            pct = 100 * batch_idx // num_batches
            label = "Train" if training else "Val  "
            print(f"\r  {label} {pct:3d}%  [{batch_idx}/{num_batches}]", end="", flush=True)

    print()
    return running_loss / total, 100.0 * correct / total


def main():
    frames_root = Path(DATASET_FRAMES_DIR)

    if not frames_root.exists():
        print(f"[ERROR] '{DATASET_FRAMES_DIR}/' not found. Run process_data.py first.")
        return

    # -----------------------------------------------------------------------
    # Skip empty class folders — ImageFolder errors if any subfolder has no images
    # -----------------------------------------------------------------------
    import shutil
    image_exts = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}
    empty = [d for d in sorted(frames_root.iterdir()) if d.is_dir()
             and d.name != "_skipped_empty"
             and not any(f.suffix.lower() in image_exts for f in d.rglob("*"))]
    if empty:
        print(f"[WARN] {len(empty)} map folder(s) have no frames and will be skipped:")
        for d in empty:
            print(f"       - {d.name}")
        print("       Run process_data.py after downloading videos for these maps.\n")
        # Move empty folders aside so ImageFolder doesn't error
        skipped_dir = frames_root.parent / "_skipped_empty"
        skipped_dir.mkdir(exist_ok=True)
        for d in empty:
            shutil.move(str(d), str(skipped_dir / d.name))

    # -----------------------------------------------------------------------
    # Load all frames with no transform — TransformSubset applies per-split
    # transforms on PIL images, so the base dataset must return PIL images.
    # -----------------------------------------------------------------------
    full_dataset = CachedDataset(datasets.ImageFolder(frames_root, transform=None))

    num_classes = len(full_dataset.classes)
    print(f"Classes ({num_classes}): {full_dataset.classes}")
    print(f"Total frames : {len(full_dataset)}")

    if num_classes < 2:
        print("[ERROR] Need at least 2 map classes to train.")
        return

    # -----------------------------------------------------------------------
    # Frame-level train / val / test split
    # -----------------------------------------------------------------------
    n       = len(full_dataset)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    n_test  = n - n_train - n_val

    train_sub, val_sub, test_sub = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"Train frames : {len(train_sub)}")
    print(f"Val frames   : {len(val_sub)}")
    print(f"Test frames  : {len(test_sub)}\n")

    # Apply per-split transforms
    train_dataset = TransformSubset(train_sub, train_transform)
    val_dataset   = TransformSubset(val_sub,   eval_transform)
    test_dataset  = TransformSubset(test_sub,  eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # -----------------------------------------------------------------------
    # Model / loss / optimizer
    # -----------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    model     = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    os.makedirs(MODELS_DIR, exist_ok=True)
    best_val_acc = 0.0

    def restore_skipped():
        skipped_dir = frames_root.parent / "_skipped_empty"
        if skipped_dir.exists():
            for d in skipped_dir.iterdir():
                shutil.move(str(d), str(frames_root / d.name))
            skipped_dir.rmdir()

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>8}")
    print("-" * 55)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, training=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, optimizer, device, training=False)
        scheduler.step()

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_acc":          val_acc,
                "classes":          full_dataset.classes,
            }, MODEL_PATH)
            marker = "  <- saved"

        print(
            f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>8.2f}%"
            f"  {val_loss:>8.4f}  {val_acc:>8.2f}%{marker}"
        )

    restore_skipped()
    print(f"\nBest val accuracy : {best_val_acc:.2f}%")
    print(f"Model saved to    : {MODEL_PATH}")

    # -----------------------------------------------------------------------
    # Final test evaluation (one-time on held-out set)
    # -----------------------------------------------------------------------
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, training=False)
    print(f"\n{'=' * 55}")
    print(f"Final test accuracy : {test_acc:.2f}%  (loss: {test_loss:.4f})")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
