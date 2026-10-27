
from pathlib import Path
import argparse
import csv
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
INDEX_FILE = PROJECT_DIR / "datasets" / "prepared" / "dataset_index.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
IMAGE_SIZE = 160
BATCH_SIZE = 16
EPOCHS = 12
PATIENCE = 4
NUM_WORKERS = 0

LEARNING_RATE = 0.00005
WEIGHT_DECAY = 0.0001

DEVICE = torch.device("cpu")

random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Train a crop-specific plant disease classifier."
)

parser.add_argument(
    "--crop",
    required=True,
    choices=["paddy", "corn", "chilli", "cotton"],
    help="Crop to train.",
)

ARGS = parser.parse_args()
CROP = ARGS.crop.lower()

CHECKPOINT_FILE = MODEL_DIR / f"{CROP}_best.pth"
HISTORY_FILE = MODEL_DIR / f"{CROP}_training_history.csv"


# ============================================================
# DATASET
# ============================================================

class PlantDataset(Dataset):
    def __init__(self, rows, label_to_id, transform):
        self.rows = rows
        self.label_to_id = label_to_id
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]

        with Image.open(row["image_path"]) as img:
            image = self.transform(img.convert("RGB"))

        label_id = self.label_to_id[row["label"]]

        return image, label_id


def read_index():
    if not INDEX_FILE.is_file():
        raise FileNotFoundError(
            f"Dataset index not found: {INDEX_FILE}"
        )

    with INDEX_FILE.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        rows = list(csv.DictReader(file))

    crop_rows = [
        row for row in rows
        if row["crop"].strip().lower() == CROP
        and row["split"].strip().lower() in {"train", "validation"}
    ]

    if not crop_rows:
        raise RuntimeError(
            f"No records found for crop '{CROP}'. "
            "Check the crop name in dataset_index.csv."
        )

    classes = sorted({
        row["label"] for row in crop_rows
    })

    label_to_id = {
        label: index
        for index, label in enumerate(classes)
    }

    train_rows = [
        row for row in crop_rows
        if row["split"].strip().lower() == "train"
    ]

    val_rows = [
        row for row in crop_rows
        if row["split"].strip().lower() == "validation"
    ]

    if not train_rows or not val_rows:
        raise RuntimeError(
            "Training or validation data is empty."
        )

    # Fixed ordering and seed make row selection reproducible.
    train_rows.sort(key=lambda row: row["image_path"])
    val_rows.sort(key=lambda row: row["image_path"])

    random.Random(SEED).shuffle(train_rows)

    return classes, label_to_id, train_rows, val_rows


# ============================================================
# VALIDATION
# ============================================================

def evaluate(model, loader, criterion, epoch):
    model.eval()

    loss_sum = 0.0
    correct = 0
    total = 0

    progress = tqdm(
        loader,
        desc=f"Validation {epoch}/{EPOCHS}",
        unit="batch",
        dynamic_ncols=True,
    )

    with torch.inference_mode():
        for images, labels in progress:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            count = labels.size(0)

            loss_sum += loss.item() * count
            correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            total += count

            progress.set_postfix(
                loss=f"{loss.item():.3f}",
                accuracy=f"{100 * correct / total:.1f}%",
                images=f"{total}/{len(loader.dataset)}",
            )

    return loss_sum / total, correct / total


# ============================================================
# TRAINING
# ============================================================

def main():
    overall_start = time.time()

    print("=" * 65, flush=True)
    print(f"{CROP.upper()} DISEASE CLASSIFIER TRAINING", flush=True)
    print("=" * 65, flush=True)

    print(f"Device: {DEVICE}", flush=True)
    print(f"Image size: {IMAGE_SIZE} x {IMAGE_SIZE}", flush=True)
    print(f"Batch size: {BATCH_SIZE}", flush=True)
    print(f"Maximum epochs: {EPOCHS}", flush=True)
    print(f"Learning rate: {LEARNING_RATE}", flush=True)

    classes, label_to_id, train_rows, val_rows = read_index()

    print(f"\nNumber of classes: {len(classes)}", flush=True)
    print(f"Training images: {len(train_rows)}", flush=True)
    print(f"Validation images: {len(val_rows)}", flush=True)

    for label in classes:
        count = sum(
            row["label"] == label for row in train_rows
        )
        print(f"  {label}: {count} training images", flush=True)

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = PlantDataset(
        train_rows, label_to_id, train_transform
    )

    val_dataset = PlantDataset(
        val_rows, label_to_id, val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # Class weights are calculated using training data only.
    class_counts = torch.zeros(
        len(classes), dtype=torch.float32
    )

    for row in train_rows:
        class_counts[label_to_id[row["label"]]] += 1

    if torch.any(class_counts == 0):
        raise RuntimeError(
            "At least one class has no training images."
        )

    class_weights = len(train_rows) / (
        len(classes) * class_counts
    )
    class_weights = class_weights / class_weights.mean()

    print("\nClass weights:", flush=True)
    for label, weight in zip(classes, class_weights.tolist()):
        print(f"  {label}: {weight:.3f}", flush=True)

    print(
        "\nLoading pretrained MobileNetV3-Small...",
        flush=True,
    )

    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)

    # Freeze early layers and fine-tune the final feature blocks.
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    for parameter in model.features[-2:].parameters():
        parameter.requires_grad = True

    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        len(classes),
    )

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(DEVICE)
    )

    optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
        min_lr=1e-6,
    )

    best_val_accuracy = -1.0
    best_epoch = 0
    no_improvement = 0
    epoch_times = []
    history = []

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model.train()

        loss_sum = 0.0
        correct = 0
        total = 0

        print(
            f"\n{'=' * 20} EPOCH {epoch}/{EPOCHS} "
            f"{'=' * 20}",
            flush=True,
        )

        progress = tqdm(
            train_loader,
            desc=f"Training {epoch}/{EPOCHS}",
            unit="batch",
            dynamic_ncols=True,
        )

        for images, labels in progress:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            count = labels.size(0)

            loss_sum += loss.item() * count
            correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            total += count

            progress.set_postfix(
                loss=f"{loss.item():.3f}",
                accuracy=f"{100 * correct / total:.1f}%",
                images=f"{total}/{len(train_dataset)}",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}",
            )

        train_loss = loss_sum / total
        train_accuracy = correct / total

        val_loss, val_accuracy = evaluate(
            model, val_loader, criterion, epoch
        )

        scheduler.step(val_accuracy)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        average_epoch_time = sum(epoch_times) / len(epoch_times)
        eta_seconds = average_epoch_time * (EPOCHS - epoch)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": epoch_time,
        })

        print(
            f"\nEpoch {epoch}/{EPOCHS}\n"
            f"  Train loss: {train_loss:.4f}\n"
            f"  Train accuracy: {train_accuracy * 100:.2f}%\n"
            f"  Validation loss: {val_loss:.4f}\n"
            f"  Validation accuracy: {val_accuracy * 100:.2f}%\n"
            f"  Learning rate: {optimizer.param_groups[0]['lr']:.2e}\n"
            f"  Epoch time: {epoch_time / 60:.2f} minutes\n"
            f"  Overall progress: {epoch / EPOCHS * 100:.1f}%\n"
            f"  Estimated remaining: {eta_seconds / 60:.2f} minutes",
            flush=True,
        )

        # Save the best checkpoint, not merely the final epoch.
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            no_improvement = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "label_to_id": label_to_id,
                "crop": CROP,
                "image_size": IMAGE_SIZE,
                "architecture": "mobilenet_v3_small",
                "validation_accuracy": best_val_accuracy,
                "validation_loss": val_loss,
                "best_epoch": best_epoch,
                "seed": SEED,
                "train_image_count": len(train_rows),
                "validation_image_count": len(val_rows),
            }, CHECKPOINT_FILE)

            print(
                f"Saved best checkpoint:\n{CHECKPOINT_FILE}",
                flush=True,
            )
        else:
            no_improvement += 1
            print(
                f"No improvement: {no_improvement}/{PATIENCE}",
                flush=True,
            )

        # Save progress after every epoch.
        with HISTORY_FILE.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(history[0].keys()),
            )
            writer.writeheader()
            writer.writerows(history)

        if no_improvement >= PATIENCE:
            print(
                "\nEarly stopping: validation accuracy did not improve.",
                flush=True,
            )
            break

    print("\n" + "=" * 65, flush=True)
    print(f"{CROP.upper()} TRAINING COMPLETED", flush=True)
    print(f"Best epoch: {best_epoch}", flush=True)
    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy * 100:.2f}%",
        flush=True,
    )
    print(
        f"Total time: {(time.time() - overall_start) / 60:.2f} minutes",
        flush=True,
    )
    print(f"Model: {CHECKPOINT_FILE}", flush=True)
    print(f"Training history: {HISTORY_FILE}", flush=True)
    print(
        "The test split was not used for training or model selection.",
        flush=True,
    )


if __name__ == "__main__":
    main()
