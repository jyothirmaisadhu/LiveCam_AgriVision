
from pathlib import Path
import csv
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image, ImageFile
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

INDEX_FILE = (
    PROJECT_DIR / "datasets" / "prepared" / "dataset_index.csv"
)

MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_FILE = MODEL_DIR / "tomato_finetuned_best.pth"
VALIDATION_LIST_FILE = MODEL_DIR / "tomato_finetuned_validation.csv"

SEED = 42
IMAGE_SIZE = 160
BATCH_SIZE = 8
EPOCHS = 5

MAX_TRAIN_PER_CLASS = 200
MAX_VAL_PER_CLASS = 50

NUM_WORKERS = 0
LEARNING_RATE = 0.0001

DEVICE = torch.device("cpu")

ImageFile.LOAD_TRUNCATED_IMAGES = False

random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


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
        image_path = Path(row["image_path"])

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            image = self.transform(img)

        label_id = self.label_to_id[row["label"]]

        return image, label_id


# ============================================================
# LOAD DATA AND CREATE REPRODUCIBLE SUBSETS
# ============================================================

def load_index():
    if not INDEX_FILE.is_file():
        raise FileNotFoundError(
            f"Dataset index not found: {INDEX_FILE}"
        )

    print("Reading dataset index...", flush=True)

    with INDEX_FILE.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        rows = list(csv.DictReader(file))

    tomato_rows = [
        row for row in rows
        if row["crop"].strip().lower() == "tomato"
        and row["split"] in {"train", "validation"}
    ]

    if not tomato_rows:
        raise RuntimeError(
            "No Tomato records found in dataset_index.csv."
        )

    classes = sorted({
        row["label"] for row in tomato_rows
    })

    label_to_id = {
        label: index
        for index, label in enumerate(classes)
    }

    train_by_class = {label: [] for label in classes}
    val_by_class = {label: [] for label in classes}

    for row in tomato_rows:
        if row["split"] == "train":
            train_by_class[row["label"]].append(row)
        elif row["split"] == "validation":
            val_by_class[row["label"]].append(row)

    train_rows = []
    val_rows = []

    for label in classes:
        # Sort before shuffling so selection is reproducible.
        train_items = sorted(
            train_by_class[label],
            key=lambda row: row["image_path"]
        )

        val_items = sorted(
            val_by_class[label],
            key=lambda row: row["image_path"]
        )

        train_rng = random.Random(f"{SEED}_train_{label}")
        val_rng = random.Random(f"{SEED}_validation_{label}")

        train_rng.shuffle(train_items)
        val_rng.shuffle(val_items)

        selected_train = train_items[:MAX_TRAIN_PER_CLASS]
        selected_val = val_items[:MAX_VAL_PER_CLASS]

        train_rows.extend(selected_train)
        val_rows.extend(selected_val)

        print(
            f"{label}: "
            f"train={len(selected_train)}, "
            f"validation={len(selected_val)}",
            flush=True,
        )

    # Use a separate shuffle for the overall row order.
    random.Random(SEED).shuffle(train_rows)
    random.Random(SEED + 1).shuffle(val_rows)

    return classes, label_to_id, train_rows, val_rows


def save_validation_list(val_rows):
    """Save exact validation paths for repeatable evaluation."""

    with VALIDATION_LIST_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image_path", "label"],
        )
        writer.writeheader()

        for row in val_rows:
            writer.writerow({
                "image_path": row["image_path"],
                "label": row["label"],
            })

    print(
        f"Saved fixed validation list:\n"
        f"{VALIDATION_LIST_FILE}",
        flush=True,
    )


# ============================================================
# VALIDATION
# ============================================================

def evaluate(model, loader, criterion, epoch, total_epochs):
    model.eval()

    loss_sum = 0.0
    correct = 0
    total = 0

    progress = tqdm(
        loader,
        desc=f"Validation {epoch}/{total_epochs}",
        unit="batch",
        dynamic_ncols=True,
    )

    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(
            progress, start=1
        ):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            loss_sum += loss.item() * batch_size
            correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()

            total += batch_size

            progress.set_postfix(
                loss=f"{loss.item():.3f}",
                accuracy=f"{100 * correct / total:.1f}%",
                processed=f"{total}/{len(loader.dataset)}",
            )

    if total == 0:
        raise RuntimeError("Validation dataset is empty.")

    return loss_sum / total, correct / total


# ============================================================
# TRAINING
# ============================================================

def main():
    overall_start = time.time()

    print("=" * 65, flush=True)
    print("TOMATO DISEASE CLASSIFICATION - FINE-TUNING", flush=True)
    print("=" * 65, flush=True)

    print(f"Device: {DEVICE}", flush=True)
    print(f"Image size: {IMAGE_SIZE} x {IMAGE_SIZE}", flush=True)
    print(f"Batch size: {BATCH_SIZE}", flush=True)
    print(f"Epochs: {EPOCHS}", flush=True)
    print(f"Learning rate: {LEARNING_RATE}", flush=True)

    classes, label_to_id, train_rows, val_rows = load_index()

    if len(classes) != 10:
        print(
            f"WARNING: Expected 10 tomato classes; found "
            f"{len(classes)}.",
            flush=True,
        )

    if not train_rows or not val_rows:
        raise RuntimeError(
            "Training or validation data is empty."
        )

    save_validation_list(val_rows)

    print(f"\nNumber of classes: {len(classes)}", flush=True)
    print(f"Training images: {len(train_rows)}", flush=True)
    print(f"Validation images: {len(val_rows)}", flush=True)

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
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

    print("\nLoading pretrained MobileNetV3-Small...", flush=True)

    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)

    # Freeze all pretrained feature layers first.
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    # Fine-tune the last two feature blocks.
    for parameter in model.features[-2:].parameters():
        parameter.requires_grad = True

    # Replace the final classifier for our tomato classes.
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(
        in_features, len(classes)
    )

    model = model.to(DEVICE)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Trainable parameters: {trainable_parameters:,}",
        flush=True,
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=LEARNING_RATE,
    )

    best_val_accuracy = -1.0
    best_epoch = 0
    epoch_durations = []

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model.train()

        loss_sum = 0.0
        correct = 0
        total = 0

        print(
            f"\n{'=' * 22} "
            f"EPOCH {epoch}/{EPOCHS} "
            f"{'=' * 22}",
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

            batch_size = labels.size(0)

            loss_sum += loss.item() * batch_size
            correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()

            total += batch_size

            progress.set_postfix(
                loss=f"{loss.item():.3f}",
                accuracy=f"{100 * correct / total:.1f}%",
                images=f"{total}/{len(train_dataset)}",
            )

        train_loss = loss_sum / total
        train_accuracy = correct / total

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            epoch,
            EPOCHS,
        )

        epoch_time = time.time() - epoch_start
        epoch_durations.append(epoch_time)

        average_epoch_time = sum(epoch_durations) / len(epoch_durations)
        remaining_epochs = EPOCHS - epoch
        eta_seconds = average_epoch_time * remaining_epochs

        print(
            f"\nEpoch {epoch}/{EPOCHS} results:\n"
            f"  Train loss:       {train_loss:.4f}\n"
            f"  Train accuracy:   {train_accuracy * 100:.2f}%\n"
            f"  Validation loss:  {val_loss:.4f}\n"
            f"  Validation acc.:  {val_accuracy * 100:.2f}%\n"
            f"  Epoch time:       {epoch_time / 60:.2f} min\n"
            f"  Overall progress: {epoch / EPOCHS * 100:.0f}%\n"
            f"  Estimated time left: {eta_seconds / 60:.2f} min",
            flush=True,
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "label_to_id": label_to_id,
                "image_size": IMAGE_SIZE,
                "architecture": "mobilenet_v3_small",
                "validation_accuracy": best_val_accuracy,
                "validation_loss": val_loss,
                "best_epoch": best_epoch,
                "seed": SEED,
                "train_image_count": len(train_rows),
                "validation_image_count": len(val_rows),
            }

            torch.save(checkpoint, CHECKPOINT_FILE)

            print(
                f"Saved best checkpoint at epoch {epoch}:\n"
                f"{CHECKPOINT_FILE}",
                flush=True,
            )

    total_minutes = (time.time() - overall_start) / 60

    print("\n" + "=" * 65, flush=True)
    print("TRAINING COMPLETED", flush=True)
    print("=" * 65, flush=True)
    print(f"Best epoch: {best_epoch}/{EPOCHS}", flush=True)
    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy * 100:.2f}%",
        flush=True,
    )
    print(f"Total time: {total_minutes:.2f} minutes", flush=True)
    print(f"Model checkpoint: {CHECKPOINT_FILE}", flush=True)
    print(f"Validation list: {VALIDATION_LIST_FILE}", flush=True)
    print(
        "Note: Validation accuracy is not final test accuracy.",
        flush=True,
    )


if __name__ == "__main__":
    main()
