
from pathlib import Path
import csv
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
INDEX_FILE = PROJECT_DIR / "datasets" / "prepared" / "dataset_index.csv"
MODEL_FILE = PROJECT_DIR / "models" / "tomato_finetuned_best.pth"
REPORT_FILE = PROJECT_DIR / "models" / "tomato_test_report.txt"
CONFUSION_FILE = PROJECT_DIR / "models" / "tomato_test_confusion_matrix.csv"

IMAGE_SIZE = 160
BATCH_SIZE = 8
NUM_WORKERS = 0
DEVICE = torch.device("cpu")


# ============================================================
# DATASET
# ============================================================

class PlantTestDataset(Dataset):
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

        return image, self.label_to_id[row["label"]]


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_rows(classes):
    if not INDEX_FILE.is_file():
        raise FileNotFoundError(f"Dataset index not found: {INDEX_FILE}")

    with INDEX_FILE.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        rows = list(csv.DictReader(file))

    test_rows = [
        row for row in rows
        if row["crop"].strip().lower() == "tomato"
        and row["split"].strip().lower() == "test"
    ]

    if not test_rows:
        raise RuntimeError(
            "No Tomato test records found in dataset_index.csv."
        )

    # Keep only labels supported by the saved model.
    unknown_labels = sorted({
        row["label"] for row in test_rows
        if row["label"] not in classes
    })

    if unknown_labels:
        raise RuntimeError(
            f"Test labels not present in checkpoint: {unknown_labels}"
        )

    return sorted(
        test_rows,
        key=lambda row: row["image_path"]
    )


# ============================================================
# EVALUATION
# ============================================================

def main():
    started = time.time()

    print("=" * 65, flush=True)
    print("TOMATO MODEL - UNTOUCHED TEST EVALUATION", flush=True)
    print("=" * 65, flush=True)

    if not MODEL_FILE.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_FILE}"
        )

    print("Loading saved model checkpoint...", flush=True)

    # This checkpoint was created by our own training script.
    checkpoint = torch.load(
        MODEL_FILE,
        map_location=DEVICE,
        weights_only=True,
    )

    classes = checkpoint["classes"]
    label_to_id = checkpoint["label_to_id"]

    print(f"Architecture: {checkpoint['architecture']}", flush=True)
    print(f"Number of classes: {len(classes)}", flush=True)
    print(
        f"Best validation accuracy recorded: "
        f"{checkpoint['validation_accuracy'] * 100:.2f}%",
        flush=True,
    )

    rows = load_test_rows(classes)

    print(f"Untouched test images: {len(rows)}", flush=True)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = PlantTestDataset(rows, label_to_id, transform)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print("Creating MobileNetV3-Small model...", flush=True)

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        len(classes),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    all_targets = []
    all_predictions = []

    progress = tqdm(
        loader,
        desc="Testing",
        unit="batch",
        dynamic_ncols=True,
    )

    with torch.inference_mode():
        for images, labels in progress:
            images = images.to(DEVICE)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_targets.extend(labels.tolist())
            all_predictions.extend(predictions.cpu().tolist())

            processed = len(all_targets)
            accuracy_so_far = accuracy_score(
                all_targets, all_predictions
            )

            progress.set_postfix(
                accuracy=f"{accuracy_so_far * 100:.1f}%",
                images=f"{processed}/{len(dataset)}",
            )

    accuracy = accuracy_score(all_targets, all_predictions)

    report = classification_report(
        all_targets,
        all_predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0,
        digits=4,
    )

    matrix = confusion_matrix(
        all_targets,
        all_predictions,
        labels=list(range(len(classes))),
    )

    print("\n" + "=" * 65, flush=True)
    print("TEST RESULTS", flush=True)
    print("=" * 65, flush=True)
    print(f"Test accuracy: {accuracy * 100:.2f}%", flush=True)
    print(f"Correct predictions: {sum(t == p for t, p in zip(all_targets, all_predictions))}", flush=True)
    print(f"Incorrect predictions: {sum(t != p for t, p in zip(all_targets, all_predictions))}", flush=True)
    print(f"Test images: {len(all_targets)}", flush=True)

    print("\nClassification report:", flush=True)
    print(report, flush=True)

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        file.write("TOMATO MODEL - UNTOUCHED TEST RESULTS\n")
        file.write("=" * 60 + "\n")
        file.write(f"Checkpoint: {MODEL_FILE}\n")
        file.write(f"Test images: {len(all_targets)}\n")
        file.write(f"Test accuracy: {accuracy * 100:.2f}%\n\n")
        file.write(report)
        file.write(f"\nElapsed time: {(time.time() - started):.1f} seconds\n")

    with CONFUSION_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["Actual / Predicted"] + classes)

        for index, row in enumerate(matrix):
            writer.writerow([classes[index]] + row.tolist())

    print(f"\nSaved report: {REPORT_FILE}", flush=True)
    print(f"Saved confusion matrix: {CONFUSION_FILE}", flush=True)
    print(
        f"Total elapsed time: {time.time() - started:.1f} seconds",
        flush=True,
    )
    print(
        "\nThis test split was not used to train the model.",
        flush=True,
    )


if __name__ == "__main__":
    main()
