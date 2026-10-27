
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
MODEL_FILE = PROJECT_DIR / "models" / "tomato_v2_best.pth"
REPORT_FILE = PROJECT_DIR / "models" / "tomato_v2_test_report.txt"
CONFUSION_FILE = PROJECT_DIR / "models" / "tomato_v2_test_confusion_matrix.csv"

BATCH_SIZE = 16
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
# MAIN EVALUATION
# ============================================================

def main():
    started = time.time()

    print("=" * 65, flush=True)
    print("TOMATO V2 - TEST SET EVALUATION", flush=True)
    print("=" * 65, flush=True)

    if not INDEX_FILE.is_file():
        raise FileNotFoundError(f"Dataset index missing: {INDEX_FILE}")

    if not MODEL_FILE.is_file():
        raise FileNotFoundError(f"Model checkpoint missing: {MODEL_FILE}")

    print("Loading model checkpoint...", flush=True)

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=DEVICE,
        weights_only=True,
    )

    classes = checkpoint["classes"]
    label_to_id = checkpoint["label_to_id"]
    image_size = checkpoint["image_size"]

    print(f"Architecture: {checkpoint['architecture']}", flush=True)
    print(f"Image size: {image_size}", flush=True)
    print(f"Classes: {len(classes)}", flush=True)
    print(
        f"Best validation accuracy: "
        f"{checkpoint['validation_accuracy'] * 100:.2f}%",
        flush=True,
    )

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
        raise RuntimeError("No Tomato test images found.")

    unknown_labels = sorted({
        row["label"] for row in test_rows
        if row["label"] not in label_to_id
    })

    if unknown_labels:
        raise RuntimeError(
            f"Unknown test labels: {unknown_labels}"
        )

    test_rows.sort(key=lambda row: row["image_path"])

    print(f"Test images: {len(test_rows)}", flush=True)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = PlantTestDataset(
        test_rows,
        label_to_id,
        transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print("Building MobileNetV3-Small...", flush=True)

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        len(classes),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    targets = []
    predictions = []

    progress = tqdm(
        loader,
        desc="Testing V2",
        unit="batch",
        dynamic_ncols=True,
    )

    with torch.inference_mode():
        for images, labels in progress:
            images = images.to(DEVICE)

            outputs = model(images)
            predicted = outputs.argmax(dim=1)

            targets.extend(labels.tolist())
            predictions.extend(predicted.cpu().tolist())

            processed = len(targets)
            running_accuracy = accuracy_score(targets, predictions)

            progress.set_postfix(
                accuracy=f"{running_accuracy * 100:.2f}%",
                images=f"{processed}/{len(dataset)}",
            )

    accuracy = accuracy_score(targets, predictions)

    report = classification_report(
        targets,
        predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        digits=4,
        zero_division=0,
    )

    matrix = confusion_matrix(
        targets,
        predictions,
        labels=list(range(len(classes))),
    )

    correct = sum(
        actual == predicted
        for actual, predicted in zip(targets, predictions)
    )

    print("\n" + "=" * 65, flush=True)
    print("V2 TEST RESULTS", flush=True)
    print("=" * 65, flush=True)
    print(f"Accuracy: {accuracy * 100:.2f}%", flush=True)
    print(f"Correct: {correct}", flush=True)
    print(f"Incorrect: {len(targets) - correct}", flush=True)
    print(f"Total test images: {len(targets)}", flush=True)

    print("\nClassification report:", flush=True)
    print(report, flush=True)

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        file.write("TOMATO V2 TEST RESULTS\n")
        file.write("=" * 60 + "\n")
        file.write(f"Checkpoint: {MODEL_FILE}\n")
        file.write(f"Test images: {len(targets)}\n")
        file.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
        file.write(report)
        file.write(
            f"\nElapsed time: {time.time() - started:.1f} seconds\n"
        )

    with CONFUSION_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["Actual / Predicted"] + classes)

        for index, matrix_row in enumerate(matrix):
            writer.writerow(
                [classes[index]] + matrix_row.tolist()
            )

    print(f"\nReport saved: {REPORT_FILE}", flush=True)
    print(f"Confusion matrix saved: {CONFUSION_FILE}", flush=True)
    print(
        f"Elapsed time: {time.time() - started:.1f} seconds",
        flush=True,
    )


if __name__ == "__main__":
    main()
