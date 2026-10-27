
from pathlib import Path
import argparse
import csv
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


PROJECT_DIR = Path(__file__).resolve().parent.parent
INDEX_FILE = PROJECT_DIR / "datasets" / "prepared" / "dataset_index.csv"
MODEL_DIR = PROJECT_DIR / "models"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--crop",
    required=True,
    choices=["tomato", "corn", "paddy", "chilli", "cotton"],
)
args = parser.parse_args()

CROP = args.crop
MODEL_FILE = MODEL_DIR / (
    "tomato_v2_best.pth" if CROP == "tomato"
    else f"{CROP}_best.pth"
)
REPORT_FILE = MODEL_DIR / f"{CROP}_test_report.txt"
MATRIX_FILE = MODEL_DIR / f"{CROP}_test_confusion_matrix.csv"

DEVICE = torch.device("cpu")
BATCH_SIZE = 16


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


def main():
    started = time.time()

    print("=" * 60, flush=True)
    print(f"{CROP.upper()} UNTOUCHED TEST EVALUATION", flush=True)
    print("=" * 60, flush=True)

    if not INDEX_FILE.is_file():
        raise FileNotFoundError(f"Dataset index missing: {INDEX_FILE}")
    if not MODEL_FILE.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {MODEL_FILE}")

    checkpoint = torch.load(
        MODEL_FILE, map_location=DEVICE, weights_only=True
    )
    classes = checkpoint["classes"]
    label_to_id = checkpoint["label_to_id"]
    image_size = checkpoint["image_size"]

    with INDEX_FILE.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        index_rows = list(csv.DictReader(file))

    test_rows = [
        row for row in index_rows
        if row["crop"].strip().lower() == CROP
        and row["split"].strip().lower() == "test"
    ]
    test_rows.sort(key=lambda row: row["image_path"])

    if not test_rows:
        raise RuntimeError(f"No test images found for {CROP}.")

    unknown = sorted({
        row["label"] for row in test_rows
        if row["label"] not in label_to_id
    })
    if unknown:
        raise RuntimeError(f"Unknown test labels: {unknown}")

    print(f"Checkpoint: {MODEL_FILE}", flush=True)
    print(f"Best validation accuracy: "
          f"{checkpoint['validation_accuracy'] * 100:.2f}%", flush=True)
    print(f"Test images: {len(test_rows)}", flush=True)
    print(f"Classes: {len(classes)}", flush=True)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = PlantTestDataset(test_rows, label_to_id, transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features, len(classes)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    targets = []
    predictions = []

    progress = tqdm(
        loader, desc=f"Testing {CROP}", unit="batch", dynamic_ncols=True
    )

    with torch.inference_mode():
        for images, labels in progress:
            outputs = model(images.to(DEVICE))
            predicted = outputs.argmax(dim=1).cpu().tolist()

            targets.extend(labels.tolist())
            predictions.extend(predicted)

            processed = len(targets)
            running_accuracy = accuracy_score(targets, predictions)
            progress.set_postfix(
                accuracy=f"{running_accuracy * 100:.2f}%",
                images=f"{processed}/{len(dataset)}",
            )

    accuracy = accuracy_score(targets, predictions)
    correct = sum(a == p for a, p in zip(targets, predictions))

    report = classification_report(
        targets,
        predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        digits=4,
        zero_division=0,
    )

    matrix = confusion_matrix(
        targets, predictions, labels=list(range(len(classes)))
    )

    print("\n" + "=" * 60, flush=True)
    print(f"{CROP.upper()} TEST RESULTS", flush=True)
    print(f"Accuracy: {accuracy * 100:.2f}%", flush=True)
    print(f"Correct: {correct}", flush=True)
    print(f"Incorrect: {len(targets) - correct}", flush=True)
    print(f"Test images: {len(targets)}", flush=True)
    print("\nClassification report:\n", report, flush=True)

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        file.write(f"{CROP.upper()} TEST RESULTS\n")
        file.write(f"Checkpoint: {MODEL_FILE}\n")
        file.write(f"Test images: {len(targets)}\n")
        file.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
        file.write(report)
        file.write(f"\nElapsed seconds: {time.time() - started:.1f}\n")

    with MATRIX_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Actual / Predicted"] + classes)
        for i, row in enumerate(matrix):
            writer.writerow([classes[i]] + row.tolist())

    print(f"\nReport saved: {REPORT_FILE}", flush=True)
    print(f"Confusion matrix saved: {MATRIX_FILE}", flush=True)
    print(f"Elapsed: {time.time() - started:.1f} seconds", flush=True)


if __name__ == "__main__":
    main()
