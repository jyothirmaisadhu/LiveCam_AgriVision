
from pathlib import Path
import csv
import time
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_DIR = Path(__file__).resolve().parent.parent
INDEX_FILE = PROJECT_DIR / "datasets" / "prepared" / "dataset_index.csv"
CHECKPOINT = PROJECT_DIR / "models" / "tomato_pilot_best.pth"
REPORT_FILE = PROJECT_DIR / "models" / "tomato_validation_report.txt"

IMAGE_SIZE = 160
BATCH_SIZE = 8
DEVICE = torch.device("cpu")


class PlantDataset(Dataset):
    def __init__(self, rows, label_to_id, transform):
        self.rows = rows
        self.label_to_id = label_to_id
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        return image, self.label_to_id[row["label"]]


def main():
    start = time.time()

    print("[1/5] Loading checkpoint and dataset index...", flush=True)
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    checkpoint = torch.load(
        CHECKPOINT, map_location=DEVICE, weights_only=False
    )
    classes = checkpoint["classes"]
    label_to_id = checkpoint["label_to_id"]

    with INDEX_FILE.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    # Use the same class-balanced validation selection as the pilot.
    rng = __import__("random").Random(42)
    validation_rows = [
        row for row in rows
        if row["crop"] == "tomato" and row["split"] == "validation"
    ]

    by_class = {label: [] for label in classes}
    for row in validation_rows:
        if row["label"] in by_class:
            by_class[row["label"]].append(row)

    selected_rows = []
    for label in classes:
        rng.shuffle(by_class[label])
        selected_rows.extend(by_class[label][:20])

    print(f"Selected validation images: {len(selected_rows)}", flush=True)

    print("[2/5] Preparing images...", flush=True)
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = PlantDataset(selected_rows, label_to_id, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("[3/5] Loading model...", flush=True)
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = torch.nn.Linear(
        model.classifier[3].in_features, len(classes)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print("[4/5] Evaluating with progress...", flush=True)
    all_labels, all_predictions = [], []

    with torch.inference_mode():
        for images, labels in tqdm(
            loader, desc="Evaluation", unit="batch"
        ):
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            all_labels.extend(labels.tolist())
            all_predictions.extend(predictions.tolist())

    print("[5/5] Creating metrics...", flush=True)
    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0,
        digits=3,
    )
    matrix = confusion_matrix(
        all_labels, all_predictions, labels=list(range(len(classes)))
    )

    accuracy = sum(
        actual == predicted
        for actual, predicted in zip(all_labels, all_predictions)
    ) / len(all_labels)

    output = []
    output.append(f"Validation images: {len(all_labels)}")
    output.append(f"Accuracy: {accuracy * 100:.2f}%")
    output.append("\nPER-CLASS METRICS\n")
    output.append(report)
    output.append("\nCONFUSION MATRIX (rows=true, columns=predicted)\n")
    output.append("Class order: " + ", ".join(classes))
    output.append(str(matrix))

    result = "\n".join(output)
    print("\n" + result)

    REPORT_FILE.write_text(result, encoding="utf-8")
    print(f"\nReport saved to: {REPORT_FILE}")
    print(f"Elapsed time: {(time.time() - start):.1f}s", flush=True)


if __name__ == "__main__":
    main()
