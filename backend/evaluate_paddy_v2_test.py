
import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(ROOT, "datasets", "prepared", "dataset_index.csv")
CHECKPOINT_PATH = os.path.join(ROOT, "models", "paddy_v2_best.pth")
REPORT_PATH = os.path.join(ROOT, "models", "paddy_v2_test_report.txt")
MATRIX_PATH = os.path.join(
    ROOT, "models", "paddy_v2_test_confusion_matrix.csv"
)

BATCH_SIZE = 16
NUM_WORKERS = 0
DEVICE = torch.device("cpu")


# --------------------------------------------------
# 2. DATASET
# --------------------------------------------------
class PaddyTestDataset(Dataset):
    def __init__(self, dataframe, class_to_idx, transform):
        self.dataframe = dataframe.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_path = str(row["image_path"])
        label = self.class_to_idx[row["label"]]

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                if self.transform:
                    image = self.transform(image)
        except Exception as exc:
            raise RuntimeError(
                f"Could not read test image: {image_path}"
            ) from exc

        return image, label


# --------------------------------------------------
# 3. MAIN EVALUATION
# --------------------------------------------------
def main():
    start_time = time.time()

    print("=" * 60)
    print("PADDY V2 UNTOUCHED TEST EVALUATION")
    print("=" * 60)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"Dataset index not found: {INDEX_PATH}"
        )

    # Load checkpoint explicitly from V2.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    classes = checkpoint["classes"]
    image_size = checkpoint.get("image_size", 160)
    class_to_idx = {name: i for i, name in enumerate(classes)}

    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Best validation accuracy: "
          f"{checkpoint.get('best_val_accuracy', float('nan')):.2f}%")
    print(f"Image size: {image_size} x {image_size}")
    print(f"Classes: {len(classes)}")

    # Load only the existing test split for paddy.
    dataframe = pd.read_csv(INDEX_PATH)
    test_df = dataframe[
        (dataframe["crop"].astype(str).str.lower() == "paddy")
        & (dataframe["split"].astype(str).str.lower() == "test")
    ].copy()

    if test_df.empty:
        raise ValueError(
            "No Paddy test images found in dataset_index.csv."
        )

    unknown_labels = sorted(set(test_df["label"]) - set(classes))
    if unknown_labels:
        raise ValueError(
            f"Test labels missing from checkpoint classes: {unknown_labels}"
        )

    print(f"Test images: {len(test_df)}")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = PaddyTestDataset(test_df, class_to_idx, transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # Rebuild the same MobileNetV3-Small architecture.
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(classes))

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    all_labels = []
    all_predictions = []
    correct = 0
    total = 0

    with torch.inference_mode():
        progress = tqdm(
            loader,
            desc="Testing Paddy V2",
            unit="batch"
        )

        for images, labels in progress:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )

            accuracy = 100.0 * correct / total
            progress.set_postfix(
                accuracy=f"{accuracy:.2f}%",
                images=f"{total}/{len(dataset)}"
            )

    accuracy = 100.0 * correct / total

    # Keep class order consistent with the checkpoint.
    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        digits=4,
        zero_division=0
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(classes)))
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=classes,
        columns=classes
    )
    matrix_df.index.name = "Actual / Predicted"
    matrix_df.to_csv(MATRIX_PATH)

    elapsed = time.time() - start_time

    report_text = (
        "PADDY V2 UNTOUCHED TEST EVALUATION\n"
        + "=" * 60 + "\n"
        + f"Checkpoint: {CHECKPOINT_PATH}\n"
        + f"Best validation accuracy: "
          f"{checkpoint.get('best_val_accuracy', float('nan')):.2f}%\n"
        + f"Test images: {total}\n"
        + f"Correct: {correct}\n"
        + f"Incorrect: {total - correct}\n"
        + f"Test accuracy: {accuracy:.2f}%\n\n"
        + "Classification report:\n"
        + report + "\n"
        + f"Elapsed: {elapsed:.1f} seconds\n"
    )

    print("\n" + "=" * 60)
    print("PADDY V2 TEST RESULTS")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Correct: {correct}")
    print(f"Incorrect: {total - correct}")
    print(f"Test images: {total}")
    print("\nClassification report:")
    print(report)
    print(f"Report saved: {REPORT_PATH}")
    print(f"Confusion matrix saved: {MATRIX_PATH}")
    print(f"Elapsed: {elapsed:.1f} seconds")

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write(report_text)


if __name__ == "__main__":
    main()
