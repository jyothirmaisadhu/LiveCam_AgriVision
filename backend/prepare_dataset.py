
from pathlib import Path
import csv
import random

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "datasets" / "prepared"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PADDY_DIR = Path(
    r"C:\Users\hp\Downloads\paddy_disease_classification"
)
PADDY_IMAGES = PADDY_DIR / "train_images"
PADDY_CSV = PADDY_DIR / "train.csv"

PLANTVILLAGE_DIR = Path(
    r"C:\Users\hp\Downloads\Plant_leave_diseases_dataset_without_augmentation"
)
COTTON_DIR = Path(r"C:\Users\hp\Downloads\Cotton_Original_Dataset")
CHILLI_DIR = Path(r"C:\Users\hp\Downloads\Chili Leaf Disease Original Dataset")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RANDOM_SEED = 42

records = []
errors = []

# 1. Read Paddy labels from the CSV and verify folder labels.
with PADDY_CSV.open("r", newline="", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)
    print("Paddy CSV columns:", reader.fieldnames)

    for row in reader:
        filename = row.get("image_id")
        label = row.get("label")

        if not filename or not label:
            errors.append(f"Missing Paddy filename/label: {row}")
            continue

        image_path = PADDY_IMAGES / label / filename

        if not image_path.is_file():
            errors.append(f"Paddy image missing or folder mismatch: {image_path}")
            continue

        records.append({
            "crop": "paddy",
            "label": label,
            "image_path": str(image_path.resolve()),
        })

# 2. Read category-folder datasets.
def add_folder_dataset(crop, root, category_function=None):
    if not root.is_dir():
        errors.append(f"Dataset folder not found: {root}")
        return

    for category_dir in sorted(root.iterdir()):
        if not category_dir.is_dir():
            continue

        label = (
            category_function(category_dir.name)
            if category_function
            else category_dir.name
        )

        for image_path in sorted(category_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append({
                    "crop": crop,
                    "label": label,
                    "image_path": str(image_path.resolve()),
                })

# PlantVillage folder names begin with Tomato___ or Corn___.
add_folder_dataset(
    "tomato",
    PLANTVILLAGE_DIR,
    lambda name: name.removeprefix("Tomato___")
    if name.startswith("Tomato___") else None
)
add_folder_dataset(
    "corn",
    PLANTVILLAGE_DIR,
    lambda name: name.removeprefix("Corn___")
    if name.startswith("Corn___") else None
)

# Skip folders belonging to the other crop.
records = [
    row for row in records
    if row["crop"] == "paddy"
    or (row["crop"] == "tomato" and row["label"] is not None)
    or (row["crop"] == "corn" and row["label"] is not None)
]

add_folder_dataset("cotton", COTTON_DIR)
add_folder_dataset("chilli", CHILLI_DIR)

# 3. Remove duplicate paths, then split each crop/category group.
unique_records = {}
for row in records:
    unique_records[row["image_path"]] = row
records = list(unique_records.values())

random.seed(RANDOM_SEED)
groups = {}
for row in records:
    groups.setdefault((row["crop"], row["label"]), []).append(row)

final_records = []
for (crop, label), group in sorted(groups.items()):
    random.shuffle(group)
    n = len(group)

    if n < 3:
        errors.append(f"Too few images to split: {crop}/{label} ({n})")
        for row in group:
            final_records.append({**row, "split": "train"})
        continue

    n_train = max(1, int(n * 0.70))
    n_val = max(1, int(n * 0.15))

    if n_train + n_val >= n:
        n_train = n - 2
        n_val = 1

    for i, row in enumerate(group):
        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "validation"
        else:
            split = "test"

        final_records.append({**row, "split": split})

# 4. Save the index.
output_csv = OUTPUT_DIR / "dataset_index.csv"
columns = ["crop", "label", "image_path", "split"]

with output_csv.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    writer.writerows(final_records)

print("\nDataset preparation summary")
print("Total indexed images:", len(final_records))
print("Index saved to:", output_csv)
print("Errors found:", len(errors))

for split in ["train", "validation", "test"]:
    count = sum(row["split"] == split for row in final_records)
    print(f"{split}: {count}")

if errors:
    error_file = OUTPUT_DIR / "dataset_errors.txt"
    error_file.write_text("\n".join(errors), encoding="utf-8")
    print("Review errors in:", error_file)
