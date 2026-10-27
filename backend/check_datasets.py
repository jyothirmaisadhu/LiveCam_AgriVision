
from pathlib import Path

# Original dataset locations on your laptop
DATASETS = {
    "Paddy": Path(
        r"C:\Users\hp\Downloads\paddy_disease_classification\train_images"
    ),
    "Tomato and Corn": Path(
        r"C:\Users\hp\Downloads\Plant_leave_diseases_dataset_without_augmentation"
    ),
    "Cotton": Path(
        r"C:\Users\hp\Downloads\Cotton_Original_Dataset"
    ),
    "Chilli": Path(
        r"C:\Users\hp\Downloads\Chili Leaf Disease Original Dataset"
    ),
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for dataset_name, dataset_path in DATASETS.items():
    print(f"\n{'=' * 55}")
    print(f"Dataset: {dataset_name}")
    print(f"Path: {dataset_path}")

    if not dataset_path.exists():
        print("ERROR: Folder not found. Check the path.")
        continue

    total_images = 0

    for category_dir in sorted(dataset_path.iterdir()):
        if not category_dir.is_dir():
            continue

        count = sum(
            1 for file in category_dir.rglob("*")
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        print(f"  {category_dir.name}: {count} images")
        total_images += count

    print(f"Total images found: {total_images}")
