
from pathlib import Path
import csv
import time
from collections import Counter, defaultdict
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
INDEX_FILE = PROJECT_DIR / "datasets" / "prepared" / "dataset_index.csv"
REPORT_FILE = PROJECT_DIR / "datasets" / "prepared" / "validation_report.txt"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
UPDATE_EVERY = 100  # Print progress every 100 images

with INDEX_FILE.open("r", newline="", encoding="utf-8-sig") as file:
    rows = list(csv.DictReader(file))

if not rows:
    raise SystemExit("ERROR: Dataset index is empty.")

total = len(rows)
missing = []
unreadable = []
unsupported = []
split_counts = Counter()
category_splits = defaultdict(Counter)
seen_paths = set()

start = time.time()

print(f"Starting validation of {total:,} images...", flush=True)

for i, row in enumerate(rows, start=1):
    path = Path(row["image_path"])
    crop = row["crop"]
    label = row["label"]
    split = row["split"]

    split_counts[split] += 1
    category_splits[(crop, label)][split] += 1

    normalized_path = str(path.resolve()).lower()
    if normalized_path in seen_paths:
        print(f"\nWARNING: Duplicate path: {path}", flush=True)
    seen_paths.add(normalized_path)

    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        unsupported.append(str(path))
    elif not path.is_file():
        missing.append(str(path))
    else:
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as error:
            unreadable.append(f"{path} | {error}")

    if i % UPDATE_EVERY == 0 or i == total:
        elapsed = time.time() - start
        percent = i / total * 100
        rate = i / elapsed if elapsed else 0
        remaining = (total - i) / rate if rate else 0

        bar_length = 30
        filled = int(bar_length * i / total)
        bar = "#" * filled + "-" * (bar_length - filled)

        print(
            f"\r[{bar}] {percent:6.2f}% "
            f"| {i:,}/{total:,} images "
            f"| elapsed {elapsed:.0f}s "
            f"| ETA {remaining:.0f}s",
            end="\n" if i == total else "",
            flush=True,
        )

missing_splits = []
for (crop, label), counts in sorted(category_splits.items()):
    absent = {"train", "validation", "test"} - set(counts)
    if absent:
        missing_splits.append(
            f"{crop}/{label}: missing {', '.join(sorted(absent))}"
        )

with REPORT_FILE.open("w", encoding="utf-8") as report:
    report.write(f"Images indexed: {total}\n")
    report.write(f"Missing files: {len(missing)}\n")
    report.write(f"Unreadable images: {len(unreadable)}\n")
    report.write(f"Unsupported extensions: {len(unsupported)}\n")
    report.write(f"Categories missing splits: {len(missing_splits)}\n")

    for title, items in [
        ("MISSING FILES", missing),
        ("UNREADABLE IMAGES", unreadable),
        ("UNSUPPORTED FILES", unsupported),
        ("MISSING CATEGORY SPLITS", missing_splits),
    ]:
        report.write(f"\n{title}\n")
        report.write("\n".join(items) + "\n")

print("\n\n========== VALIDATION SUMMARY ==========")
print(f"Images checked: {total:,}")
print(f"Missing files: {len(missing)}")
print(f"Unreadable images: {len(unreadable)}")
print(f"Unsupported extensions: {len(unsupported)}")
print(f"Categories missing splits: {len(missing_splits)}")
print(f"Split counts: {dict(split_counts)}")
print(f"Elapsed time: {time.time() - start:.1f} seconds")
print(f"Report saved to: {REPORT_FILE}")
print("Validation finished.", flush=True)
