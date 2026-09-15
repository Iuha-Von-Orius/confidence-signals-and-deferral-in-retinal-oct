"""Count Kermany images by split and class."""

from paths import DATASET

from collections import Counter

DATA_ROOT = (DATASET / "Paul_Mooney/OCT2017")

SPLITS = ["train", "val", "test"]
CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]


def count_split(split_dir):
    """Return image counts by class, using None for a missing class directory."""
    counts = {}
    for cls in CLASSES:
        cls_dir = split_dir / cls
        if not cls_dir.exists():
            counts[cls] = None
            continue
        images = [p for p in cls_dir.iterdir()
                  if p.suffix.lower() in {".jpeg", ".jpg", ".png"}]
        counts[cls] = len(images)
    return counts


def main():
    if not DATA_ROOT.exists():
        print(f"Dataset path not found: {DATA_ROOT.resolve()}")
        print("Check OCT_DATA_ROOT and the dataset directory layout.")
        return

    print(f"Dataset path: {DATA_ROOT.resolve()}\n")

    grand_total = 0
    all_suffixes = Counter()

    for split in SPLITS:
        split_dir = DATA_ROOT / split
        if not split_dir.exists():
            print(f"[{split}] directory missing; skipped\n")
            continue

        counts = count_split(split_dir)
        total = sum(v for v in counts.values() if v is not None)
        grand_total += total

        print(f"[{split}]")
        for cls in CLASSES:
            n = counts[cls]
            if n is None:
                print(f"  {cls:<8} directory missing")
            else:
                pct = n / total * 100 if total else 0
                print(f"  {cls:<8} {n:>7,}  ({pct:5.1f}%)")
        print(f"  {'TOTAL':<8} {total:>7,}\n")

        for cls in CLASSES:
            cls_dir = split_dir / cls
            if cls_dir.exists():
                for p in cls_dir.iterdir():
                    if p.is_file():
                        all_suffixes[p.suffix.lower()] += 1


    print(f"Total: {grand_total:,} images")
    print(f"File extensions: {dict(all_suffixes)}")


if __name__ == "__main__":
    main()


