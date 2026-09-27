"""
Creates a NEW, filtered+rebalanced copy of the training set, containing
only images that have a real bounding box in LOC_train_solution.csv,
rebalanced so every class has the same count (the smallest annotated
class). The original in9L_final folder is never modified - this only
ever COPIES files into a new output directory.

Since this is a filtered SUBSET of the already leakage-checked
in9L_final, no leakage re-check is needed - removing images cannot
introduce leakage that wasn't already ruled out.
"""

import os
import random
import shutil
from collections import defaultdict

from cory_data_wrapper import load_bbox_csv, ImageFolder


def build_annotated_only_copy(source_root, csv_path, output_root, seed=0):
    print("Loading bounding box CSV...")
    bboxes = load_bbox_csv(csv_path)
    csv_ids = set(bboxes.keys())
    print(f"CSV has {len(csv_ids)} annotated images")

    print("Scanning source folder...")
    folder = ImageFolder(source_root)
    idx_to_class = {i: c for c, i in folder.class_to_idx.items()}

    annotated_by_class = defaultdict(list)
    for path, label_idx in folder.samples:
        image_id = os.path.splitext(os.path.basename(path))[0]
        if image_id in csv_ids:
            annotated_by_class[idx_to_class[label_idx]].append(path)

    counts = {c: len(paths) for c, paths in annotated_by_class.items()}
    print("\nAnnotated counts per class:")
    for c, n in sorted(counts.items(), key=lambda x: x[1]):
        print(f"  {c}: {n}")

    target_count = min(counts.values())
    print(f"\nRebalancing every class to {target_count} (smallest annotated class)")

    rng = random.Random(seed)
    os.makedirs(output_root, exist_ok=True)
    total_copied = 0

    for class_name, paths in annotated_by_class.items():
        class_out_dir = os.path.join(output_root, class_name)
        os.makedirs(class_out_dir, exist_ok=True)

        shuffled = paths[:]
        rng.shuffle(shuffled)
        selected = shuffled[:target_count]

        for src_path in selected:
            dst_path = os.path.join(class_out_dir, os.path.basename(src_path))
            shutil.copy2(src_path, dst_path)
            total_copied += 1

    print(f"\nDone. Copied {total_copied} images to {output_root}")
    print(f"({target_count} per class x {len(annotated_by_class)} classes)")
    return output_root, target_count, total_copied


if __name__ == "__main__":
    # Self-test with fake data
    import csv
    from PIL import Image

    os.makedirs("fake_source/Fish", exist_ok=True)
    os.makedirs("fake_source/Dog", exist_ok=True)

    # Fish: 5 images, 3 annotated. Dog: 5 images, 5 annotated.
    # Expect: annotated_by_class = {Fish: 3, Dog: 5}, target_count = 3,
    # so both classes end up with 3 images each = 6 total copied.
    for i in range(5):
        Image.new("RGB", (10, 10)).save(f"fake_source/Fish/n01440764_{i}.JPEG")
        Image.new("RGB", (10, 10)).save(f"fake_source/Dog/n02085620_{i}.JPEG")

    with open("fake_bboxes.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ImageId", "PredictionString"])
        for i in range(3):
            writer.writerow([f"n01440764_{i}", f"n01440764 1 1 5 5"])
        for i in range(5):
            writer.writerow([f"n02085620_{i}", f"n02085620 1 1 5 5"])

    output_root, target_count, total_copied = build_annotated_only_copy(
        "fake_source", "fake_bboxes.csv", "fake_output", seed=0
    )

    assert target_count == 3, f"FAILED: expected target_count 3, got {target_count}"
    assert total_copied == 6, f"FAILED: expected 6 total copied, got {total_copied}"

    fish_out = os.listdir("fake_output/Fish")
    dog_out = os.listdir("fake_output/Dog")
    assert len(fish_out) == 3, f"FAILED: expected 3 Fish images, got {len(fish_out)}"
    assert len(dog_out) == 3, f"FAILED: expected 3 Dog images, got {len(dog_out)}"

    # Confirm the ORIGINAL source folder was untouched
    assert len(os.listdir("fake_source/Fish")) == 5, "FAILED: original source was modified!"
    assert len(os.listdir("fake_source/Dog")) == 5, "FAILED: original source was modified!"

    print("\nPASSED: correctly filtered to annotated-only, rebalanced to smallest class,")
    print("copied (not moved) the right count, and left the original source untouched.")
