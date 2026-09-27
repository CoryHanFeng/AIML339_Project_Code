"""
Checks the per-class breakdown of annotated (CSV-covered) images,
so you know the real bottleneck if filtering to annotated-only images
(rather than trusting the raw 76k total, which hides class imbalance).
"""
import os
from cory_data_wrapper import ImageFolder


def annotated_counts_per_class(root_dir, csv_ids):
    """
    root_dir: your in9L_final folder
    csv_ids: set of image_ids that DO have a CSV bounding box
    Returns: {class_name: count_with_annotation}
    """
    folder = ImageFolder(root_dir)
    counts = {c: 0 for c in folder.classes}
    idx_to_class = {i: c for c, i in folder.class_to_idx.items()}

    for path, label_idx in folder.samples:
        image_id = os.path.splitext(os.path.basename(path))[0]
        if image_id in csv_ids:
            counts[idx_to_class[label_idx]] += 1

    return counts


if __name__ == "__main__":
    # Self-test with fake data
    os.makedirs("fake_class_check/Fish", exist_ok=True)
    os.makedirs("fake_class_check/Dog", exist_ok=True)
    from PIL import Image
    for i in range(5):
        Image.new("RGB", (20, 20)).save(f"fake_class_check/Fish/n01440764_{i}.JPEG")
    for i in range(10):
        Image.new("RGB", (20, 20)).save(f"fake_class_check/Dog/n02085620_{i}.JPEG")

    # Only some have "annotations" (simulate 3/5 Fish, 2/10 Dog annotated)
    fake_csv_ids = {"n01440764_0", "n01440764_1", "n01440764_2", "n02085620_0", "n02085620_1"}

    counts = annotated_counts_per_class("fake_class_check", fake_csv_ids)
    print(f"Result: {counts}")
    assert counts == {"Dog": 2, "Fish": 3}, f"FAILED: {counts}"
    print("PASSED: correctly counts annotated images per class.")
