"""
Standalone label check for peace of mind.

Confirms that fix_test_labels() correctly maps each numbered test
folder (00_dog, 01_bird, ...) to the SAME class index your training
data uses. Run this any time you want direct confirmation, without
touching your main notebooks.
"""

import pathlib
from cory_data_wrapper import (
    load_bbox_csv, IN9BoundingBoxDataset, ImageFolderEval, fix_test_labels
)

# --- Edit these to match your real paths ---
DATA_ROUTE = r"E:\AIML339\in9L_annotated_only"
BOUNDING_BOX_PATH = r"LOC_train_solution.csv"  # adjust to wherever it actually is
TEST_INSTANCES = pathlib.Path(r"E:\AIML339\backgrounds_challenge_data\bg_challenge")
IMAGE_SIZE = 224
# --------------------------------------------

# Rebuild the training class mapping, exactly as your notebooks do
bboxes = load_bbox_csv(BOUNDING_BOX_PATH)
base_train_dataset = IN9BoundingBoxDataset(DATA_ROUTE, bboxes, image_size=IMAGE_SIZE)
train_class_to_idx = base_train_dataset.class_to_idx

print("Training class_to_idx (the ground truth ordering):")
for cls, idx in sorted(train_class_to_idx.items(), key=lambda x: x[1]):
    print(f"  {idx}: {cls}")

idx_to_class = {v: k for k, v in train_class_to_idx.items()}

# Check every one of the four variants, not just one
all_ok = True
for variant in ["original", "no_fg", "mixed_same", "mixed_rand"]:
    print(f"\n{'='*60}\nChecking variant: {variant}\n{'='*60}")

    test_ds = fix_test_labels(
        ImageFolderEval(TEST_INSTANCES / variant / "val", image_size=IMAGE_SIZE),
        train_class_to_idx
    )

    # Group a handful of sample paths per assigned label, so we can
    # visually confirm the folder name matches the assigned class name
    seen_labels = {}
    for path, label in test_ds.samples:
        if label not in seen_labels:
            seen_labels[label] = path

    print(f"Total images: {len(test_ds.samples)}")
    print(f"{'Label':<6}{'Assigned class':<14}{'Example file path'}")
    print("-" * 90)
    for label in sorted(seen_labels.keys()):
        path = seen_labels[label]
        assigned_class = idx_to_class[label]
        print(f"{label:<6}{assigned_class:<14}{path}")

        # Sanity check: does the folder name in the path roughly match
        # the assigned class name? (case-insensitive substring check)
        folder_in_path = pathlib.Path(path).parent.name.lower()
        if assigned_class.lower() not in folder_in_path and \
           not any(word in folder_in_path for word in [assigned_class.lower()]):
            print(f"    WARNING: path doesn't obviously match assigned class - check manually")
            all_ok = False

    if len(seen_labels) != 9:
        print(f"\nWARNING: expected 9 distinct labels, found {len(seen_labels)}")
        all_ok = False

print(f"\n\n{'='*60}")
if all_ok:
    print("All variants show 9 distinct, sensibly-matched labels.")
    print("Eyeball the file paths above yourself too - if 'wheeled vehicle'")
    print("shows up next to label 8 (Vehicle) and 'musical instrument' shows")
    print("up next to label 5 (Instrument), the fix is working correctly.")
else:
    print("Something looks off above - check the WARNING lines.")