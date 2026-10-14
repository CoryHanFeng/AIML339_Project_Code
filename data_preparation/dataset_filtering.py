"""
Removes the photos flagged by checkwholetraining.py from the validation set and builds a clean copy of the four test variants.
A test image is left out if its object photo or any of its background photos has a near-duplicate in the ImageNet-1k training data.
The same object photo is removed from every variant, so the four variants stay paired.
The original test files in data/bg_challenge are never changed.
"""

# https://docs.python.org/3/library/shutil.html
import shutil
import re
from collections import Counter
from pathlib import Path

# Paths are built from where this file is saved, so it can be run from any folder
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VAL_PATH = ROOT / "data" / "rebuild" / "in9_val"
TEST_PATH = ROOT / "data" / "bg_challenge"
CLEAN_TEST_PATH = ROOT / "data" / "bg_challenge_clean"
VARIANTS = ["original", "no_fg", "mixed_same", "mixed_rand"]

# Read the ImageNet validation numbers of every flagged photo from the output of checkwholetraining.py
overlapping_files = {int(n) for n in re.findall(r"ILSVRC2012_val_(\d+)", (HERE / "pretraining_overlap.txt").read_text())}
if not overlapping_files:
    raise ValueError("No images found in file")

# delete flagged photos
files_to_remove = [p for p in VAL_PATH.rglob("*JPEG") if int(p.stem.split("_")[-1]) in overlapping_files]
for path in files_to_remove:
    path.unlink()

# Collect every test image with the photo numbers in its name
# the first number is the object photo and any others are the background photos
new_test_set_values = []
for variant in VARIANTS:
    for path in (TEST_PATH / variant).rglob("*JPEG"):
        image_numbers = [int(n) for n in re.findall(r"_(\d+)", path.stem)]
        new_test_set_values.append((path,image_numbers))

# Leave out an object photo in every variant if any photo used to make it was flagged
dont_include_image = {image_numbers[0] for path, image_numbers in new_test_set_values if overlapping_files.intersection(image_numbers)}

# Copy the rest into the clean test folder, keeping the same folder layout
for path, image_numbers in new_test_set_values:
    if image_numbers[0] not in dont_include_image:
        new_path = CLEAN_TEST_PATH / path.relative_to(TEST_PATH)
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, new_path)



