"""
Builds the training and validation sets straight from the ImageNet zip, Only the images that are chosen are copied out of the zip.

Before running check the three paths below, and put in9_download_list.json (the list of
which ImageNet synsets belong to each of the 9 classes) in the same folder as this script.

What it makes (inside OUT_DIR):
  - in9_train: ImageNet training images from the IN-9 synsets that have a bounding box, with the same number of images in every class.
  - in9_val: ImageNet validation images from the same synsets, leaving out every image the Backgrounds Challenge uses, balanced the same way.
  - train_manifest.csv and val_manifest.csv (next to this script): the exact list of files chosen. (so it is easy to reproduce the exact same dataset)

Generative AI was used to help write, debug and make it adapt to original splitting, leakage functions into this one script.
"""
import csv
import json
import os
import random
import time
import zipfile

# Only zip-path and test path should be edited when trying to recreate the dataset
ZIP_PATH = r"E:\ILSVRC.zip"

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
OUT_DIR = os.path.join(DATA, "rebuild")
TEST_DIR = os.path.join(DATA,"bg_challenge")

def read_csv_rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))

def pick_balanced(files_by_class):
    # Every class is cut down to the size of the smallest class.
    # The choice is random but uses a fixed seed (0), so the same input gives the same output.
    # The lists are sorted and the classes go in alphabetical order because one random
    # generator is shared by all classes, so the order has to be the same every time.
    smallest = min(len(files) for files in files_by_class.values())
    assert smallest > 0, "at least one class should have at least one image"
    rng = random.Random(0)
    return {cls: rng.sample(sorted(files_by_class[cls]), smallest) for cls in sorted(files_by_class.keys())}


def copy_from_zip(zf, chosen, out_dir, manifest_path):
    # Copies the chosen images out of the zip into one folder per class, and writes a
    # manifest: a CSV with one row (class, filename) for every image that was copied.
    with open(manifest_path, "w",newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "filename"])
        for cls, files in chosen.items():
            os.makedirs(os.path.join(out_dir, cls))
            for name in sorted(files):
                filename = os.path.basename(name)
                with open(os.path.join(out_dir, cls, filename), "wb") as out:
                    out.write(zf.read(name))
                writer.writerow([cls,filename])
            print("copied", cls)


def main():
    # Stop early if something is missing, so nothing is overwritten by accident.
    if os.path.exists(OUT_DIR):
        raise SystemExit(f"{OUT_DIR} already exists, delete or rename it first")
    if not os.path.exists(ZIP_PATH):
        raise SystemExit(f"zip not found {ZIP_PATH}")
    print("reading from:", ZIP_PATH)
    print("saving to:", OUT_DIR)

    # Map every synset (an ImageNet category ID such as n02085620) to its class
    # (Dog, Bird, ...), using the list of 370 synsets.
    with open(os.path.join(HERE, "in9_download_list.json")) as f:
        download_list = json.load(f)
    synset_to_class = {e["synset_id"]: cls for cls, entries in download_list.items() for e in entries}

    # The Backgrounds Challenge test images are ImageNet validation images, and the number
    # in each filename is the validation image number. Collect those numbers in the
    # ImageNet validation naming style (ILSVRC2012_val_00016757) so we can leave these
    # images out of our validation set.
    # Filename examples, both give 16757:
    #   n02085620_16757.JPEG
    #   fg_n02085620_16757_bg_n01532829_06835.JPEG (only the first number is used)
    challenge_ids = set()
    for folder, _, files in os.walk(TEST_DIR):
        for name in files:
            if name.endswith(".JPEG"):
                number = name.removeprefix("fg_").split("_")[1].split(".")[0]
                challenge_ids.add(f"ILSVRC2012_val_" + number.zfill(8))

    # If this is empty the challenge folder is missing, and the validation set would
    # wrongly include test images, stop running script.
    if not challenge_ids:
        raise SystemExit(f"no Backgrounds Challenge images found in {TEST_DIR}")
    print("challenge images to leave out of validation:", len(challenge_ids))

    with zipfile.ZipFile(ZIP_PATH) as zf:
        print("reading the zip's file list (can take awhile)")
        names = zf.namelist()

        # The IDs of the training images that have a bounding box, and the category
        # (synset) of every validation image.
        annotated = {row["ImageId"] for row in read_csv_rows(os.path.join(DATA, "LOC_train_solution.csv"))}
        val_synset = {row["ImageId"]: row["PredictionString"].split()[0] for row in read_csv_rows(os.path.join(DATA, "LOC_val_solution.csv"))}

        # Go through every file in the zip
        # Splitting on "/" and reading from the END works whatever folder the zip starts with.
        train_by_class = {cls: [] for cls in download_list}
        val_by_class = {cls: [] for cls in download_list}
        for name in names:
            if not name.lower().endswith(".jpeg"):
                continue
            parts = name.split("/")
            image_id = parts[-1][:-5]  # the file name without ".JPEG"
            if len(parts) >= 4 and parts[-3] == "train" and parts[-4] == "CLS-LOC":
                # Training image: keep it if its synset is one of the 370 and it has a bounding box.
                synset = parts[-2]
                if synset in synset_to_class and image_id in annotated:
                    train_by_class[synset_to_class[synset]].append(name)
            elif len(parts) >= 3 and parts[-2] == "val" and parts[-3] == "CLS-LOC":
                # Validation image: keep it if its synset is one of the 370 and it is not
                # a challenge image.
                synset = val_synset.get(image_id)
                if synset in synset_to_class and image_id not in challenge_ids:
                    val_by_class[synset_to_class[synset]].append(name)

        # Shows how many images each class has before balancing; the smallest class sets the size.
        print("train available per class (will balance to lowest class size):", {c: len(v) for c, v in train_by_class.items()})
        print("val available per class (will balance to lowest class size):", {c: len(v) for c, v in val_by_class.items()})

        # Choose first, so a problem stops the script before any file is copied.
        train_chosen = pick_balanced(train_by_class)
        val_chosen = pick_balanced(val_by_class)
        copy_from_zip(zf, train_chosen, os.path.join(OUT_DIR, "in9_train"), os.path.join(HERE,"train_manifest.csv"))
        copy_from_zip(zf, val_chosen, os.path.join(OUT_DIR, "in9_val"), os.path.join(HERE,"val_manifest.csv"))
if __name__ == "__main__":
    main()