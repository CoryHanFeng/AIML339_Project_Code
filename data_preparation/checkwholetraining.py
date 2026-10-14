"""
Looks through every ImageNet training image in the zip (the images both backbones were pretrained on) for
near-duplicates of the photos behind the four test variants (the photos the objects come from and the photos
used as backgrounds) and of the validation photos.
Nothing is deleted or written, the results are printed. Save the printed output as data_preparation/pretraining_overlap.txt, which dataset_filtering.py reads.
Each photo is shrunk to a 32x32 grey picture and compared by pixel correlation, as in data_leakage_check (Depricated).py.

Generative AI was used to help identify a method for finding near-duplicate images and to help implement it and validating with most of the print statements.
"""
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

# Only ZIP_PATH needs editing; the other paths are built from where this file is saved
ZIP_PATH = Path(r"E:\ILSVRC.zip")
# Paths are built from where this file is saved, so it can be run from any folder
ROOT = Path(__file__).resolve().parent.parent
CHALLENGE = ROOT / "data" / "bg_challenge"
VALIDATION = ROOT / "data" / "rebuild" / "in9_val"
VARIANTS = ["original", "no_fg", "mixed_same", "mixed_rand"]
IMG_SIZE = 32
THRESHOLD = 0.03  # a distance (1 - correlation) of this or less counts as the same photo
WORKERS = 6       # threads used to decode images
BATCH = 2000      # training images compared with the other photos at a time


def shrink(image):
    # The whole photo and its centre square as grey IMG_SIZE x IMG_SIZE pictures, with the average
    # brightness subtracted and the pixels scaled to length 1
    if image is None:  # an unreadable file can never match
        return [np.zeros(IMG_SIZE * IMG_SIZE, dtype=np.float32)] * 2
    h, w = image.shape
    s = min(w, h) * .875
    y, x = round((h - s) / 2), round((w - s) / 2)
    # The centre square (87.5%) is kept as well, so a training photo that was cropped still matches
    versions = []
    for picture in (image, image[y:y + round(s), x:x + round(s)]):
        pixels = cv2.resize(picture, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
        versions.append(cv2.normalize(pixels - pixels.mean(), None, norm_type=cv2.NORM_L2).ravel())
    return versions


def shrink_bytes(data):
    # Decode straight to grey at half size (faster), then shrink
    return shrink(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_2))


# The ImageNet validation number of every photo behind the four test variants. Filenames look like
# n02085620_16757.JPEG (original, no_fg) or fg_n02085620_16757_bg_n01532829_06835.JPEG (the object photo, then the background photo)
numbers = set()
for variant in VARIANTS:
    for path in (CHALLENGE / variant).rglob("*JPEG"):
        numbers.update(int(n) for n in re.findall(r"_(\d+)", path.stem))
if not numbers:
    raise ValueError(f"No images found in {CHALLENGE}, check the data folder")
object_photos = {int(re.findall(r"_(\d+)", p.stem)[0]) for p in (CHALLENGE / "original").rglob("*JPEG")}
print(f"{len(numbers)} different photos are behind the test variants, {len(object_photos)} of them as the object photo")

# The validation photos are checked as well. Their filenames look like ILSVRC2012_val_00016757.JPEG
validation_photos = {int(p.stem.split("_")[-1]) for p in VALIDATION.rglob("*JPEG")}
if not validation_photos:
    raise ValueError(f"No images found in {VALIDATION}, check the data folder")
print(f"{len(validation_photos)} validation photos are checked as well, {len(validation_photos & numbers)} of them also behind the test variants")
numbers |= validation_photos

with zipfile.ZipFile(ZIP_PATH) as zf, ThreadPoolExecutor(WORKERS) as pool:
    # Every image in the zip, and the validation images by their number
    names = [n for n in zf.namelist() if n.lower().endswith(".jpeg")]
    val_files = {int(n.split("_")[-1].split(".")[0]): n for n in names if n.split("/")[-2:-1] == ["val"]}
    ids = sorted(n for n in numbers if n in val_files)
    print(f"Found {len(ids)} of them in the zip's validation folder, {len(numbers) - len(ids)} missing")
    # Only the whole photo is used for the test and validation photos, both versions for the training photos
    test = np.array([shrink_bytes(zf.read(val_files[n]))[0] for n in ids])
    train_names = [n for n in names if n.split("/")[-3:-2] == ["train"]]
    print(f"Found {len(train_names)} training images in the zip")
    # for each photo, the smallest distance found so far
    # and the training image that gave it
    best = np.full(len(ids), 9.0)
    match = [""] * len(ids)
    start = time.time()
    # Compare the training images in batches so they do not all have to be in memory
    for i in range(0, len(train_names), BATCH):
        chunk = train_names[i:i + BATCH]
        pictures = np.array(list(pool.map(shrink_bytes, [zf.read(n) for n in chunk])))
        # The pictures are scaled to length 1, so the dot product is the correlation; keep the better of the two versions
        correlation = np.matmul(pictures.reshape(-1, IMG_SIZE * IMG_SIZE), test.T).reshape(-1, 2, len(test)).max(axis=1)
        # Closest training image in this batch for each photo
        distance, source = 1 - correlation.max(axis=0), correlation.argmax(axis=0)
        # Keep it if it is closer than any found in earlier batches
        for j in np.flatnonzero(distance < best):
            best[j], match[j] = distance[j], chunk[source[j]]
        # Print progress every 25 batches
        if (i // BATCH) % 25 == 0:
            done = i + len(chunk)
            print(f"  {done}/{len(train_names)} checked, about {(time.time() - start) / done * (len(train_names) - done) / 60:.0f} minutes left")

# Photos with a near-duplicate, and how many photos fall under each distance
found = np.flatnonzero(best <= THRESHOLD)
print(f"{len(found)} of {len(ids)} photos have a near-duplicate among the {len(train_names)} training images")
print("photos by distance:", {d: int((best <= d).sum()) for d in (0.005, 0.01, 0.02, 0.03, 0.05)})
# List every flagged photo, closest first, with what it is used for and the matching training image
for j in sorted(found, key=lambda j: best[j]):
    role = "object photo" if ids[j] in object_photos else "validation photo" if ids[j] in validation_photos else "background photo only"
    print(f"  ILSVRC2012_val_{ids[j]:08d} ({role}) {match[j]} {round(best[j], 4)}")