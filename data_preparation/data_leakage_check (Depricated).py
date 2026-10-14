import csv
from pathlib import Path
import numpy as np
import cv2

DATA = Path("data")
TRAIN_DATA = DATA / "rebuild" /"in9_train"
VAL_DATA = DATA / "rebuild" /"in9_val"
TEST_DATA = DATA / "bg_challenge" / "original"
SETS = {"training": TRAIN_DATA, "validation": VAL_DATA, "test": TEST_DATA}

IMG_SIZE = 32
corresponding_threshold = 0.05
DELETE_FILES = False

def resize_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_2)
    h, w = image.shape
    # 224/256, which gives the usual ImageNet centre crop
    s = min(w, h) * .875
    y, x = round((h-s)/2), round((w-s)/2)
    resized_image = []
    for picture in (image, image[y:y+round(s), x:x+round(s)]):
        image_pixels = cv2.resize(picture, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
        resized_image.append(cv2.normalize(image_pixels - image_pixels.mean(),None, norm_type=cv2.NORM_L2).ravel())
    return resized_image

def find_closest_image(img_1, img_2):
    step = len(img_1) // 10 + 1
    closest_distance, index = np.zeros(len(img_1)), np.zeros(len(img_1), dtype=int)
    for i in range(0, len(img_1), step):
        correlation = np.matmul(img_1[i:i+step].reshape(-1, IMG_SIZE * IMG_SIZE),img_2[:,0].T).reshape(-1,2,len(img_2)).max(axis=1)
        closest_distance[i:i+step], index[i:i+step] = 1 - correlation.max(axis=1), correlation.argmax(axis=1)
    return closest_distance, index

paths = {}
new_pictures = {}
for name, folder in SETS.items():
    paths[name] = sorted(folder.rglob("*JPEG"))
    if not paths[name]:
        raise ValueError(f"No images found in {folder}")
    print(f"Found {len(paths[name])} images in {folder}")
    new_pictures[name] = np.array([resize_image(p) for p in paths[name]])

remove = set()
test_copies = set()
for name1, name2 in [("validation", "test"), ("training", "test"), ("training", "validation")]:
    closest_image, closest_index = find_closest_image(new_pictures[name1], new_pictures[name2])
    found = np.flatnonzero(closest_image <= corresponding_threshold)
    print(f"{name1} and {name2} have {len(found)} near duplicate images")
    for k in found:
        remove.add(paths[name1][k])
        print("   ", paths[name1][k], paths[name2][closest_index[k]], round(closest_image[k], 4))
        if name2 == "test":
            test_copies.add(paths[name2][closest_index[k]])

print(f"There are {len(test_copies)} possible duplicate photos in the training and validation sets that could create leakage")
if DELETE_FILES:
    print(f"Deleting {len(remove)} files")
else:
    print(f"These files would be deleted if DELETE_FILES is TRUE: {len(remove)}")
for path in sorted(remove):
    print(" ", path.relative_to(DATA))
    if DELETE_FILES:
        path.unlink()


