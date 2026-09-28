import os
import random
import csv
import re

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_tensor
from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VIT_MEAN = (0.5, 0.5, 0.5)
VIT_STD = (0.5, 0.5, 0.5)

try:
    import cv2
    cv2.setNumThreads(0)
except ImportError:
    cv2 = None

def load_bbox_csv(csv_path):
    """Load LOC_train_solution.csv into {image_id: (xmin, ymin, xmax, ymax)}, keeps only the first box per image."""
    bboxes = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            # split the class ID and the bounding box values and store them against the ID
            xmin, ymin, xmax, ymax = map(float, row["PredictionString"].split()[1:5])
            bboxes[row["ImageId"]] = (xmin, ymin, xmax, ymax)
    return bboxes

class IN9BoundingBoxDataset(Dataset):
    """Loads in9L_final images and returns (image, label, scaled bounding box), bounding box falls back to full frame."""

    def __init__(self, root_dir, bbox_dict, image_size=224):
        self.bbox_dict = bbox_dict
        self.image_size = image_size
        folder = ImageFolder(root_dir)
        self.classes, self.class_to_idx, self.samples = folder.classes, folder.class_to_idx, folder.samples
        self.resize = transforms.Resize((image_size, image_size))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size

        image_id = os.path.splitext(os.path.basename(image_path))[0]
        bbox = self.bbox_dict.get(image_id, (0, 0, orig_w, orig_h))

        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h
        xmin, ymin, xmax, ymax = bbox
        bbox_scaled = (xmin * scale_x, ymin * scale_y, xmax * scale_x, ymax * scale_y)

        return to_tensor(self.resize(img)), label, bbox_scaled

def build_mask_from_bbox(bbox, image_size):
    """Binary mask, 1.0 = background, 0.0 = inside the box."""
    H, W = image_size
    xmin, ymin, xmax, ymax = [int(round(v)) for v in bbox]
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(W, xmax), min(H, ymax)
    # Create grid full of ones
    mask = torch.ones((1, H, W), dtype=torch.float32)
    # select rectangular region defined by box, set tgat to 0 for object.
    mask[:, ymin:ymax, xmin:xmax] = 0.0
    return mask

def composite_background(fg_image, mask, new_background):
    """combined = fg * (1 - mask) + mask * new_bg.
    swaps in the background image for the foreground image.
    """
    return fg_image * (1 - mask) + mask * new_background

def grabcut_mask_from_bbox(image_tensor, bbox, iterations=2, dilation_px=4, min_fg_fraction=0.02):
    """Conservative GrabCut mask, fallbacks to the bounding box if doesnt work; returns (mask, succeeded)."""
    C, H, W = image_tensor.shape
    # if doesnt have cv2, return the bounding box mask
    if cv2 is None:
        return build_mask_from_bbox(bbox, (H, W)), False

    # covert image to BGR
    img_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # get the bounding box
    xmin, ymin, xmax, ymax = [int(round(v)) for v in bbox]
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(W, xmax), min(H, ymax)
    rect = (xmin, ymin, xmax - xmin, ymax - ymin)

    gc_mask = np.zeros((H, W), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    # try grabcut
    try:
        cv2.grabCut(img_bgr, gc_mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
        fg_binary = np.where((gc_mask == 1) | (gc_mask == 3), 1, 0).astype(np.uint8)

        # if the mask is too small, return the bounding box mask
        if fg_binary.sum() / (H * W) < min_fg_fraction:
            raise ValueError("GrabCut result too small")
        # potentially grow the mask to try prevent cutting into object
        if dilation_px > 0:
            kernel = np.ones((dilation_px, dilation_px), np.uint8)
            fg_binary = cv2.dilate(fg_binary, kernel, iterations=1)

        mask = torch.from_numpy(1 - fg_binary).float().unsqueeze(0)
        return mask, True
    except Exception:
        mask = torch.ones((1, H, W), dtype=torch.float32)
        mask[:, ymin:ymax, xmin:xmax] = 0.0
        return mask, False

class BackgroundRandomizationDataset(Dataset):
    """Wraps IN9BoundingBoxDataset and swaps backgrounds with probability = augmentation_strength."""

    def __init__(self, base_dataset, augmentation_strength=1.0, seed=None,
                 min_donor_background_fraction=0.3, use_grabcut=True, mean = IMAGENET_MEAN, std = IMAGENET_STD):
        assert 0.0 <= augmentation_strength <= 1.0
        self.base_dataset = base_dataset
        self.strength = augmentation_strength
        self.rng = random.Random(seed)
        self.min_donor_background_fraction = min_donor_background_fraction
        self.use_grabcut = use_grabcut
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __len__(self):
        return len(self.base_dataset)

    def _mask_for(self, image, bbox):
        if self.use_grabcut:
            return grabcut_mask_from_bbox(image, bbox)[0]
        H, W = image.shape[1], image.shape[2]
        return build_mask_from_bbox(bbox, (H, W))

    def __getitem__(self, index):
        image, label, bbox = self.base_dataset[index]

        if self.rng.random() >= self.strength:
            return self._normalise(image), label

        H, W = image.shape[1], image.shape[2]
        mask = self._mask_for(image, bbox)

        bg_image, bg_own_mask = image, torch.ones((1, H, W), dtype=torch.float32)
        # Checks 5 potential donor images to receive background from, skips if same if the current image
        for _ in range(5):
            bg_index = self.rng.randrange(len(self.base_dataset))
            if bg_index == index and len(self.base_dataset) > 1:
                continue
            candidate_image, _, candidate_bbox = self.base_dataset[bg_index]
            candidate_mask = build_mask_from_bbox(candidate_bbox, (H, W))
            # if above threshold, accept, if no 5 images work, go back to original background
            if candidate_mask.mean().item() >= self.min_donor_background_fraction:
                bg_image, bg_own_mask = candidate_image, candidate_mask
                break
        # build final image
        composited = composite_background(image, mask, bg_image * bg_own_mask)
        return self._normalise(composited), label

    def _normalise(self,image):
        return( image - self.mean ) / self.std

def _seed_worker(worker_id):
    info = torch.utils.data.get_worker_info()
    if isinstance(info.dataset, BackgroundRandomizationDataset):
        info.dataset.rng.seed(info.seed
                              )

class ImageFolderEval(Dataset):
    """loader for the IN-9 test variant folders (w/ no bounding boxes)."""

    def __init__(self, root_dir, image_size=224, mean = IMAGENET_MEAN, std = IMAGENET_STD):
        folder = ImageFolder(root_dir)
        self.classes, self.class_to_idx, self.samples = folder.classes, folder.class_to_idx, folder.samples
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean= list(mean), std = list(std))
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

def stratified_train_val_split(dataset, val_fraction=0.1, seed=0):
    """Split into train/val subsets, stratified by class so val stays balanced."""
    labels = [label for _, label in dataset.samples]
    indices = list(range(len(labels)))
    train_indices, val_indices = train_test_split(
        indices, test_size=val_fraction, stratify=labels, random_state=seed
    )
    return Subset(dataset, train_indices), Subset(dataset, val_indices)

def _dataset_returns_bbox(dataset):
    # look at how much the first instance of the dataset returns
    if len(dataset) == 0:
        return False
    return len(dataset[0]) == 3


def _collate_drop_bbox(batch):
    # build the batch
    images = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch])
    return images, labels


def make_loader(dataset, batch_size=32, shuffle=True, num_workers=4, persistent_workers=False):
    """DataLoader that drops the bbox from 3-tuple datasets so batches are (images, labels)."""
    collate = _collate_drop_bbox if _dataset_returns_bbox(dataset) else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, collate_fn=collate, worker_init_fn=_seed_worker ,persistent_workers= persistent_workers and num_workers >0)


## FIX FOR DATA FOLDER LABEL INCONSISTENCIES
# Generative AI was used to help make this

TEST_FOLDER_NAME_TO_TRAIN_CLASS = {
    "dog": "Dog", "bird": "Bird", "wheeled vehicle": "Vehicle",
    "reptile": "Reptile", "carnivore": "Carnivore", "insect": "Insect",
    "musical instrument": "Instrument", "primate": "Primate", "fish": "Fish",
}

def fix_test_labels(test_dataset, train_class_to_idx):
    """
    Fixes the test set labels, the test set has different name prefixes that give different label order than the training folder.
    This function rewrites the labels to match the training label order to prevent inaccurate model results.
    """
    old_idx_to_name = {v: k for k, v in test_dataset.class_to_idx.items()}
    remap = {}
    for old_idx, folder_name in old_idx_to_name.items():
        fragment = re.sub(r"^\d+_", "", folder_name).lower()
        train_class = TEST_FOLDER_NAME_TO_TRAIN_CLASS[fragment]
        remap[old_idx] = train_class_to_idx[train_class]

    test_dataset.samples = [(path, remap[label]) for path, label in test_dataset.samples]
    test_dataset.class_to_idx = train_class_to_idx
    return test_dataset


if __name__ == "__main__":
    # Create a fake folder, that has the exact same structure as the original dataset, just for matching to ensure that the test works.
    class FakeImageFolderEval:
        def __init__(self, root):
            classes = sorted(os.listdir(root))
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
            self.samples = []
            for c in classes:
                for fname in os.listdir(os.path.join(root, c)):
                    self.samples.append((os.path.join(root, c, fname), self.class_to_idx[c]))


    fake_test_folders = {
        "00_dog": "Dog", "01_bird": "Bird", "02_wheeled vehicle": "Vehicle",
        "03_reptile": "Reptile", "04_carnivore": "Carnivore", "05_insect": "Insect",
        "06_musical instrument": "Instrument", "07_primate": "Primate", "08_fish": "Fish",
    }
    root = "fake_test_short"
    for folder_name in fake_test_folders:
        os.makedirs(os.path.join(root, folder_name), exist_ok=True)
        open(os.path.join(root, folder_name, "img_0.JPEG"), "w").close()

    train_class_to_idx = {
        'Bird': 0, 'Carnivore': 1, 'Dog': 2, 'Fish': 3, 'Insect': 4,'Instrument': 5, 'Primate': 6, 'Reptile': 7, 'Vehicle': 8
    }
    # test if the mapping worked
    test_ds = FakeImageFolderEval(root)
    print(f"before fix, class_to_idx: {test_ds.class_to_idx}\n")

    fix_test_labels(test_ds, train_class_to_idx)
    print(f"after fix, class_to_idx: {test_ds.class_to_idx}\n")

    all_correct = True
    for path, label in test_ds.samples:
        folder_name = os.path.basename(os.path.dirname(path))
        expected_class = fake_test_folders[folder_name]
        expected_label = train_class_to_idx[expected_class]
        status = "good" if label == expected_label else "error, does not match"
        if label != expected_label:
            all_correct = False
        print(f"  {folder_name:<25} -> label {label} (expected {expected_label})  {status}")

    assert all_correct, "did not fix all test labels"
    print("\nall test labels fixed successfully.")




