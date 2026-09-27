"""
Prevents train/test leakage between the reconstructed IN-9L training set
and the released IN-9 test variants.

The IN-9 test images use ImageNet's training-style filenames
(e.g. n02085620_16757.JPEG), meaning they were drawn from the same pool
of ImageNet training images that we are now re-downloading via Kaggle to
rebuild IN-9L. Without exclusion, some exact files could end up in both
our training set and the (fixed, external) test set.

Usage:
    1. Point EXCLUDE_SOURCE_DIRS at the folder(s) containing the test
       variant data you already downloaded (e.g. the "original" variant
       is enough, since filenames repeat across variants).
    2. Build the exclusion set once.
    3. When copying/filtering files from your Kaggle IN-9L download,
       skip any filename present in the exclusion set.
"""

import os


def build_exclusion_set(*root_dirs):
    """
    Scan one or more directories (recursively) and collect every image
    filename found, regardless of which subfolder it's in.

    Returns a set of filenames (not full paths), since ImageNet filenames
    are unique identifiers on their own (e.g. n02085620_16757.JPEG always
    refers to the same source photo, wherever it appears).
    """
    excluded = set()
    for root_dir in root_dirs:
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                    excluded.add(fname)
    return excluded


def filter_out_leaked_files(source_dir, exclusion_set, dry_run=True):
    """
    Walk source_dir (your freshly downloaded Kaggle training data) and
    report/remove any file whose name appears in exclusion_set.

    dry_run=True only reports what WOULD be removed, without deleting
    anything -- use this first to sanity check before running for real.
    """
    matches = []
    for dirpath, _, filenames in os.walk(source_dir):
        for fname in filenames:
            if fname in exclusion_set:
                full_path = os.path.join(dirpath, fname)
                matches.append(full_path)
                if not dry_run:
                    os.remove(full_path)
    return matches


if __name__ == "__main__":
    # Self-test using a fake directory structure standing in for the real
    # test-variant folder and the real Kaggle training download.
    import tempfile
    import shutil

    tmp = tempfile.mkdtemp()
    test_variant_dir = os.path.join(tmp, "original", "val", "00_dog")
    training_download_dir = os.path.join(tmp, "kaggle_train", "n02085620")

    os.makedirs(test_variant_dir)
    os.makedirs(training_download_dir)

    # These files are "used in the test set" (like the real n02085620_16757.JPEG)
    test_files = ["n02085620_16757.JPEG", "n02085620_99999.JPEG"]
    for f in test_files:
        open(os.path.join(test_variant_dir, f), "w").close()

    # The fresh Kaggle training download contains a MIX of:
    #   - files that overlap with the test set (leakage risk)
    #   - files that are genuinely new / safe to train on
    overlapping = ["n02085620_16757.JPEG"]  # appears in both - should be caught
    safe_files = ["n02085620_11111.JPEG", "n02085620_22222.JPEG"]
    for f in overlapping + safe_files:
        open(os.path.join(training_download_dir, f), "w").close()

    print("=== Test setup ===")
    print(f"Test-variant files: {test_files}")
    print(f"Kaggle download files: {overlapping + safe_files}")
    print(f"(Expect exactly 1 file to be flagged as leakage: {overlapping})\n")

    exclusion_set = build_exclusion_set(os.path.join(tmp, "original"))
    print(f"Built exclusion set from test variants: {exclusion_set}\n")

    leaked = filter_out_leaked_files(training_download_dir, exclusion_set, dry_run=True)
    print(f"Files flagged as leaked (dry run, nothing deleted): {leaked}")

    assert len(leaked) == 1, f"FAILED: expected 1 leaked file, got {len(leaked)}"
    assert "n02085620_16757.JPEG" in leaked[0], "FAILED: wrong file flagged"
    print("\nPASSED: exactly the overlapping file was correctly identified.")

    remaining = set(os.listdir(training_download_dir)) - {os.path.basename(f) for f in leaked}
    print(f"Files that would remain for training: {sorted(remaining)}")
    assert remaining == set(safe_files), "FAILED: safe files incorrectly affected"
    print("PASSED: safe, non-overlapping files were left untouched.")

    shutil.rmtree(tmp)
    print("\nAll self-tests passed. Script is ready for the real data.")
