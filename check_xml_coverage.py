"""
Checks whether the XML Annotations tree has bounding boxes for images
that LOC_train_solution.csv is missing - i.e. tests whether the CSV
and XML sources are genuinely equivalent (as Kaggle's description
claims) or whether XML has extra coverage the CSV lacks.
"""

import os
import xml.etree.ElementTree as ET


def has_xml_bbox(image_id, annotations_root):
    """
    image_id: e.g. 'n02085620_16757' (no extension)
    Returns True if a valid XML annotation with a bounding box exists.
    """
    synset = image_id.split("_")[0]
    xml_path = os.path.join(annotations_root, synset, image_id + ".xml")
    if not os.path.exists(xml_path):
        return False
    try:
        root = ET.parse(xml_path).getroot()
        return root.find("object") is not None
    except Exception:
        return False


def check_xml_coverage_for_missing(missing_ids, annotations_root, sample_size=None):
    """
    missing_ids: set/list of image_ids that had NO row in the CSV.
    Checks each (or a random sample) against the XML tree.
    Returns (recoverable_ids, still_missing_ids).
    """
    import random
    ids_to_check = list(missing_ids)
    if sample_size and sample_size < len(ids_to_check):
        ids_to_check = random.sample(ids_to_check, sample_size)

    recoverable, still_missing = [], []
    for image_id in ids_to_check:
        if has_xml_bbox(image_id, annotations_root):
            recoverable.append(image_id)
        else:
            still_missing.append(image_id)

    return recoverable, still_missing


if __name__ == "__main__":
    # Self-test with the fake data
    missing_ids = {"n01440764_100", "n02085620_200", "n01440764_999"}
    recoverable, still_missing = check_xml_coverage_for_missing(
        missing_ids, "fake_xml_check"
    )
    print(f"Recoverable via XML: {sorted(recoverable)}")
    print(f"Still missing: {sorted(still_missing)}")
    assert set(recoverable) == {"n01440764_100", "n02085620_200"}, "FAILED: wrong recoverable set"
    assert set(still_missing) == {"n01440764_999"}, "FAILED: wrong still-missing set"
    print("\nPASSED: correctly identifies which missing images ARE recoverable via XML,")
    print("and which are genuinely missing from both sources.")
