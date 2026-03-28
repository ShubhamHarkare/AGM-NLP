#! Vibe-Code Alert: This validation script was originally written by Gemini Pro, updated for 4-split structure.

from datasets import load_from_disk
import os
import sys

BASE_DIR = os.path.dirname(__file__)

# CHANGE 1: updated all expected counts to match our agreed split sizes
# and added val and ads splits which did not exist before
EXPECTED_TRAIN = 10000
EXPECTED_VAL   = 2000
EXPECTED_TEST  = 3000
EXPECTED_ADS   = 500
# ------------------------------------

def validate_dataset_dir(dataset_dir: str):
    """Load all 4 splits under `dataset_dir` and assert they
    contain the exact expected number of rows.
    """
    dataset_name = os.path.basename(dataset_dir)

    # CHANGE 2: check for all 4 split directories instead of just train/test
    split_names = ["train", "val", "test", "ads"]
    paths = {split: os.path.join(dataset_dir, split) for split in split_names}

    # CHANGE 3: validate all 4 directories exist before loading
    for split, path in paths.items():
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Missing '{split}' directory under {dataset_name}. "
                f"Expected path: {path}"
            )

    # CHANGE 4: load all 4 splits
    train_ds = load_from_disk(paths["train"])
    val_ds   = load_from_disk(paths["val"])
    test_ds  = load_from_disk(paths["test"])
    ads_ds   = load_from_disk(paths["ads"])

    train_len = len(train_ds)
    val_len   = len(val_ds)
    test_len  = len(test_ds)
    ads_len   = len(ads_ds)

    # CHANGE 5: assert all 4 splits against their expected counts
    assert train_len == EXPECTED_TRAIN, (
        f"❌ Train count mismatch in {dataset_name}! "
        f"Expected {EXPECTED_TRAIN}, but got {train_len}."
    )
    assert val_len == EXPECTED_VAL, (
        f"❌ Val count mismatch in {dataset_name}! "
        f"Expected {EXPECTED_VAL}, but got {val_len}."
    )
    assert test_len == EXPECTED_TEST, (
        f"❌ Test count mismatch in {dataset_name}! "
        f"Expected {EXPECTED_TEST}, but got {test_len}."
    )
    assert ads_len == EXPECTED_ADS, (
        f"❌ ADS pool count mismatch in {dataset_name}! "
        f"Expected {EXPECTED_ADS}, but got {ads_len}."
    )

    return train_len, val_len, test_len, ads_len


def main():
    failures = []

    # CHANGE 6: updated header to reflect all 4 split targets
    print(f"--- Starting Validation ---")
    print(f"Target: {EXPECTED_TRAIN} Train | {EXPECTED_VAL} Val | {EXPECTED_TEST} Test | {EXPECTED_ADS} ADS\n")

    for name in sorted(os.listdir(BASE_DIR)):
        path = os.path.join(BASE_DIR, name)
        if not os.path.isdir(path):
            continue
        if name in ("raw_data", "__pycache__"):
            continue

        try:
            if os.path.exists(os.path.join(path, "train")):
                # CHANGE 7: unpack all 4 return values instead of just train/test
                train_len, val_len, test_len, ads_len = validate_dataset_dir(path)
                print(
                    f"✅ {name}: "
                    f"train={train_len}, "
                    f"val={val_len}, "
                    f"test={test_len}, "
                    f"ads={ads_len} OK"
                )
        except AssertionError as e:
            failures.append(str(e))
            print(e)
        except Exception as e:
            failures.append(str(e))
            print(f"❌ Error validating {name}: {e}")

    print("\n---------------------------")
    if failures:
        print("❌ Validation failed! Some datasets do not match your target numbers.")
        sys.exit(1)
    print("🎉 All datasets validated successfully and match your targets perfectly!")


if __name__ == "__main__":
    main()