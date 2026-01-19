import os

DATASET_DIR = "dataset"

for split in ["train", "val", "test"]:
    print(f"\n--- {split.upper()} ---")
    split_path = os.path.join(DATASET_DIR, split)

    for cls in os.listdir(split_path):
        cls_path = os.path.join(split_path, cls)
        count = len(os.listdir(cls_path))
        print(f"{cls}: {count}")
