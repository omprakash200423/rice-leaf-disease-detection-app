import os
import hashlib

DATASET_DIR = "dataset"

def hash_image(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

seen = set()
removed = 0

for split in ["train", "val", "test"]:
    split_path = os.path.join(DATASET_DIR, split)

    for cls in os.listdir(split_path):
        cls_path = os.path.join(split_path, cls)

        for img in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img)
            img_hash = hash_image(img_path)

            if img_hash in seen:
                os.remove(img_path)
                removed += 1
            else:
                seen.add(img_hash)

print(f"✅ Duplicate removal completed. Removed {removed} images.")
