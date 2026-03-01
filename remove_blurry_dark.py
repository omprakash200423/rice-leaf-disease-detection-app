import cv2
import os
import numpy as np

DATASET_DIR = "dataset"

BLUR_THRESHOLD = 100      # lower = more strict
DARK_THRESHOLD = 40       # lower = darker

removed = 0

for split in ["train", "val", "test"]:
    split_path = os.path.join(DATASET_DIR, split)

    for cls in os.listdir(split_path):
        cls_path = os.path.join(split_path, cls)

        for img in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img)

            image = cv2.imread(img_path)
            if image is None:
                os.remove(img_path)
                removed += 1
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Blur detection
            blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Darkness detection
            brightness = np.mean(gray)

            if blur_value < BLUR_THRESHOLD or brightness < DARK_THRESHOLD:
                os.remove(img_path)
                removed += 1

print(f"✅ Blurry/Dark image removal completed. Removed {removed} images.")
