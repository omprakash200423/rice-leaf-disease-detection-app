import os, shutil, random

SOURCE_DIR = "."          # current folder (Rice_Leaf_AUG)
DEST_DIR = "dataset"

splits = [0.7, 0.85]      # 70% train, 15% val, 15% test

for cls in os.listdir(SOURCE_DIR):
    if cls == "dataset" or not os.path.isdir(cls):
        continue

    images = os.listdir(cls)
    random.shuffle(images)

    train_cut = int(len(images) * splits[0])
    val_cut = int(len(images) * splits[1])

    for i, img in enumerate(images):
        if i < train_cut:
            split = "train"
        elif i < val_cut:
            split = "val"
        else:
            split = "test"

        os.makedirs(f"{DEST_DIR}/{split}/{cls}", exist_ok=True)
        shutil.copy(
            f"{cls}/{img}",
            f"{DEST_DIR}/{split}/{cls}/{img}"
        )

print("✅ Dataset split completed")
