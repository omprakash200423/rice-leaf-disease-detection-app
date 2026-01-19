import os

VAL_PATH = "dataset/val"

fix_map = {
    "Healthy_Rice_Leaf": "Healthy",
    "Leaf_scald": "Leaf_Scald"
}

for old, new in fix_map.items():
    old_path = os.path.join(VAL_PATH, old)
    new_path = os.path.join(VAL_PATH, new)

    if os.path.exists(old_path):
        os.rename(old_path, new_path)

print("✅ Validation folder names fixed")
