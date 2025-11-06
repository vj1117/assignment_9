import os
import requests
from datasets import load_dataset
from tqdm import tqdm  # for progress bar

dataset = load_dataset("ILSVRC/imagenet-1k", cache_dir="/mnt/sdb/huggingface_cache")

target_folder = "/mnt/sdb/imagenet/train"
os.makedirs(target_folder, exist_ok=True)

for split_name, split_ds in dataset.items():
    target_folder = f"/mnt/sdb/imagenet/{split_name}"
    os.makedirs(target_folder, exist_ok=True)
    for i, item in enumerate(tqdm(split_ds, desc=f"Saving {split_name}")):
        label = str(item["label"])
        class_folder = os.path.join(target_folder, label)
        os.makedirs(class_folder, exist_ok=True)

        img_path = os.path.join(class_folder, f"{i}.JPEG")
        if os.path.exists(img_path):
            continue
        try:
            image = item["image"]
            image.save(img_path, format="JPEG")
        except Exception as e:
            print(f"Failed to save {split_name} image {i}: {e}")
