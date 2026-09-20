import json
import shutil
from pathlib import Path

# Base path to the extracted Roboflow directory
SOURCE_DIR = Path(
    "raw_coco_dataset/Electrical-Clutter-Segmentation.coco-segmentation"
)
OUTPUT_DIR = Path("dataset")

splits = ["train", "valid"]

for split in splits:
  split_src = SOURCE_DIR / split
  json_path = split_src / "_annotations.coco.json"

  if not json_path.exists():
    print(f"Warning: {json_path} not found. Skipping {split}.")
    continue

  with open(json_path, "r") as f:
    coco = json.load(f)

  # Filter out Electrical-Clutter-Segmentation and ensure: 0 = pole, 1 = wire_cluster
  raw_categories = [
      c
      for c in coco["categories"]
      if "clutter-segmentation" not in c["name"].lower()
  ]
  raw_categories.sort(key=lambda c: (0 if "pole" in c["name"].lower() else 1))

  category_map = {}
  class_names = []
  for idx, cat in enumerate(raw_categories):
    category_map[cat["id"]] = idx
    class_names.append(cat["name"])

  images_info = {img["id"]: img for img in coco["images"]}
  img_annotations = {img["id"]: [] for img in coco["images"]}
  for ann in coco["annotations"]:
    if ann["category_id"] in category_map:
      img_annotations[ann["image_id"]].append(ann)

  # Prepare destination folders
  out_img_dir = OUTPUT_DIR / split / "images"
  out_lbl_dir = OUTPUT_DIR / split / "labels"
  out_img_dir.mkdir(parents=True, exist_ok=True)
  out_lbl_dir.mkdir(parents=True, exist_ok=True)

  for img_id, anns in img_annotations.items():
    img_meta = images_info[img_id]
    w, h = img_meta["width"], img_meta["height"]
    file_name = img_meta["file_name"]
    file_stem = Path(file_name).stem

    # Copy image
    src_img = split_src / file_name
    if src_img.exists():
      shutil.copy(src_img, out_img_dir / file_name)

    # Convert polygon segmentations to YOLO format
    lines = []
    for ann in anns:
      cls_idx = category_map[ann["category_id"]]
      for seg in ann.get("segmentation", []):
        if len(seg) < 6:
          continue
        normalized_coords = []
        for i in range(0, len(seg), 2):
          nx = seg[i] / w
          ny = seg[i + 1] / h
          normalized_coords.extend([f"{nx:.6f}", f"{ny:.6f}"])
        lines.append(f"{cls_idx} " + " ".join(normalized_coords))

    with open(out_lbl_dir / f"{file_stem}.txt", "w") as lf:
      lf.write("\n".join(lines))

  print(f"Processed {split}: {len(images_info)} images")

# Write data.yaml
yaml_content = f"""path: {OUTPUT_DIR.resolve()}
train: train/images
val: valid/images

names:
"""
for idx, name in enumerate(class_names):
  yaml_content += f"  {idx}: {name}\n"

with open(OUTPUT_DIR / "data.yaml", "w") as yf:
  yf.write(yaml_content)

print("Updated data.yaml successfully.")