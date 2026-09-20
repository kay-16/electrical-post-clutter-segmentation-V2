from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO


def compute_clutter_ratio(image_path: str, model_path: str, conf: float = 0.25):
  """Runs segmentation on an image and calculates wire clutter severity ratio.

  Args:
      image_path: Path to the target pole image.
      model_path: Path to trained best.pt weights.
      conf: Confidence threshold for predictions.
  """
  model = YOLO(model_path)
  results = model.predict(source=image_path, conf=conf, save=False)[0]

  # Check if instances with masks were detected
  if results.masks is None or len(results.masks) == 0:
    print(f"No objects detected in {Path(image_path).name}")
    return 0.0, None

  # Retrieve image dimensions and class definitions
  orig_h, orig_w = results.orig_shape
  class_names = results.names  # e.g., {0: 'pole', 1: 'wire_cluster'}

  # Initialize blank canvas for cumulative masks
  pole_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
  wire_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

  # Convert predicted masks (scaled to original image resolution)
  # results.masks.data is shape (N, H, W)
  masks_data = results.masks.data.cpu().numpy()
  cls_ids = results.boxes.cls.cpu().numpy().astype(int)

  for mask, cls_id in zip(masks_data, cls_ids):
    # Resize mask tensor to original image resolution if needed
    resized_mask = cv2.resize(
        mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
    )
    c_name = class_names[cls_id].lower()

    if "pole" in c_name:
      pole_mask = np.bitwise_or(pole_mask, resized_mask)
    elif "wire" in c_name:
      wire_mask = np.bitwise_or(wire_mask, resized_mask)

  # Calculate pixel surface areas
  pole_pixels = int(np.sum(pole_mask))
  wire_pixels = int(np.sum(wire_mask))

  # Prevent zero division if no pole is found
  if pole_pixels == 0:
    ratio = float("inf") if wire_pixels > 0 else 0.0
  else:
    ratio = wire_pixels / pole_pixels

  # Severity Level Classification
  if ratio < 0.5:
    severity = "Low Clutter / Organized"
  elif 0.5 <= ratio < 1.5:
    severity = "Moderate Clutter"
  else:
    severity = "High / Critical Clutter"

  print(f"\n--- Clutter Report: {Path(image_path).name} ---")
  print(f"Pole Mask Area        : {pole_pixels:,} px")
  print(f"Wire Cluster Area     : {wire_pixels:,} px")
  print(f"Wire-to-Pole Ratio    : {ratio:.3f}")
  print(f"Severity Category     : {severity}")

  # Generate color overlay visual
  annotated_bgr = results.orig_img.copy()
  # Cyan highlight for wires, Blue highlight for poles
  overlay = annotated_bgr.copy()
  overlay[wire_mask > 0] = [255, 255, 0]  # Cyan in BGR
  overlay[pole_mask > 0] = [255, 0, 0]  # Blue in BGR
  cv2.addWeighted(overlay, 0.4, annotated_bgr, 0.6, 0, annotated_bgr)

  return ratio, annotated_bgr


if __name__ == "__main__":
  # Set paths
  project_root = Path.cwd()
  weights_file = (
      project_root / "runs" / "pilot_experiment_v1" / "weights" / "best.pt"
  )

  # Pick an image from your validation split to test
  sample_img = list((project_root / "dataset" / "valid" / "images").glob("*"))[
      0
  ]

  ratio, visual_img = compute_clutter_ratio(str(sample_img), str(weights_file))

  if visual_img is not None:
    out_path = project_root / "clutter_assessment_sample.jpg"
    cv2.imwrite(str(out_path), visual_img)
    print(f"Assessment overlay saved to: {out_path}")