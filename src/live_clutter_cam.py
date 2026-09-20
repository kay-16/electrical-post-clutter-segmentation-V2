from pathlib import Path
import time
import cv2
import numpy as np
from ultralytics import YOLO


def run_live_clutter_inference(
    model_path: str, camera_index: int = 0, conf: float = 0.25
):
  """Captures live webcam video, segments poles and wire clusters,

  and displays the live clutter severity ratio.
  """
  model = YOLO(model_path)
  cap = cv2.VideoCapture(camera_index)

  # Optional: set camera resolution (e.g. 1280x720 or 640x480)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

  if not cap.isOpened():
    print(
        f"Error: Could not open camera with index {camera_index}. Try index 1 or"
        " check permissions."
    )
    return

  print(
      "Starting live camera stream. Press 'q' to exit or 's' to take a"
      " snapshot."
  )

  prev_time = time.time()

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      print("Failed to read frame from camera.")
      break

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if curr_time > prev_time else 0.0
    prev_time = curr_time

    # Run YOLO inference on the current frame
    results = model.predict(source=frame, conf=conf, verbose=False)[0]

    h, w = frame.shape[:2]
    pole_mask = np.zeros((h, w), dtype=np.uint8)
    wire_mask = np.zeros((h, w), dtype=np.uint8)

    pole_pixels = 0
    wire_pixels = 0
    ratio = 0.0
    severity = "No Detections"
    status_color = (180, 180, 180)  # Gray default

    if results.masks is not None and len(results.masks) > 0:
      class_names = results.names
      segments = results.masks.xy
      cls_ids = results.boxes.cls.cpu().numpy().astype(int)

      for seg, cls_id in zip(segments, cls_ids):
        if len(seg) < 3:
          continue
        poly = np.array(seg, dtype=np.int32)
        c_name = class_names[cls_id].lower()

        if "pole" in c_name:
          cv2.fillPoly(pole_mask, [poly], 1)
        elif "wire" in c_name:
          cv2.fillPoly(wire_mask, [poly], 1)

      pole_pixels = int(np.sum(pole_mask))
      wire_pixels = int(np.sum(wire_mask))

      if pole_pixels == 0:
        ratio = float("inf") if wire_pixels > 0 else 0.0
      else:
        ratio = wire_pixels / pole_pixels

      # Severity Level
      if ratio < 0.5:
        severity = "Low Clutter"
        status_color = (0, 200, 0)  # Green
      elif 0.5 <= ratio < 1.5:
        severity = "Moderate Clutter"
        status_color = (0, 200, 255)  # Amber / Yellow
      else:
        severity = "Critical Clutter"
        status_color = (0, 0, 255)  # Red

    # Apply color overlays
    overlay = frame.copy()
    overlay[wire_mask > 0] = [255, 255, 0]  # Cyan for wire clusters
    overlay[pole_mask > 0] = [255, 0, 0]  # Blue for poles
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # UI Dashboard HUD
    cv2.rectangle(frame, (10, 10), (320, 130), (20, 20, 20), -1)
    cv2.rectangle(frame, (10, 10), (320, 130), (80, 80, 80), 1)

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        f"Poles: {pole_pixels:,} px",
        (20, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 150, 50),
        1,
    )
    cv2.putText(
        frame,
        f"Wires: {wire_pixels:,} px",
        (20, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 0),
        1,
    )
    ratio_str = f"{ratio:.2f}" if ratio != float("inf") else "INF"
    cv2.putText(
        frame,
        f"Ratio: {ratio_str}",
        (20, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        f"Status: {severity}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        status_color,
        2,
    )

    cv2.imshow("Live Clutter Detection (Press 'q' to quit)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
      break
    elif key == ord("s"):
      snap_path = Path.cwd() / f"snapshot_{int(time.time())}.jpg"
      cv2.imwrite(str(snap_path), frame)
      print(f"Snapshot saved to: {snap_path}")

  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  project_root = Path.cwd()
  weights = project_root / "runs" / "pilot_experiment_v1" / "weights" / "best.pt"

  if not weights.exists():
    raise FileNotFoundError(f"Could not locate best.pt weights at {weights}")

  # camera_index=0 uses default built-in/primary webcam; change to 1 if using an external USB cam
  run_live_clutter_inference(model_path=str(weights), camera_index=0, conf=0.25)