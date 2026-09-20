from pathlib import Path
from ultralytics import YOLO


def main():
  # Current project root
  project_root = Path.cwd()

  data_yaml_path = (project_root / "dataset" / "data.yaml").resolve()
  runs_dir = (project_root / "runs").resolve()

  model = YOLO("yolov8n-seg.pt")

  results = model.train(
      data=str(data_yaml_path),
      epochs=100,
      patience=20,
      imgsz=640,
      batch=8,
      workers=2,
      device=0,
      project=str(runs_dir),  # Absolute path forces it into current repo
      name="pilot_experiment_v1",
      save=True,
      plots=True,
      exist_ok=True,  # Overwrites/updates folder instead of creating v2, v3
  )

  best_weights = runs_dir / "pilot_experiment_v1" / "weights" / "best.pt"
  print(f"Weights successfully saved to:\n{best_weights}")


if __name__ == "__main__":
  main()