import cv2
from ultralytics import YOLO
from roboflow import Roboflow

model = YOLO("yolov8s.pt")

results = model.train(
    data="downloaded_dataset/data.yaml",
    epochs=150,
    batch=32,           # Adjust based on GPU memory
    imgsz=640,
    optimizer="SGD",    # Explicitly set (optional, SGD is default)
    lr0=0.01,          # Initial LR (matches data.yaml)
    momentum=0.937,    # SGD momentum
    weight_decay=0.0005,
    augment=True,      # Enable built-in augmentations
    hsv_h=0.03,        # Fire hue augmentations
    hsv_s=0.7,         # Saturation for smoke
    device=0           # Use GPU 0
)
# rf = Roboflow(api_key="KNok77kTa9fDlg1ty7co")
# project = rf.workspace("vision-rqll3").project("smoke_fire_detect_dataset1")
# version = project.version(1)
# dataset = version.download("yolov8")


