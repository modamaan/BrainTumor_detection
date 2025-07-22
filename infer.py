from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
model(source="MRI-3/valid/images", conf=0.25, save=True)