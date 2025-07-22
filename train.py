from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/last.pt")
model.train(
    data="MRI-3/data.yaml",
    epochs=25,
    batch=32,
    plots=True,
    device=0, 
    resume=True  
)