from roboflow import Roboflow
import os

rf = Roboflow(api_key="aj3VeHSfSTCgQujNVxtC")
project = rf.workspace("brain-mri").project("mri-rskcu")
version = project.version(3)
dataset = version.download("yolov8")  # Folder like "MRI-3"

# Download YOLOv10n weights
if not os.path.exists("yolov10n.pt"):
    import urllib.request
    url = "https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10n.pt"
    urllib.request.urlretrieve(url, "yolov10n.pt")
