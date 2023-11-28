from ultralytics import YOLO
from epicureai.params import *
import os
import comet_ml


def train_model(epochs: int = 10, img_size: int = 512):
    comet_ml.init()
    yaml_path = os.path.join(
        os.path.expanduser("~"), ".lewagon", "data", "roboflow", "data.yaml"
    )
    # Load the pre-trained model
    model = YOLO("yolov8n.pt")

    # Train the model
    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=img_size,
        save=True,
        device=DIVICE,
    )

    # Export the model to ONNX format
    path = model.export()
