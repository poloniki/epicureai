from ultralytics import YOLO
from epicureai.params import *
import os
import comet_ml


def train_model(epochs: int = 10, img_size: int = 512):
    comet_ml.init()

    # Load the pre-trained model
    model = YOLO("yolov8n.pt")

    # Train the model
    model.train(
        data="epicureai/mllogic/data.yaml",
        epochs=epochs,
        imgsz=img_size,
        device=DEVICE,
    )
