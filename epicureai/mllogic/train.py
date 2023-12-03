from ultralytics import YOLO
from epicureai.params import *
import os
import comet_ml
from comet_ml import API

workspace = "poloniki"
model_name = "yolo-model"
project = "epicure"

os.environ["COMET_MODEL_NAME"] = model_name


def train_model(epochs: int = 10, img_size: int = 512):
    # Initialize YOLO connection
    api = API()
    comet_ml.init()

    # If there are already pretrained weights - take them if not load new ones
    try:
        models = api.get_model(workspace=workspace, model_name=model_name)
        model_versions = models.find_versions()
        last_version = model_versions[0]
        weights_path = os.path.join(LOCAL_DATA_PATH, "weights")
        os.makedirs(weights_path, exist_ok=True)
        models.download(
            version=last_version,
            output_folder=weights_path,
            expand=True,
        )
        model = YOLO(os.path.join(weights_path, "best.pt"))
        print("✅ Loaded weights from the comet ML")
    except Exception as error:
        print(f"❌ Could not loaded weights: {error}")
        model = YOLO("yolov8n.pt")
        print("❗️ Initialized new weights from scratch")

    # Train the model
    model.train(
        data=os.path.join(LOCAL_DATA_PATH, "data.yaml"),
        epochs=epochs,
        imgsz=img_size,
        device="mps",
        patience=20,
        device="gpu",
    )

    # Saving latest model weights to the COMET ML online
    experiments = api.get(workspace=workspace, project_name=project)
    experiment = api.get(
        workspace=workspace,
        project_name=project,
        experiment=experiments[-1]._name,
    )
    experiment.register_model(model_name)
    print("✅ Save weights of this run to Comet ML")


if __name__ == "__main__":
    train_model(epochs=NUM_EPOCHS)
