from ultralytics import YOLO
from epicureai.params import *
import os
import comet_ml
from comet_ml import API

# Set up workspace, model name, and project variables
workspace = "poloniki"
model_name = "yolo-model"
project = "epicure"

# Set an environment variable for the model name
os.environ["COMET_MODEL_NAME"] = model_name


# Function to train the model
def train_model(epochs: int = 10, img_size: int = 512):
    # Initialize Comet ML API connection
    api = API()
    comet_ml.init()

    # Try to use pretrained weights if available
    try:
        # Fetching the model from Comet ML
        models = api.get_model(workspace=workspace, model_name=model_name)
        model_versions = models.find_versions()
        last_version = model_versions[0]

        # Preparing local path for weights
        weights_path = os.path.join(LOCAL_DATA_PATH, "weights")
        os.makedirs(weights_path, exist_ok=True)

        # Downloading the weights
        models.download(
            version=last_version,
            output_folder=weights_path,
            expand=True,
        )

        # Load the model with the downloaded weights
        model = YOLO(os.path.join(weights_path, "best.pt"))
        model.train(resume=True)
        print("✅ Loaded weights from the comet ML")

    # If loading pretrained weights fails, initialize a new model
    except Exception as error:
        print(f"❌ Could not load weights: {error}")

        # Initialize a new YOLO model with default weights
        model = YOLO("yolov8n.pt")
        model.train(
            data=os.path.join(LOCAL_DATA_PATH, "data.yaml"),
            epochs=epochs,
            imgsz=img_size,
            patience=20,
        )
        print("❗️ Initialized new weights from scratch")

    # Save the trained model weights to Comet ML
    experiments = api.get(workspace=workspace, project_name=project)

    # Registering the latest experiment and model
    experiment = api.get(
        workspace=workspace,
        project_name=project,
        experiment=experiments[-1]._name,
    )
    experiment.register_model(model_name)
    print("✅ Saved weights of this run to Comet ML")


# Main execution
if __name__ == "__main__":
    train_model(epochs=NUM_EPOCHS)
