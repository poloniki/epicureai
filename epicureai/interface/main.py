from epicureai.data.load.roboflow_load import load_data_from_roboflow
from epicureai.mllogic.train import train_model
from epicureai.params import *


def main():
    load_data_from_roboflow()
    train_model(epochs=NUM_EPOCHS)


if __name__ == "__main__":
    main()
