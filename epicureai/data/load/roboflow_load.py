import requests
import zipfile
import os


def load_data_from_roboflow():
    url = "https://app.roboflow.com/ds/4d4aY2rAVy?key=W1qncyfr9X"
    zipfile_path = "dataset.zip"

    response = requests.get(url).content

    with open(zipfile_path, "wb") as f:
        f.write(response)

    # Destination directory using the home directory
    destination_directory = os.path.join(
        os.path.expanduser("~"), ".lewagon", "data", "roboflow"
    )

    # Check if the destination directory exists, if not, create it
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory, exist_ok=True)

    # Open and extract the ZIP file
    with zipfile.ZipFile(zipfile_path, "r") as zip_ref:
        zip_ref.extractall(destination_directory)
