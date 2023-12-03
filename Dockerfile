FROM ultralytics/ultralytics:latest
COPY setup.py setup.py
RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -e .
COPY epicureai epicureai/
CMD python epicureai/interface/main.py

# Build (docker build)
# Push (docker push)
# Pull in cloud (docker pull)
# How to run: docker run -d --rm --runtime=nvidia FULL_IMAGE_NAME
