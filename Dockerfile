FROM python:3.8-slim
COPY epicureai epicureai/
COPY setup.py setup.py
RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -e .
CMD python epicureai/interface/main.py

# Build (docker build)
# Push (docker push)
# Pull in cloud (docker pull)
# How to run: docker run -d --rm --runtime=nvidia FULL_IMAGE_NAME
