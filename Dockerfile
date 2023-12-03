FROM ultralytics/ultralytics:latest
COPY setup.py setup.py
RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -e .
COPY epicureai epicureai/
CMD python epicureai/interface/main.py
