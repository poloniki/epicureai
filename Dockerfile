FROM python:3.8-slim
COPY epicureai epicureai/
COPY setup.py setup.py
RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -e .
CMD python epicureai/interface/main.py
