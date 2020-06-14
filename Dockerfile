FROM tensorflow/tensorflow:latest-gpu-jupyter

COPY requeriments.txt .

RUN pip install -r requeriments.txt
