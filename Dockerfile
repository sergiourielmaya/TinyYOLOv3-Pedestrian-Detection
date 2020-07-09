FROM tensorflow/tensorflow:latest-gpu-jupyter

COPY requeriments.txt .

RUN pip install -r requeriments.txt

RUN pip install --upgrade cython

RUN apt update && apt install -y libsm6 libxext6 libxrender1

RUN pip install pycocotools

