#FROM tensorflow/tensorflow:latest-gpu-jupyter
#FROM tensorflow/tensorflow:nightly-gpu-jupyter
FROM tensorflow/tensorflow:2.3.0-gpu-jupyter 
COPY requeriments.txt .

RUN apt update && apt install -y libsm6 libxext6 libxrender1 libgl1-mesa-glx

RUN pip install -r requeriments.txt

RUN pip install --upgrade cython



RUN pip install pycocotools
