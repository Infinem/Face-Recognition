FROM continuumio/miniconda3


RUN pip install flask
RUN pip install sklearn
RUN pip install pillow
RUN pip install opencv-python
RUN pip install scikit-image
RUN pip install keras
RUN pip install tensorflow==1.13.1

RUN apt-get update
RUN apt-get install -y cmake
RUN apt-get install -y build-essential
RUN pip install face-recognition

RUN apt-get install -y software-properties-common
RUN apt-get install -y libsm6 libxrender1 libxext6

RUN mkdir -p /app
COPY . /app
WORKDIR /app
CMD python server.py