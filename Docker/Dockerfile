FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get -y install git

RUN pip install opencv-python matplotlib kornia chumpy ConfigArgParse face-alignment face-detection-tflite mediapipe wandb
WORKDIR /app

RUN pip install fvcore iopath
RUN pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html

RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install librosa transformers
RUN pip install torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install moviepy imageio
RUN git config --global --add safe.directory /app

RUN apt-get install espeak -y
RUN pip install phonemizer
RUN pip install pytorch-lightning omegaconf scikit-image