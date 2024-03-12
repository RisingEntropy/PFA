FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
MAINTAINER "Haoyu Deng"

RUN apt update \
    && apt install -y python3.10 \
    && apt install -y python3-pip \
    && apt install ffmpeg libsm6 libxext6 libgl1 -y \
    && python3 -m pip install --upgrade pip

# self-build mirror
RUN pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple\
    && pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 -f http://127.0.0.1:8000/ --trusted-host 127.0.0.1

# Official mirror
#RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

RUN pip install spikingjelly==0.0.0.0.12\
    && pip install spikingjelly numpy==1.22 tensorboard argparse torchtoolbox einops Pillow cupy-cuda11x prefetch_generator opencv-python-headless

