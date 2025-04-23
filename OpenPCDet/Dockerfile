FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    lsb-release \
    gnupg2 \
    software-properties-common \
    && echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list \
    && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - \
    && apt-get update 

RUN apt-get update && apt-get install -y git python3-pip
RUN apt install -y ros-noetic-desktop-full
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
RUN sudo rosdep init
RUN rosdep update

RUN pip install networkx==2.8.8
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install spconv-cu118
RUN pip install "numpy<2.0"
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install scikit-image==0.18.3
RUN pip install "numba==0.56.4"
RUN pip install SharedArray

RUN git clone https://github.com/open-mmlab/OpenPCDet.git

ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV PYTHONPATH=/OpenPCDet

WORKDIR /OpenPCDet
RUN python3 setup.py develop

COPY ImageSets /OpenPCDet/data/custom/ImageSets
COPY labels    /OpenPCDet/data/custom/labels
COPY points    /OpenPCDet/data/custom/points
COPY __init__.py /OpenPCDet/pcdet/datasets/__init__.py
COPY custom_dataset.py /OpenPCDet/pcdet/datasets/custom/custom_dataset.py
COPY custom_dataset.yaml /OpenPCDet/tools/cfgs/dataset_configs/custom_dataset.yaml
COPY pointpillar.yaml /OpenPCDet/tools/cfgs/custom_models/pointpillar.yaml

RUN pip install tqdm
RUN pip install easydict
RUN pip install pyquaternion
RUN pip install tensorboardX

RUN python3 -m pcdet.datasets.custom.custom_dataset create_custom_infos tools/cfgs/dataset_configs/custom_dataset.yaml