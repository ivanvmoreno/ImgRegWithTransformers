FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME="/usr/local/cuda"
ENV TORCH_CUDA_ARCH_LIST="8.6"

RUN rm /etc/apt/sources.list.d/cuda.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
RUN apt-get -y update && apt-get -y install curl
RUN conda config --add channels nvidia
RUN conda config --add channels pytorch-lts
RUN conda env create -f environment.yaml
RUN export CUDA_HOME="/usr/local/cuda"
RUN export TORCH_CUDA_ARCH_LIST="8.6"
