# Use an official Ubuntu base image
FROM ubuntu:24.04

# Set non-interactive mode for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install common build and ML dependency packages
RUN apt-get update && \
    apt-get install -y \
        software-properties-common \
        wget curl unzip sudo \
        build-essential git cmake \
        libatlas-base-dev libblas-dev liblapack-dev libopenblas-dev \
        gfortran libsm6 libxext6 libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

# Deterministically install both 3.11 and 3.12, users can choose between them
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        python3.11 python3.11-venv python3.11-dev \
        python3.12 python3.12-venv python3.12-dev && \
    rm -rf /var/lib/apt/lists/*

# Set default python version to 3.12
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 2
# users can switch to 3.11 by running `update-alternatives --set python3 /usr/bin/python3.11`

# you would then
# 1. make a /submission dir 
# 2. copy the submission there
# 3. run bash /submission/reproduce.sh
