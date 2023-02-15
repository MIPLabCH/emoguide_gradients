# syntax=docker/dockerfile:1

FROM ubuntu:20.04
RUN apt-get update && apt-get -y upgrade \
  && apt-get install -y --no-install-recommends \
    git \
    wget \
    g++ \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

COPY requirements.txt requirements.txt

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && echo "Running $(conda --version)" && \
    conda init bash && \
    . /root/.bashrc && \
    conda update conda && \
    conda create -n brain && \
    conda activate brain && \
    conda install python=3.8.13 pip && \
    conda install vtk --y &&\
    pip3 install --upgrade pip && pip3 install packaging &&\
    pip3 install -r requirements.txt &&\
    echo 'print("Hello World!")'


RUN git clone https://github.com/MICA-MNI/ENIGMA.git  && cd ENIGMA && \
	python setup.py install


WORKDIR /home
COPY . .




