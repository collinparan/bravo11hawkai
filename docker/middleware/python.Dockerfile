FROM nvidia/cuda:11.3.1-base-ubuntu20.04

WORKDIR /middleware

RUN apt -y update \
    && apt install -y software-properties-common \
    && add-apt-repository universe \
    && add-apt-repository ppa:deadsnakes/ppa
RUN apt -y install python3.11
RUN apt -y install python-is-python3
RUN apt -y install python3-pip
RUN apt install command-not-found ufw unattended-upgrades git -y
RUN pip install --upgrade pip 


COPY requirements.txt ./
RUN pip install -r ./requirements.txt

COPY ./ ./
