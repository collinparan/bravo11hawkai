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
RUN pip install torch
RUN pip install pydantic
RUN pip install langchain
RUN pip -q install git+https://github.com/huggingface/transformers git+https://github.com/huggingface/peft.git
RUN pip install -q datasets loralib sentencepiece
RUN pip install pdfminer
RUN pip install pypdf
RUN pip install huggingface_hub
RUN pip install sentence_transformers
RUN pip install xformers
RUN pip install accelerate>=0.20.3 bitsandbytes

COPY requirements.txt ./
RUN pip install -r ./requirements.txt

COPY ./ ./
