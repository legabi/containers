# Utilisez l'image officielle Python 3.10 comme image de base
FROM tensorflow/tensorflow:latest-gpu-py3

# Change default shell
RUN chsh -s /bin/bash
ENV SHELL=/bin/bash
RUN rm /bin/sh && ln -s /bin/bash /bin/sh


ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris
ENV TF_ENABLE_ONEDNN_OPTS=1

# RUN 1
LABEL author="Gabriel BTD (le_gabi)"
LABEL maintainer="gabriel.btd@agamicloud.tech"
LABEL org.opencontainers.image.source="https://github.com/legabi/containers"
LABEL org.opencontainers.image.description="A Docker image to AI training (Python 3.10 X tensorflow X huggingface)"

WORKDIR /app

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
RUN apt-get -qy update && apt-get install -y wget curl gnupg2
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN rm cuda-keyring_1.0-1_all.deb

RUN apt-get -qy update && apt-get install -qy \
    man \
    vim \
    nano \
    htop \
    curl \
    wget \
    rsync \
    ca-certificates \
    git \
    zip \
    procps \
    ssh \
    gettext-base \
    transmission-cli \
    && apt-get -qq clean \
    && rm -rf /var/lib/apt/lists/*

# install python3.10
# RUN apt-get update && apt-get install -y software-properties-common curl wget
# RUN add-apt-repository ppa:deadsnakes/ppa
# RUN apt-get update && apt-get install -y python3.10 python3.10-dev python3.10-distutils
# RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 
# RUN wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
# RUN sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit --toolkitpath=/usr/local/cuda-11.8 --override
# RUN rm cuda_11.8.0_520.61.05_linux.run

# # Install drivers
# RUN wget https://fr.download.nvidia.com/tesla/550.54.15/NVIDIA-Linux-x86_64-550.54.15.run
# RUN sh NVIDIA-Linux-x86_64-550.54.15.run --silent --no-kernel-module
# RUN rm NVIDIA-Linux-x86_64-550.54.15.run

# Copiez le fichier requirements.txt dans le répertoire de travail
COPY ../requirements.txt /requirements.txt
COPY ../main.py /main.py

# Installez les dépendances spécifiées dans requirements.txt
RUN python -m pip install --no-cache-dir -r /requirements.txt
# # Nvidia H100
# RUN python3.10 -m pip uninstall -y tensorflow
# RUN python3.10 -m pip install tensorflow[and-cuda]

# Vérif

# Commande par défaut pour exécuter lorsqu'un conteneur basé sur cette image est démarré
CMD ["python3", "/main.py"]