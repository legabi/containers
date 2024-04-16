# Utilisez l'image officielle Python 3.10 comme image de base
FROM nvidia/cuda:11.3.1-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris
ENV TF_ENABLE_ONEDNN_OPTS=1

# RUN 1
LABEL author="Gabriel BTD (le_gabi)"
LABEL maintainer="gabriel.btd@agamicloud.tech"
LABEL org.opencontainers.image.source="https://github.com/legabi/containers"
LABEL org.opencontainers.image.description="A Docker image to AI training (Python 3.10 X tensorflow X huggingface)"

WORKDIR /app

# install python3.10
RUN apt-get update && apt-get install -y software-properties-common curl wget
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.10 python3.10-dev python3.10-distutils
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 
RUN wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.ru
RUN sh cuda_11.8.0_520.61.05_linux.ru --silent --toolkit --toolkitpath=/usr/local/cuda-11.8 --override
RUN rm cuda_11.8.0_520.61.05_linux.ru

# CUDNN
RUN wget https://developer.download.nvidia.com/compute/machine-learning/cudnn/secure/8.2.4/11.3_20211119/cudnn-11.3-linux-x64-v8.2.4.15.tgz
RUN tar -xzvf cudnn-11.3-linux-x64-v8.2.4.15.tgz
RUN cp cuda/include/cudnn*.h /usr/local/cuda-11.8/include
RUN cp cuda/lib64/libcudnn* /usr/local/cuda-11.8/lib64
RUN chmod a+r /usr/local/cuda-11.8/include/cudnn*.h /usr/local/cuda-11.8/lib64/libcudnn*

# Copiez le fichier requirements.txt dans le répertoire de travail
COPY ../requirements.txt /requirements.txt
COPY ../main.py /main.py

# Installez les dépendances spécifiées dans requirements.txt
RUN python3.10 -m pip install --no-cache-dir -r /requirements.txt
# Nvidia H100
RUN python3.10 -m pip uninstall -y tensorflow
RUN python3.10 -m pip install tensorflow[and-cuda]

# Vérif

# Commande par défaut pour exécuter lorsqu'un conteneur basé sur cette image est démarré
CMD ["python3.10", "/main.py"]