# Utilisez l'image officielle Python 3.10 comme image de base
FROM ubuntu:20.04

# RUN 1
LABEL author="Gabriel BTD (le_gabi)"
LABEL maintainer="gabriel.btd@agamicloud.tech"
LABEL org.opencontainers.image.source="https://github.com/legabi/containers"
LABEL org.opencontainers.image.description="A Docker image to AI training (Python 3.10 X tensorflow X huggingface)"

WORKDIR /app

# Copiez le fichier requirements.txt dans le répertoire de travail
COPY ../requirements.txt /requirements.txt
COPY ../main.py /main.py

# INSTALL CUDA DRIVER
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 \
    curl \
    ca-certificates \
    && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub | apt-key add - \
    && echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    cuda-drivers \
    && apt-mark hold cuda-drivers \
    && rm -rf /var/lib/apt/lists/*

# Installez les dépendances spécifiées dans requirements.txt
RUN python3.10 -m pip install --no-cache-dir -r /requirements.txt
RUN python3.10 -m pip install tensorflow[and-cuda]

# Vérif

# Commande par défaut pour exécuter lorsqu'un conteneur basé sur cette image est démarré
CMD ["python3.10", "/main.py"]