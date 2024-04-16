# Utilisez l'image officielle Python 3.10 comme image de base
FROM nvidia/cuda:11.3.1-base-ubuntu20.04

# RUN 1
LABEL author="Gabriel BTD (le_gabi)"
LABEL maintainer="gabriel.btd@agamicloud.tech"
LABEL org.opencontainers.image.source="https://github.com/legabi/containers"
LABEL org.opencontainers.image.description="A Docker image to AI training (Python 3.10 X tensorflow X huggingface)"

WORKDIR /app

# install python3.10
RUN apt-get update && apt-get install -y python3.10 python3-pip

# Copiez le fichier requirements.txt dans le répertoire de travail
COPY ../requirements.txt /requirements.txt
COPY ../main.py /main.py

# Installez les dépendances spécifiées dans requirements.txt
RUN python3.10 -m pip install --no-cache-dir -r /requirements.txt
RUN python3.10 -m pip install tensorflow[and-cuda]

# Vérif

# Commande par défaut pour exécuter lorsqu'un conteneur basé sur cette image est démarré
CMD ["python3.10", "/main.py"]