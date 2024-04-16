# Utilisez l'image officielle Python 3.10 comme image de base
FROM python:3.10

LABEL author="Gabriel BTD (le_gabi)"
LABEL maintainer="gabriel.btd@agamicloud.tech"
LABEL org.opencontainers.image.source="https://github.com/legabi/containers"
LABEL org.opencontainers.image.description="A Docker image to AI training"

# Définissez le répertoire de travail dans le conteneur
USER container
ENV USER=container HOME=/home/container
WORKDIR /home/container

# Copiez le fichier requirements.txt dans le répertoire de travail
COPY ../requirements.txt /requirements.txt

# Installez les dépendances spécifiées dans requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Commande par défaut pour exécuter lorsqu'un conteneur basé sur cette image est démarré
CMD ["python", "-c", "print('Python 3.10 container is ready')"]