FROM ghcr.io/pterodactyl/installers:debian

LABEL author="Gabriel BTD (le_gabi)"
LABEL maintainer="gabriel.btd@agamicloud.tech"
LABEL org.opencontainers.image.source="https://github.com/legabi/containers"
LABEL org.opencontainers.image.description="A Docker image to run Node.js 18 on Pterodactyl"

# Mise à jour et installation des dépendances nécessaires pour Node.js
RUN apt-get update && apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    software-properties-common

# Téléchargement et ajout de la clé GPG pour le dépôt NodeSource
RUN curl -fsSL https://deb.nodesource.com/gpgkey/nodesource.gpg.key | apt-key add -

# Ajout du dépôt NodeSource pour Node.js 18
RUN echo "deb https://deb.nodesource.com/node_18.x $(lsb_release -sc) main" | tee /etc/apt/sources.list.d/nodesource.list

# Installation de Node.js 18
RUN apt-get update && apt-get install -y nodejs

# Vérification de la version de Node.js et création de l'utilisateur container
RUN node -v && npm -v && useradd -d /home/container -m container

USER container
ENV USER=container HOME=/home/container
WORKDIR /home/container

COPY ../entrypoint.sh /entrypoint.sh
CMD [ "/bin/bash", "/entrypoint.sh" ]
