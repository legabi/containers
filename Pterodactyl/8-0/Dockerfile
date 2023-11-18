FROM debian:buster-slim

LABEL author="Gabriel BTD (le_gabi)"
LABEL maintainer="gabriel.btd@agamicloud.tech"
LABEL org.opencontainers.image.source="https://github.com/legabi/containers"
LABEL org.opencontainers.image.description="A Docker image to run PHP 8.0 on Pterodactyl"

# Mise à jour et installation des dépendances nécessaires pour PHP
RUN apt-get update && apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    software-properties-common

# Téléchargement et ajout de la clé GPG pour le dépôt Sury PHP
RUN curl https://packages.sury.org/php/apt.gpg -o /etc/apt/trusted.gpg.d/php.gpg

# Ajout du dépôt Sury PHP dans sources.list.d
RUN echo "deb https://packages.sury.org/php/ $(lsb_release -sc) main" | tee /etc/apt/sources.list.d/php.list

# Installation de PHP 8.0
RUN apt-get update && apt-get install -y php8.0

# Installation des extensions PHP individuellement
RUN apt-get install -y \
    php8.0-{common,cli,gd,mysql,mbstring,bcmath,xml,fpm,curl,zip}

# Configuration de PHP et autres étapes de configuration
RUN ln -s /usr/bin/php8.0 /usr/bin/php \
    && php -v \
    && useradd -d /home/container -m container

USER container
ENV USER=container HOME=/home/container
WORKDIR /home/container

# Installation de Composer
RUN curl -sS https://getcomposer.org/installer | php -- --install-dir=/usr/local/bin --filename=composer

COPY ../entrypoint.sh /entrypoint.sh
CMD [ "/bin/bash", "/entrypoint.sh" ]