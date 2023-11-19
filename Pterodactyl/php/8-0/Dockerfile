FROM debian:buster-slim

LABEL author="Gabriel BTD"
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

# Installation des extensions PHP spécifiées
RUN apt-get install -y \
    php8.0-common \
    php8.0-cli \
    php8.0-gd \
    php8.0-mysql \
    php8.0-mbstring \
    php8.0-bcmath \
    php8.0-xml \
    php8.0-fpm \
    php8.0-curl \
    php8.0-zip \
    php8.0-sqlite3 \
    php8.0-ldap

# Création du lien symbolique s'il n'existe pas déjà
RUN if [ ! -f /usr/bin/php ]; then ln -s /usr/bin/php8.0 /usr/bin/php; fi

# Vérification de la version PHP et création de l'utilisateur container
RUN php -v && useradd -d /home/container -m container

# Installation de Composer
RUN curl -sS https://getcomposer.org/installer | php -- --install-dir=/usr/bin --filename=composer

# Changement de l'emplacement des sessions PHP pour /home/container/tmp
RUN sed -i 's/\/var\/lib\/php\/sessions/\/home\/container\/tmp/g' /etc/php/8.0/fpm/php.ini

USER container
ENV USER=container HOME=/home/container
WORKDIR /home/container

COPY ../entrypoint.sh /entrypoint.sh
CMD [ "/bin/bash", "/entrypoint.sh" ]