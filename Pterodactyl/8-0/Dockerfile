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
    php8.0-bcmath \
    php8.0-bz2 \
    php8.0-cgi \
    php8.0-cli \
    php8.0-common \
    php8.0-curl \
    php8.0-dev \
    php8.0-enchant \
    php8.0-fpm \
    php8.0-gd \
    php8.0-gmp \
    php8.0-imap \
    php8.0-interbase \
    php8.0-intl \
    php8.0-json \
    php8.0-ldap \
    php8.0-mbstring \
    php8.0-mysql \
    php8.0-odbc \
    php8.0-opcache \
    php8.0-pgsql \
    php8.0-phpdbg \
    php8.0-pspell \
    php8.0-readline \
    php8.0-recode \
    php8.0-soap \
    php8.0-sodium \
    php8.0-sqlite3 \
    php8.0-sybase \
    php8.0-tidy \
    php8.0-xml \
    php8.0-xmlrpc \
    php8.0-xsl \
    php8.0-zip

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