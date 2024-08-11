FROM debian:buster-slim

LABEL author="Gabriel BTD"
LABEL maintainer="gabriel.btd@agamicloud.tech"
LABEL org.opencontainers.image.source="https://github.com/legabi/containers"
LABEL org.opencontainers.image.description="A Docker image to run PHP 5.6 on Pterodactyl"

# Mise à jour et installation des dépendances nécessaires pour PHP
RUN apt-get update && apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    software-properties-common \
    wget

# Téléchargement et ajout de la clé GPG pour le dépôt Sury PHP
RUN wget -qO - https://packages.sury.org/php/apt.gpg | apt-key add -

# Ajout du dépôt Sury PHP dans sources.list.d
RUN echo "deb https://packages.sury.org/php/ $(lsb_release -sc) main" | tee /etc/apt/sources.list.d/php.list

# Installation de PHP 5.6
RUN apt-get update && apt-get install -y php5.6

# Installation des extensions PHP spécifiées
RUN apt-get install -y \
    php5.6-common \
    php5.6-cli \
    php5.6-gd \
    php5.6-mysql \
    php5.6-mbstring \
    php5.6-bcmath \
    php5.6-xml \
    php5.6-fpm \
    php5.6-curl \
    php5.6-zip \
    php5.6-sqlite3 \
    php5.6-ldap

# Création du lien symbolique s'il n'existe pas déjà
RUN if [ ! -f /usr/bin/php ]; then ln -s /usr/bin/php5.6 /usr/bin/php; fi

# Vérification de la version PHP et création de l'utilisateur container
RUN php -v && useradd -d /home/container -m container

# Installation de Composer
RUN curl -sS https://getcomposer.org/installer | php -- --install-dir=/usr/bin --filename=composer

# Changement de l'emplacement des sessions PHP pour /home/container/tmp
RUN sed -i 's/\/var\/lib\/php\/sessions/\/home\/container\/tmp/g' /etc/php/5.6/fpm/php.ini

USER container
ENV USER=container HOME=/home/container
WORKDIR /home/container

# finish
COPY ../entrypoint.sh /entrypoint.sh
CMD [ "/bin/bash", "/entrypoint.sh" ]