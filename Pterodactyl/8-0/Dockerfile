FROM --platform=$TARGETOS/$TARGETARCH amd64/debian

LABEL author="Gabriel BTD (me_gabi)" maintainer="gabriel.btd@agamicloud.tech"
LABEL org.opencontainers.image.source="https://github.com/legabi/containers"
LABEL org.opencontainers.image.description="A Docker image to run PHP 8.0 on Pterodactyl"

# Linux things
ARG DEBIAN_FRONTEND=noninteractive

# Mise à jour et installation des dépendances nécessaires pour PHP
RUN apt-get update && apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    software-properties-common

# Ajout de la clé GPG et du dépôt de PHP
RUN curl -fsSL https://packages.sury.org/php/$(lsb_release -sc)/gpg | apt-key add - \
    && echo "deb https://packages.sury.org/php/ $(lsb_release -sc) main" | tee /etc/apt/sources.list.d/php.list

# Installation de PHP 8.0 et de ses extensions
RUN apt-get update && apt-get install -y \
    php8.0 \
    php8.0-{bcmath,bz2,cgi,cli,common,curl,dev,enchant,fpm,gd,gmp,imap,interbase,intl,json,ldap,mbstring,mysql,odbc,opcache,pgsql,phpdbg,pspell,readline,recode,soap,sodium,sqlite3,sybase,tidy,xml,xmlrpc,xsl,zip}

# Configuration de PHP et autres étapes de configuration
RUN ln -s /usr/bin/php8.0 /usr/bin/php \
    && php -v \
    && useradd -d /home/container -m container


USER container
ENV USER=container HOME=/home/container
WORKDIR /home/container

RUN curl -sS https://getcomposer.org/installer | php -- --install-dir=/usr/local/bin --filename=composer

COPY ../entrypoint.sh /entrypoint.sh
CMD [ "/bin/bash", "/entrypoint.sh" ]