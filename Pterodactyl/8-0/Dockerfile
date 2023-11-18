FROM --platform=$TARGETOS/$TARGETARCH amd64/debian

LABEL author="Gabriel BTD (me_gabi)" maintainer="gabriel.btd@agamicloud.tech"
LABEL org.opencontainers.image.source="https://github.com/legabi/containers"
LABEL org.opencontainers.image.description="A Docker image to run PHP 8.0 on Pterodactyl"

# Linux things
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y \
    && apt-get upgrade -y \
    && apt-get install -y lsof curl ca-certificates openssl git tar sqlite3 fontconfig libfreetype6 tzdata iproute2 libstdc++6 locales locales-all font-manager php8.0


# PHP things
RUN apt-get install -y php8.0-{bcmath,bz2,cgi,cli,common,curl,dev,enchant,fpm,gd,gmp,imap,interbase,intl,json,ldap,mbstring,mysql,odbc,opcache,pgsql,phpdbg,pspell,readline,recode,soap,sodium,sqlite3,sybase,tidy,xml,xmlrpc,xsl,zip}

# export PATH="$PATH:/usr/bin/php8.0"
RUN ln -s /usr/bin/php8.0 /usr/bin/php

RUN php -v

# Pterodactyl things
RUN useradd -d /home/container -m container

USER container
ENV USER=container HOME=/home/container
WORKDIR /home/container

RUN curl -sS https://getcomposer.org/installer | php -- --install-dir=/usr/local/bin --filename=composer

COPY ../entrypoint.sh /entrypoint.sh
CMD [ "/bin/bash", "/entrypoint.sh" ]