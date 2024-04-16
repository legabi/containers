# Utilisez l'image officielle Python 3.10 comme image de base
FROM python:3.10

# Définissez le répertoire de travail dans le conteneur
WORKDIR /app

# Copiez le fichier requirements.txt dans le répertoire de travail
COPY ./requirements.txt /requirements.txt

# Installez les dépendances spécifiées dans requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Commande par défaut pour exécuter lorsqu'un conteneur basé sur cette image est démarré
CMD ["python", "-c", "print('Python 3.10 container is ready')"]