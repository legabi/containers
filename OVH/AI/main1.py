# /waf_ai/data_generation.py

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

def generate_attack_data(num_samples=1000):
    """
    Génère des données factices à partir de deux tableaux : un contenant des valeurs sûres,
    l'autre contenant des valeurs malveillantes.
    """

    # Définir des tableaux pour les caractéristiques
    safe_user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Safari/605.1.15",
    ]
    malicious_user_agents = [
        "sqlmap/1.5.2",
        "python-requests/2.25.1",
    ]

    safe_urls = [
        "/index.html",
        "/about-us",
        "/contact",
    ]
    malicious_urls = [
        "/login.php?id=1' OR '1'='1",
        "/search?q=<script>alert('XSS')</script>",
    ]

    safe_query_strings = [
        "id=123&name=john",
        "product=456&category=books",
    ]
    malicious_query_strings = [
        "id=1' UNION SELECT * FROM users",
        "q=<script>alert('XSS')</script>",
    ]

    # Combiner les tableaux pour créer des exemples sûrs et malveillants
    data = []
    labels = []

    for _ in range(num_samples):
        is_safe = np.random.choice([0, 1])  # 0 = malveillant, 1 = sûr

        # User-Agent
        user_agent = (
            np.random.choice(safe_user_agents)
            if is_safe
            else np.random.choice(malicious_user_agents)
        )

        # URL
        url = (
            np.random.choice(safe_urls)
            if is_safe
            else np.random.choice(malicious_urls)
        )

        # Query String
        query_string = (
            np.random.choice(safe_query_strings)
            if is_safe
            else np.random.choice(malicious_query_strings)
        )

        # Score basé sur IP (aléatoire, mais pondéré)
        ip_score = np.random.uniform(0.5, 1) if is_safe else np.random.uniform(0, 0.5)

        # Nombre de requêtes par seconde
        request_rate = (
            np.random.randint(1, 10) if is_safe else np.random.randint(20, 50)
        )

        # Générer les caractéristiques
        features = [
            ip_score,
            len(url),  # Longueur de l'URL
            len(query_string.split("&")),  # Nombre de paramètres
            len(user_agent),  # Longueur du User-Agent
            request_rate,
            int("<script>" in query_string or "UNION" in query_string.upper()),
        ]

        data.append(features)
        labels.append(is_safe)

    df = pd.DataFrame(
        data,
        columns=[
            "IP_Score",
            "URL_Length",
            "Num_Params",
            "User_Agent_Length",
            "Request_Rate",
            "Has_Malicious_Keywords",
        ],
    )

    return df, labels


def train_model(num_iterations=3, batch_size=32, epochs=10):
    """
    Entraîne un modèle TensorFlow basé sur les données générées.
    """
    # Génération des données
    df, labels = generate_attack_data(num_samples=2000)
    X = df.values
    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Création du modèle
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(512, activation="relu", input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Entraînement avec tqdm pour afficher la progression
    for iteration in tqdm(range(num_iterations), desc="Training iterations"):
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2)

    # Évaluation finale
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Final Accuracy: {accuracy:.2f}")

    model.summary()

    # Sauvegarde du modèle
    model.save("waf_ai_model_tf.h5")
    return model


def detect_attack(input_data):
    """
    Détecte une attaque à partir d'une requête HTTP.
    :param input_data: dict contenant les informations de la requête.
    :return: tuple (decision, reason_code)
    """
    model = tf.keras.models.load_model("waf_ai_model_tf.h5")

    # Extraction des caractéristiques
    features = np.array(
        [
            [
                input_data.get("IP_Score", 0.5),
                len(input_data.get("URL", "")),
                input_data.get("Num_Params", 0),
                len(input_data.get("User_Agent", "")),
                input_data.get("Request_Rate", 1),
                int(
                    "<script>" in input_data.get("Query", "").lower()
                    or "union" in input_data.get("Query", "").lower()
                ),
            ]
        ]
    )

    # Prédiction
    prediction = model.predict(features)[0][0]
    decision = prediction > 0.5
    reason_code = 1001 if features[0, -1] == 1 else 1002 if features[0, 4] > 10 else 1000
    return decision, reason_code


if __name__ == "__main__":
    # Entraîner le modèle
    model = train_model(num_iterations=5, batch_size=64, epochs=50)

    # Exemple de détection
    example_request = {
        "IP_Score": 0.1,
        "URL": "/login.php?id=1' OR '1'='1",
        "Num_Params": 3,
        "User_Agent": "sqlmap/1.5.2",
        "Request_Rate": 20,
        "Query": "id=1' UNION SELECT * FROM users",
    }

    decision, reason = detect_attack(example_request)
    print(f"Decision: {'Block' if decision else 'Allow'}, Reason Code: {reason}")
