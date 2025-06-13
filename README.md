# API de Prédiction de Churn - Fortuneo Banque

Cette API permet de prédire le risque de désabonnement (churn) des clients de Fortuneo Banque en utilisant un modèle de Machine Learning.

## Fonctionnalités

- **Prédiction individuelle** : Évaluation du risque de churn pour un client spécifique
- **Prédiction par lot** : Analyse du risque pour un groupe de clients
- **Statistiques** : Données agrégées sur les taux de churn par différentes dimensions

## Installation

1. Clonez ce dépôt ou téléchargez les fichiers
2. Installez les dépendances :

```bash
pip install -r requirements.txt
```

## Lancement de l'API

```bash
cd app
uvicorn main:app --host 0.0.0.0 --port 8000
```

L'API sera accessible à l'adresse : http://localhost:8000
dans notre cas après l'avoir déployé on peut y accéder sur : https://machinelearning2api.onrender.com

## Documentation interactive

Une fois l'API lancée, vous pouvez accéder à la documentation interactive Swagger UI à l'adresse :
http://localhost:8000/docs
dans notre cas après l'avoir déployé on peut y accéder sur : https://machinelearning2api.onrender.com/docs#/
## Endpoints

### Statut de l'API

```
GET /
```

Retourne l'état de l'API et la liste des endpoints disponibles.

### Prédiction individuelle

```
POST /predict
```

Prédit la probabilité de churn pour un client individuel.

**Exemple de requête :**

```json
{
  "CreditScore": 619,
  "Geography": "France",
  "Gender": "Female",
  "Age": 42,
  "Tenure": 2,
  "Balance": 0.00,
  "NumOfProducts": 1,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 101348.88
}
```

**Exemple de réponse :**

```json
{
  "client_id": 0,
  "churn_probability": 0.12,
  "churn_prediction": false,
  "risk_level": "Faible"
}
```

### Prédiction par lot

```
POST /predict/batch
```

Prédit la probabilité de churn pour un groupe de clients.

**Exemple de requête :**

```json
{
  "clients": [
    {
      "CreditScore": 619,
      "Geography": "France",
      "Gender": "Female",
      "Age": 42,
      "Tenure": 2,
      "Balance": 0.00,
      "NumOfProducts": 1,
      "HasCrCard": 1,
      "IsActiveMember": 1,
      "EstimatedSalary": 101348.88
    },
    {
      "CreditScore": 608,
      "Geography": "Spain",
      "Gender": "Female",
      "Age": 41,
      "Tenure": 1,
      "Balance": 83807.86,
      "NumOfProducts": 1,
      "HasCrCard": 0,
      "IsActiveMember": 1,
      "EstimatedSalary": 112542.58
    }
  ]
}
```

**Exemple de réponse :**

```json
{
  "predictions": [
    {
      "client_id": 0,
      "churn_probability": 0.12,
      "churn_prediction": false,
      "risk_level": "Faible"
    },
    {
      "client_id": 1,
      "churn_probability": 0.45,
      "churn_prediction": false,
      "risk_level": "Moyen"
    }
  ],
  "summary": {
    "total_clients": 2,
    "high_risk": 0,
    "high_risk_percent": 0.0,
    "medium_risk": 1,
    "medium_risk_percent": 50.0,
    "low_risk": 1,
    "low_risk_percent": 50.0
  }
}
```

### Statistiques

```
GET /statistics
```

Retourne des statistiques sur les données de churn.

**Exemple de réponse :**

```json
{
  "total_clients": 10000,
  "churn_rate": 20.37,
  "churn_by_country": {
    "France": {
      "count": 5014,
      "churn_rate": 16.2
    },
    "Germany": {
      "count": 2509,
      "churn_rate": 32.4
    },
    "Spain": {
      "count": 2477,
      "churn_rate": 16.7
    }
  },
  "churn_by_gender": {
    "Female": {
      "count": 4543,
      "churn_rate": 25.1
    },
    "Male": {
      "count": 5457,
      "churn_rate": 16.5
    }
  },
  "churn_by_age_group": {
    "<30": {
      "count": 1538,
      "churn_rate": 10.2
    },
    "30-40": {
      "count": 3213,
      "churn_rate": 15.8
    },
    "40-50": {
      "count": 3042,
      "churn_rate": 21.3
    },
    "50-60": {
      "count": 1666,
      "churn_rate": 28.7
    },
    ">60": {
      "count": 541,
      "churn_rate": 42.1
    }
  },
  "churn_by_products": {
    "1": {
      "count": 5084,
      "churn_rate": 17.2
    },
    "2": {
      "count": 4590,
      "churn_rate": 19.8
    },
    "3": {
      "count": 266,
      "churn_rate": 69.5
    },
    "4": {
      "count": 60,
      "churn_rate": 85.0
    }
  },
  "timestamp": "2025-06-03T09:55:00.123456"
}
```

## Intégration avec l'application Streamlit

Cette API est conçue pour être utilisée avec l'application Streamlit de prédiction de churn de Fortuneo Banque. Pour intégrer l'API à l'application, consultez le guide d'utilisation fourni.
vous pouvez voir le rendu avec une application que nous avons déployé sur https://application-de-churn-fortuneo.streamlit.app

## Source des données

- Base de données : [GitHub](https://github.com/Awoutokoffisamson/machine_learning2_Documents/blob/main/Churn_Modelling.csv)
- Modèle : [Google Drive](https://drive.google.com/file/d/1aSFJ-Vc9VsX1dH-dYAiKVxX1SPdARZMX/view)

