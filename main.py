from fastapi import FastAPI, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import pandas as pd
import numpy as np
import pickle
import requests
import io
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import uvicorn
import os
from datetime import datetime

# Création de l'application FastAPI
app = FastAPI(
    title="Fortuneo Banque - API de Prédiction de Churn",
    description="API pour la prédiction de désabonnement des clients de Fortuneo Banque",
    version="1.0.0"
)

# Configuration CORS pour permettre les requêtes cross-origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autorise toutes les origines en développement
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# URL des ressources
GITHUB_DATA_URL = "https://raw.githubusercontent.com/Awoutokoffisamson/machine_learning2_Documents/main/Churn_Modelling.csv"
DRIVE_MODEL_ID = "1JZji6K_r-Msko1xuk3R9ONycgPtliSK2" 

# Variables globales pour stocker les données et le modèle
data = None
model = None

# Modèles Pydantic pour la validation des données
class ClientData(BaseModel):
    CreditScore: int = Field(..., description="Score de crédit du client", example=619)
    Geography: str = Field(..., description="Pays du client", example="France")
    Gender: str = Field(..., description="Genre du client", example="Female")
    Age: int = Field(..., description="Âge du client", example=42)
    Tenure: int = Field(..., description="Nombre d'années en tant que client", example=2)
    Balance: float = Field(..., description="Solde du compte", example=0.00)
    NumOfProducts: int = Field(..., description="Nombre de produits bancaires", example=1)
    HasCrCard: int = Field(..., description="Possède une carte de crédit (0=Non, 1=Oui)", example=1)
    IsActiveMember: int = Field(..., description="Est un membre actif (0=Non, 1=Oui)", example=1)
    EstimatedSalary: float = Field(..., description="Salaire estimé", example=101348.88)

class BatchPredictionRequest(BaseModel):
    clients: List[ClientData] = Field(..., description="Liste des données clients pour prédiction par lot")

class PredictionResponse(BaseModel):
    client_id: int = Field(..., description="Identifiant du client")
    churn_probability: float = Field(..., description="Probabilité de désabonnement")
    churn_prediction: bool = Field(..., description="Prédiction de désabonnement (True=Churné, False=Fidèle)")
    risk_level: str = Field(..., description="Niveau de risque (Faible, Moyen, Élevé)")

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse] = Field(..., description="Liste des prédictions pour chaque client")
    summary: Dict = Field(..., description="Résumé des prédictions (nombre de clients à risque élevé, moyen, faible)")

class StatisticsResponse(BaseModel):
    total_clients: int = Field(..., description="Nombre total de clients")
    churn_rate: float = Field(..., description="Taux de churn global")
    churn_by_country: Dict = Field(..., description="Taux de churn par pays")
    churn_by_gender: Dict = Field(..., description="Taux de churn par genre")
    churn_by_age_group: Dict = Field(..., description="Taux de churn par tranche d'âge")
    churn_by_products: Dict = Field(..., description="Taux de churn par nombre de produits")
    timestamp: str = Field(..., description="Horodatage de la génération des statistiques")

# Fonction pour charger les données depuis GitHub
def load_data():
    global data
    if data is None:
        try:
            response = requests.get(GITHUB_DATA_URL)
            response.raise_for_status()
            data_io = io.StringIO(response.text)
            data = pd.read_csv(data_io)
            return data
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erreur lors du chargement des données: {str(e)}")
    return data

# Fonction pour charger ou entraîner le modèle
def load_or_train_model():
    global model
    if model is None:
        try:
            # Essayer de charger le modèle depuis Google Drive
            try:
                response = requests.get(f"https://drive.google.com/uc?export=download&id={DRIVE_MODEL_ID}")
                response.raise_for_status()
                model = pickle.loads(response.content)
                return model
            except Exception as e:
                print(f"Erreur lors du chargement du modèle depuis Drive: {str(e)}")
                print("Entraînement d'un nouveau modèle...")
                
                # Si échec, entraîner un nouveau modèle
                df = load_data()
                
                # Préparation des données
                X = df.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
                y = df['Exited']
                
                # Définition des colonnes catégorielles et numériques
                categorical_cols = ['Geography', 'Gender']
                numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
                

                # Création du préprocesseur
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', StandardScaler(), numerical_cols),
                        ('cat', OneHotEncoder(drop='first'), categorical_cols)
                    ])
                
                # Création du pipeline avec préprocesseur et modèle
                pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
                ])
                
                # Entraînement du modèle
                pipeline.fit(X, y)
                model = pipeline
                
                return model
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erreur lors du chargement/entraînement du modèle: {str(e)}")
    return model

# Endpoint pour vérifier l'état de l'API
@app.get("/", tags=["Statut"])
async def root():
    return {
        "status": "online",
        "api": "Fortuneo Banque - API de Prédiction de Churn",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "Prédiction pour un client individuel",
            "/predict/batch": "Prédiction pour un groupe de clients",
            "/statistics": "Statistiques sur les données de churn"
        }
    }

# Endpoint pour la prédiction individuelle
@app.post("/predict", response_model=PredictionResponse, tags=["Prédiction"])
async def predict_churn(client: ClientData):
    try:
        # Charger ou entraîner le modèle
        model = load_or_train_model()
        
        # Convertir les données client en DataFrame
        client_df = pd.DataFrame([client.dict()])
        
        # Faire la prédiction
        churn_proba = model.predict_proba(client_df)[0, 1]
        churn_pred = bool(model.predict(client_df)[0])
        
        # Déterminer le niveau de risque
        if churn_proba < 0.3:
            risk_level = "Faible"
        elif churn_proba < 0.7:
            risk_level = "Moyen"
        else:
            risk_level = "Élevé"
        
        # Retourner la réponse
        return {
            "client_id": 0,  # ID générique pour prédiction individuelle
            "churn_probability": float(churn_proba),
            "churn_prediction": churn_pred,
            "risk_level": risk_level
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")

# Endpoint pour la prédiction par lot
@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prédiction"])
async def predict_batch(batch: BatchPredictionRequest):
    try:
        # Charger ou entraîner le modèle
        model = load_or_train_model()
        
        # Convertir les données clients en DataFrame
        clients_df = pd.DataFrame([client.dict() for client in batch.clients])
        
        # Faire les prédictions
        churn_probas = model.predict_proba(clients_df)[:, 1]
        churn_preds = model.predict(clients_df)
        
        # Préparer les résultats
        predictions = []
        high_risk = 0
        medium_risk = 0
        low_risk = 0
        
        for i, (proba, pred) in enumerate(zip(churn_probas, churn_preds)):
            # Déterminer le niveau de risque
            if proba < 0.3:
                risk_level = "Faible"
                low_risk += 1
            elif proba < 0.7:
                risk_level = "Moyen"
                medium_risk += 1
            else:
                risk_level = "Élevé"
                high_risk += 1
            
            predictions.append({
                "client_id": i,
                "churn_probability": float(proba),
                "churn_prediction": bool(pred),
                "risk_level": risk_level
            })
        
        # Résumé des prédictions
        summary = {
            "total_clients": len(predictions),
            "high_risk": high_risk,
            "high_risk_percent": round(high_risk / len(predictions) * 100, 1),
            "medium_risk": medium_risk,
            "medium_risk_percent": round(medium_risk / len(predictions) * 100, 1),
            "low_risk": low_risk,
            "low_risk_percent": round(low_risk / len(predictions) * 100, 1)
        }
        
        return {
            "predictions": predictions,
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction par lot: {str(e)}")

# Endpoint pour les statistiques
@app.get("/statistics", response_model=StatisticsResponse, tags=["Statistiques"])
async def get_statistics():
    try:
        # Charger les données
        df = load_data()
        
        # Calculer les statistiques
        total_clients = len(df)
        churn_rate = df['Exited'].mean() * 100
        
        # Taux de churn par pays
        churn_by_country = df.groupby('Geography')['Exited'].agg(['count', 'mean']).reset_index()
        churn_by_country.columns = ['country', 'count', 'churn_rate']
        churn_by_country['churn_rate'] = churn_by_country['churn_rate'] * 100
        churn_by_country_dict = {row['country']: {
            'count': int(row['count']),
            'churn_rate': float(row['churn_rate'])
        } for _, row in churn_by_country.iterrows()}
        
        # Taux de churn par genre
        churn_by_gender = df.groupby('Gender')['Exited'].agg(['count', 'mean']).reset_index()
        churn_by_gender.columns = ['gender', 'count', 'churn_rate']
        churn_by_gender['churn_rate'] = churn_by_gender['churn_rate'] * 100
        churn_by_gender_dict = {row['gender']: {
            'count': int(row['count']),
            'churn_rate': float(row['churn_rate'])
        } for _, row in churn_by_gender.iterrows()}
        
        # Taux de churn par tranche d'âge
        df['Age_Group'] = pd.cut(
            df['Age'],
            bins=[0, 30, 40, 50, 60, 100],
            labels=['<30', '30-40', '40-50', '50-60', '>60']
        )
        churn_by_age = df.groupby('Age_Group')['Exited'].agg(['count', 'mean']).reset_index()
        churn_by_age.columns = ['age_group', 'count', 'churn_rate']
        churn_by_age['churn_rate'] = churn_by_age['churn_rate'] * 100
        churn_by_age_dict = {str(row['age_group']): {
            'count': int(row['count']),
            'churn_rate': float(row['churn_rate'])
        } for _, row in churn_by_age.iterrows()}
        
        # Taux de churn par nombre de produits
        churn_by_products = df.groupby('NumOfProducts')['Exited'].agg(['count', 'mean']).reset_index()
        churn_by_products.columns = ['num_products', 'count', 'churn_rate']
        churn_by_products['churn_rate'] = churn_by_products['churn_rate'] * 100
        churn_by_products_dict = {str(row['num_products']): {
            'count': int(row['count']),
            'churn_rate': float(row['churn_rate'])
        } for _, row in churn_by_products.iterrows()}
        
        return {
            "total_clients": total_clients,
            "churn_rate": float(churn_rate),
            "churn_by_country": churn_by_country_dict,
            "churn_by_gender": churn_by_gender_dict,
            "churn_by_age_group": churn_by_age_dict,
            "churn_by_products": churn_by_products_dict,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération des statistiques: {str(e)}")

# Point d'entrée pour exécuter l'API directement
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload= False)
