import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Carrega a pipeline completa (pré-processamento + modelo)
pipeline = joblib.load("model/pipeline_churn.pkl")

app = FastAPI(
    title="Churn Prediction API",
    description="API para predição de churn de clientes",
    version="1.0.0"
)

# Define o schema de entrada — exatamente as features do modelo
class Cliente(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def root():
    return {"status": "API de Churn funcionando"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(cliente: Cliente):
    dados = pd.DataFrame([cliente.model_dump()])
    
    # replica o pré-processamento feito antes da pipeline no treinamento
    dados['SeniorCitizen'] = dados['SeniorCitizen'].astype('object')
    dados['TotalCharges'] = dados['TotalCharges'].astype('float64')

    probabilidade = pipeline.predict_proba(dados)[0][1]
    churn = probabilidade >= 0.5

    return {
        "churn": bool(churn),
        "probabilidade_churn": round(float(probabilidade), 4)
    }