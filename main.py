import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ─────────────────────────────────────────
# RATE LIMITING
# ─────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ─────────────────────────────────────────
# APP
# ─────────────────────────────────────────
app = FastAPI(
    title="Churn Prediction API",
    description="API para predição de churn de clientes",
    version="2.0.0"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ─────────────────────────────────────────
# MODELO
# ─────────────────────────────────────────
pipeline = joblib.load("model/pipeline_churn.pkl")

# ─────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────
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

class BatchRequest(BaseModel):
    clientes: List[Cliente]

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def preprocessar(dados: pd.DataFrame) -> pd.DataFrame:
    dados = dados.copy()
    dados['SeniorCitizen'] = dados['SeniorCitizen'].astype('object')
    dados['TotalCharges'] = dados['TotalCharges'].astype('float64')
    return dados

# ─────────────────────────────────────────
# ENDPOINTS
# /predict       — predição individual
# /predict_batch — predição em lote; usado pelo bot Telegram que sorteia 10 clientes
#                  e envia todos de uma vez, eliminando overhead de chamadas individuais
# ─────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "API de Churn funcionando"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
@limiter.limit("30/minute")
def predict(request: Request, cliente: Cliente):
    dados = pd.DataFrame([cliente.model_dump()])
    dados = preprocessar(dados)

    probabilidade = pipeline.predict_proba(dados)[0][1]
    churn = probabilidade >= 0.5

    return {
        "churn": bool(churn),
        "probabilidade_churn": round(float(probabilidade), 4)
    }


@app.post("/predict_batch")
@limiter.limit("4/minute")
def predict_batch(request: Request, batch: BatchRequest):
    dados = pd.DataFrame([c.model_dump() for c in batch.clientes])
    dados = preprocessar(dados)

    probabilidades = pipeline.predict_proba(dados)[:, 1]

    resultados = [
        {
            "churn": bool(prob >= 0.5),
            "probabilidade_churn": round(float(prob), 4)
        }
        for prob in probabilidades
    ]

    return {"predicoes": resultados, "total": len(resultados)}
