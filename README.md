# Telecom Customer Churn — Classificação + Deploy

Modelo de classificação para prever churn de clientes de uma empresa de telecomunicações, com deploy em produção via API REST — permitindo integração com sistemas de CRM e ações proativas de retenção antes do cancelamento.

**Dataset:** [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7.032 clientes, 20 features.

---

## Problema de Negócio

Reter clientes é mais barato do que adquirir novos. O modelo identifica clientes com alta probabilidade de churn para que a empresa direcione campanhas de retenção de forma eficiente.

A métrica de otimização escolhida foi o **F2-score**, que prioriza o Recall (detecção de churn) sem ignorar completamente a Precision — refletindo o custo assimétrico dos erros: um Falso Negativo (cliente que cancela sem ser detectado) é mais custoso do que um Falso Positivo (campanha desnecessária).

---

## Pipeline

```
Dados brutos
  → Limpeza (tipos, remoção de registros inválidos)
  → Split treino/teste com estratificação
  → EDA (apenas no conjunto de treino)
  → Pipeline sklearn (RobustScaler + OneHotEncoder)
  → Baseline (Logistic Regression, Random Forest, XGBoost)
  → Tuning com Optuna (F2-score, cv=5)
  → Avaliação final no conjunto de teste
  → Serialização da pipeline completa (joblib)
  → API REST com FastAPI
  → Containerização com Docker
  → Deploy no Google Cloud Run
  → Bot Telegram consumindo a API em produção
```

---

## Principais Insights da EDA

- Clientes com contrato `Month-to-month` têm taxa de churn significativamente maior
- `Fiber optic` concentra maior proporção de churn entre os tipos de internet
- Clientes sem serviços adicionais (`OnlineSecurity`, `TechSupport`, etc.) churnam mais
- `Electronic check` como método de pagamento está fortemente associado ao churn
- `tenure` baixo é o principal indicador numérico de churn

---

## Resultados

| Modelo | AUC-ROC (baseline) | F2 (após tuning) |
|---|---|---|
| Logistic Regression | 0.8457 | 0.7269 |
| XGBoost | 0.8259 | 0.7519 |
| Random Forest | 0.8175 | — |

**Modelo final:** XGBoost (melhor F2 após tuning com `scale_pos_weight`)

Avaliação no conjunto de teste:

| Métrica | Classe 0 (No Churn) | Classe 1 (Churn) |
|---|---|---|
| Precision | 0.93 | 0.45 |
| Recall | 0.61 | 0.86 |
| F1-score | 0.74 | 0.59 |

---

## Análise Financeira

Simulação com base nos clientes do conjunto de teste (1.407 clientes), custo de campanha de R$50 por cliente abordado e LTV estimado como `MonthlyCharges × tenure médio dos churns`.

| Cenário | Resultado |
|---|---|
| Sem modelo (empresa não age) | -R$ 500.683 |
| Com modelo (XGBoost tunado) | +R$ 353.953 |
| Modelo perfeito (teto teórico) | +R$ 481.983 |

O modelo detectou **86% dos churns reais** (322 de 374), transformando uma perda de meio milhão em resultado positivo. Atinge **73.4% do teto teórico**.

O trade-off de 384 Falsos Positivos (R$19.200 em campanhas desnecessárias) é justificável dado que apenas 52 clientes escaparam sem ser abordados — coerente com o objetivo de negócio definido.

---

## Deploy

Um modelo que fica restrito a um notebook não gera valor real. O objetivo do deploy é transformar o modelo em um serviço consumível — onde um sistema de CRM, uma plataforma de marketing ou qualquer aplicação pode enviar os dados de um cliente e receber, em tempo real, a probabilidade de churn.

### API REST com FastAPI

A pipeline treinada (pré-processamento + modelo) foi serializada e exposta como API REST usando FastAPI. A API oferece os seguintes endpoints:

- **`GET /`** — health check básico
- **`GET /health`** — status da API
- **`POST /predict`** — recebe os dados de um cliente em JSON e retorna a predição com a probabilidade associada

```bash
# Predição individual
curl -X POST "https://churn-api-xxxxxxxx.us-central1.run.app/predict" \
  -H "Content-Type: application/json" \
  -d '{"gender": "Female", "SeniorCitizen": 0, "tenure": 12, ...}'

# Resposta
{"churn": true, "probabilidade_churn": 0.7821}
```

### Docker

Empacotar a API em um container Docker garante que ela rode de forma idêntica em qualquer ambiente — sem conflitos de dependências, sem o clássico "funciona na minha máquina". A imagem contém tudo que a aplicação precisa: código, dependências e o modelo serializado.

```bash
docker build -t churn-api .
docker run -p 8080:8080 churn-api
```

### Google Cloud Run

O Cloud Run é a escolha natural para este tipo de serviço: escala automaticamente conforme a demanda, cobra apenas pelo tempo de processamento real e não exige gerenciamento de servidores. A API fica disponível em uma URL pública, pronta para ser integrada a qualquer sistema.

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/churn-api
gcloud run deploy churn-api --image gcr.io/PROJECT_ID/churn-api \
  --platform managed --region us-central1 --allow-unauthenticated
```

**API disponível em:** `https://churn-api-729603812414.us-central1.run.app/docs`

### Bot Telegram

Para demonstrar a API em uso real, um bot Telegram consome o endpoint `/predict` e apresenta os resultados de forma interativa. A cada comando `/start`, o bot sorteia clientes do dataset `clientes_demo.csv`, envia as predições para a API e exibe os resultados com navegação por botões.

---

## Estrutura do Projeto

```
Telecom-Churn-Classification/
├── main.py                    # API FastAPI (predict + health check)
├── bot.py                     # Bot Telegram consumindo a API
├── clientes_demo.csv          # Amostra do dataset para demonstração (70 obs.)
├── Modelagem_Churn.py         # Código completo de modelagem (EDA → tuning)
├── Modelagem_Churn.ipynb      # Versão notebook do pipeline de modelagem
├── model/
│   └── pipeline_churn.pkl     # Pipeline serializada (pré-processamento + modelo)
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── requirements.txt
└── Dockerfile
```

---

## Tecnologias

- **Modelagem:** Python, pandas, numpy, scikit-learn, XGBoost, Optuna
- **API:** FastAPI, uvicorn, pydantic
- **Infra:** Docker, Google Cloud Run
- **Integração:** Bot Telegram (python-telegram-bot)
- **Visualização:** matplotlib, seaborn
