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
  → Baseline (Logistic Regression, Random Forest, XGBoost) — avaliado em F2 e AUC-ROC
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
- Segmento **Month-to-month + Fiber optic** concentra 55% de taxa de churn — o de maior risco na base
- Clientes sem serviços adicionais (`OnlineSecurity`, `TechSupport`, etc.) churnam mais
- `Electronic check` como método de pagamento está fortemente associado ao churn
- `tenure` baixo é o principal indicador numérico de churn

---

## Resultados

### Baseline

| Modelo | AUC-ROC | F2 |
|---|---|---|
| Logistic Regression | 0.8457 | 0.5651 |
| XGBoost | 0.8259 | 0.5244 |
| Random Forest | 0.8175 | 0.5107 |

O baseline foi avaliado com **F2 e AUC-ROC** — garantindo que a seleção dos modelos para tuning seja consistente com a métrica de otimização do projeto. Logistic Regression e XGBoost seguiram para o tuning com Optuna.

### Após Tuning (Optuna, cv=5)

| Modelo | F2 |
|---|---|
| Logistic Regression | 0.7283 |
| XGBoost | 0.7517 ← modelo escolhido |

**Modelo final:** XGBoost com `scale_pos_weight` tunado — permite ao modelo penalizar mais os erros na classe minoritária (churn).

### Avaliação no Conjunto de Teste

<img width="507" height="453" alt="download" src="https://github.com/user-attachments/assets/8cf47f6c-6472-49d4-a60c-eec095538f81" />

| Métrica | Classe 0 (No Churn) | Classe 1 (Churn) |
|---|---|---|
| Precision | 0.93 | 0.44 |
| Recall | 0.59 | 0.87 |
| F1-score | 0.73 | 0.59 |

O modelo detectou **324 dos 374 churns reais** (Recall 87%) — trade-off coerente com o F2-score, que prioriza não deixar churners passarem despercebidos.

---

## Feature Importance

### Importância Nativa do XGBoost (Top 15)

<img width="990" height="590" alt="download" src="https://github.com/user-attachments/assets/569a3e3b-018c-4339-9152-26419534064d" />


- `Contract_Month-to-month` domina com folga (~0.35) — ser cliente mensal é de longe o maior preditor de churn
- `OnlineSecurity_No` e `InternetService_Fiber optic` aparecem em seguida (~0.06)
- Os dois outros tipos de contrato (`Two year`, `One year`) entram no top 5 — confirmando que o tipo de contrato é a variável mais relevante
- `tenure` aparece apenas na 11ª posição — relevante, mas menos do que as categóricas

### SHAP — Explicabilidade

<img width="790" height="940" alt="download" src="https://github.com/user-attachments/assets/82876b4b-2f05-4d1f-9891-cc42ad820477" />


- `Contract_Month-to-month`: clientes com contrato mensal têm forte impacto positivo no churn
- `tenure` baixo aumenta o churn, tenure alto reduz — clientes antigos são mais fiéis
- `OnlineSecurity_No` e `PaymentMethod_Electronic check`: ausência de segurança online e pagamento por cheque eletrônico aumentam o churn
- `Contract_Two year`: contrato longo reduz fortemente o churn

---

## Análise Financeira

Simulação com base nos clientes do conjunto de teste (1.407 clientes), custo de campanha de R$50 por cliente abordado e LTV estimado como `MonthlyCharges × tenure médio dos churners (18.2 meses)`.

| Cenário | Resultado |
|---|---|
| Sem modelo (empresa não age) | -R$ 500.683 |
| Com modelo (XGBoost tunado) | +R$ 140.184 |

O modelo detectou **324 de 374 churns reais**, gerando R$ 176.234 em receita salva (324 VP × 40% de taxa de retenção × LTV estimado), com custo de R$ 36.050 em campanhas (721 clientes abordados × R$50).

**Taxa de retenção de 40%:** referência conservadora-realista para o setor de telecom (campanhas de retenção têm sucesso em 20–40% dos casos).

---

## Deploy

Um modelo restrito a um notebook não gera valor real. O objetivo do deploy é transformá-lo em um serviço consumível — onde qualquer sistema pode enviar os dados de um cliente e receber, em tempo real, a probabilidade de churn.

### API REST com FastAPI

A pipeline treinada (pré-processamento + modelo) foi serializada e exposta como API REST usando FastAPI:

- **`POST /predict`** — recebe um único cliente e retorna a predição com probabilidade
- **`POST /predict_batch`** — recebe N clientes de uma vez, retorna todas as predições em uma única chamada. Usado pelo bot Telegram, que sorteia 10 clientes e envia todos de uma vez, eliminando o overhead de múltiplas requisições

Ambos os endpoints têm **rate limiting** via `slowapi`: `/predict` aceita até 30 req/min por IP, `/predict_batch` até 4/min.

```bash
# Predição individual
curl -X POST ".../predict" \
  -H "Content-Type: application/json" \
  -d '{"gender": "Female", "SeniorCitizen": 0, "tenure": 12, ...}'

# Resposta
{"churn": true, "probabilidade_churn": 0.7821}

# Predição em lote
curl -X POST ".../predict_batch" \
  -H "Content-Type: application/json" \
  -d '{"clientes": [{...}, {...}]}'

# Resposta
{"predicoes": [{"churn": true, "probabilidade_churn": 0.7821}, ...], "total": 10}
```

### Docker

Empacotar a API em um container Docker garante que ela rode de forma idêntica em qualquer ambiente — sem conflitos de dependências, sem o clássico "funciona na minha máquina". A imagem contém tudo que a aplicação precisa: código, dependências Python e o modelo serializado.

```bash
docker build -t churn-api .
docker run -p 8080:8080 churn-api
```

### Google Cloud Platform (GCP)

O deploy utiliza três serviços do GCP:

**Artifact Registry** — repositório privado de imagens Docker. O Cloud Build envia a imagem gerada pelo Dockerfile diretamente para o Artifact Registry, que a armazena e versiona.

```bash
gcloud builds submit --tag us-central1-docker.pkg.dev/PROJECT_ID/churn-api/churn-api
```

**Cloud Run** — serviço serverless que executa a imagem do Artifact Registry como um container. Escala automaticamente conforme a demanda, cobra apenas pelo tempo de processamento real e não exige gerenciamento de servidores. A API FastAPI roda aqui.

```bash
gcloud run deploy churn-api \
  --image us-central1-docker.pkg.dev/PROJECT_ID/churn-api/churn-api \
  --platform managed --region us-central1 --allow-unauthenticated
```

**Compute Engine** — máquina virtual onde o bot Telegram fica hospedado. Uma instância de baixo custo é suficiente para rodar o processo do bot continuamente, já que ele apenas consome a API do Cloud Run e responde aos usuários.

### Bot Telegram

Para demonstrar a API em uso real, um bot Telegram consome o endpoint `/predict_batch`. A cada comando `/start`, o bot sorteia 10 clientes do dataset, envia todos de uma vez para a API e exibe os resultados com navegação por botões.

**Bot:** [@churn_predictor_gfx_bot](https://web.telegram.org/k/#@churn_predictor_gfx_bot)

---

## Estrutura do Projeto

```
Telecom-Churn-Classification/
├── main.py                    # API FastAPI (predict + predict_batch + rate limiting)
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

- **Modelagem:** Python, pandas, numpy, scikit-learn, XGBoost, Optuna, SHAP
- **API:** FastAPI, uvicorn, pydantic, slowapi
- **Infra:** Docker, Google Cloud Run, Google Artifact Registry, Google Compute Engine
- **Integração:** Bot Telegram (python-telegram-bot)
- **Visualização:** matplotlib, seaborn
