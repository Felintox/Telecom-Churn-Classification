# Telecom Customer Churn — Classificação

Modelo de classificação para prever churn de clientes de uma empresa de telecomunicações, permitindo ações proativas de retenção antes do cancelamento.

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
| Logistic Regression | 0.8457 | 0.5669 |
| XGBoost | 0.8259 | 0.5572 |
| Random Forest | 0.8175 | — |

**Modelo final:** Logistic Regression (melhor desempenho em ambas as etapas)

Avaliação no conjunto de teste:

| Métrica | Classe 0 (No Churn) | Classe 1 (Churn) |
|---|---|---|
| Precision | 0.85 | 0.65 |
| Recall | 0.89 | 0.57 |
| F1-score | 0.87 | 0.61 |

---

## Tecnologias

- Python, pandas, numpy
- scikit-learn (Pipeline, ColumnTransformer, LogisticRegression)
- XGBoost, Optuna
- matplotlib, seaborn
