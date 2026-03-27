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

## Tecnologias

- Python, pandas, numpy
- scikit-learn (Pipeline, ColumnTransformer, LogisticRegression)
- XGBoost, Optuna
- matplotlib, seaborn
