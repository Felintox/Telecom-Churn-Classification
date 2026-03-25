# Projeto Churn - Telecom Customer

## Contexto
Construir um modelo de classificação para prever churn de clientes de uma empresa de telecomunicações, permitindo que a empresa tome ações proativas de retenção antes que o cliente cancele.

**Problema de negócio:** Reter clientes é mais barato do que adquirir novos. Identificar clientes com alta probabilidade de churn permite campanhas de retenção direcionadas e eficientes.

---

## To-Do List

### Fase 1 — Entendimento e Preparação dos Dados
- [x] Importação das bibliotecas
- [x] Leitura do dataset
- [x] Verificação do shape, tipos e nulos (`data.info()`)
- [x] Remoção da coluna `customerID` (sem valor preditivo)
- [x] Conversão de `SeniorCitizen` para `object` (é categórica)
- [x] Identificação e remoção dos 11 registros com `TotalCharges == ' '`
- [x] Conversão de `TotalCharges` para `float64`
- [x] Conversão de `Churn` para binário (`Yes` → 1, `No` → 0)
- [x] Análise da distribuição da variável alvo
- [x] Split treino/teste com estratificação (`stratify=y`)

### Fase 2 — Análise Exploratória dos Dados (EDA)
- [x] Análise univariada — variáveis categóricas (gráficos de barra)
- [x] Análise univariada — variáveis numéricas (histogramas)
- [x] Verificação de correlações entre variáveis numéricas (`corr()`)
- [ ] Análise bivariada — relação das features com Churn

### Fase 3 — Pré-processamento (Pipeline)
- [ ] Definir variáveis categóricas e numéricas
- [ ] Encoding das variáveis categóricas (OneHotEncoder ou OrdinalEncoder)
- [ ] Scaling das variáveis numéricas (StandardScaler ou MinMaxScaler)
- [ ] Montar ColumnTransformer + Pipeline sklearn

### Fase 4 — Modelagem
- [ ] Definir métricas de avaliação (F1, AUC-ROC, Recall)
- [ ] Treinar modelos baseline (Logistic Regression, Decision Tree)
- [ ] Treinar modelos mais robustos (Random Forest, XGBoost, LightGBM)
- [ ] Validação cruzada (cross_val_score)
- [ ] Comparação de desempenho entre modelos

### Fase 5 — Otimização
- [ ] Tuning de hiperparâmetros (GridSearchCV ou RandomizedSearchCV)
- [ ] Análise de feature importance
- [ ] Threshold tuning (ajuste do ponto de corte da probabilidade)

### Fase 6 — Entrega / Produção
- [ ] Salvar pipeline treinada (`.pkl`)
- [ ] Criar função `preprocess_raw(df)` para dados novos brutos
- [ ] Estruturar para uso em API (ex: FastAPI)
