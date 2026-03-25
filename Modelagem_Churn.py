#%%
### 1.0 Importação das Bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


#%%
### 2.0 Contexto do Projeto
### Objetivo: Construir um modelo de classificação para prever churn de clientes de uma empresa de telecomunicações, permitindo que a empresa tome ações proativas de retenção antes que o cliente cancele.
### 
### Problema de negócio: Reter clientes é mais barato do que adquirir novos. Identificar clientes com alta probabilidade de churn permite campanhas de retenção direcionadas e eficientes.

#%%
###  3.0 Entendimento dos Dados
data=pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
                 )
print(f'Tamanho do Dataset: {data.shape[0]} linhas e {data.shape[1]} colunas')
data.head()
#%%
### A coluna 'customerID' é um identificador único para cada cliente, e não tem valor preditivo para o modelo. Portanto, podemos removê-la do conjunto de dados.
### Mas vale a pena verificar se existem valores duplicados na coluna 'customerID' antes de removê-la, para garantir que cada cliente seja representado apenas uma vez no dataset.
data['customerID'].nunique()/data.shape[0]
### Como não temos duplicados, podemos remover a coluna 'customerID' sem perder informações importantes para o modelo.
data.drop('customerID', axis=1, inplace=True)
# %%
data.info()
#%%
### A coluna 'SeniorCitizen' é do tipo numérico, mas na verdade representa uma variável categórica (0 para não idoso e 1 para idoso). Vamos converter essa coluna para o tipo categórico para facilitar a análise e o treinamento do modelo.
data['SeniorCitizen'].hist()
data['SeniorCitizen']=data['SeniorCitizen'].astype('object')
#%%
### Não temos valores nulos no dataset
### A Coluna TotalCharges tem o tipo de dado errado, vamos corrigir isso
### Percebe-se que existe valores na coluna TotalCharges (11 valores) que estão espaços vazios ', o que impossibilita a conversão direta para float. Precisamos tratar esses valores antes de converter a coluna para o tipo numérico.
#%%
masc1=data[data['TotalCharges'] == ' '].index
data.drop(masc1, inplace=True)
data['TotalCharges'] = data['TotalCharges'].astype('float64')
#%%
data.info()
#%%
### Pronto, agora o TotalCharges é um valor do tipo float64.
#%%
### Antes de partir para uma analise explotaria vou fazer entre dados de treino e dados de teste, para evitar o vazamento de dados.
### E se queremos realmente prever o churn, precisamos garantir que nosso modelo seja treinado apenas com dados anteriores ao momento da previsão. Se usarmos dados futuros (dados de teste) durante a fase de análise exploratória, corremos o risco de introduzir informações que não estariam disponíveis no momento da previsão, o que pode levar a um modelo superestimado e com desempenho irrealista.
### Portanto, é fundamental separar os dados de treino e teste antes de realizar qualquer análise exploratória para garantir que o modelo seja treinado e avaliado de maneira justa e realista.
#%%
### Antes de separar os dados, vamos verificar a distribuição da variável alvo 'Churn' para entender o equilíbrio das classes. Isso é importante para garantir que nosso modelo seja treinado de maneira adequada e para escolher as métricas de avaliação apropriadas.
### Se necessario podemos adicionar uma estratificação na hora de separar os dados para garantir que a proporção de classes seja mantida tanto no conjunto de treino quanto no conjunto de teste.
data['Churn'].value_counts()
#%%
### Como temos uma distribuição de classes relativamente pequena para os clientes que cancelaram (churn), é importante garantir que essa proporção seja mantida tanto no conjunto de treino quanto no conjunto de teste. Para isso, podemos usar a estratificação na função `train_test_split` do scikit-learn, garantindo que a proporção de classes seja preservada em ambos os conjuntos.
#%%
### Eu não vou separar um conjunto de validação, pois vou usar a validação cruzada para avaliar o desempenho do modelo durante o processo de treinamento. 
### A validação cruzada é uma técnica que divide os dados de treino em várias partes (folds) e treina o modelo em diferentes combinações desses folds, permitindo uma avaliação mais robusta do desempenho do modelo sem a necessidade de um conjunto de validação separado.
X = data.drop('Churn', axis=1)
y = data['Churn'].map({'Yes': 1, 'No': 0})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,
                                                    stratify=y)
# %%
print(f'Tamanho do Conjunto de Treino: {X_train.shape[0]} linhas')
print(f'Tamanho do Conjunto de Teste: {X_test.shape[0]} linhas')
print(f'Tamanho do Conjunto de Treino: {y_train.shape[0]} linhas')
print(f'Tamanho do Conjunto de Teste: {y_test.shape[0]} linhas')
# %%
### 4.0 Análise Exploratória dos Dados (EDA)
### 4.1 Análise Univariada
var_cat=X_train.select_dtypes(include='object').columns
var_num=X_train.select_dtypes(exclude='object').columns
print(f'Variáveis Categóricas: {var_cat}\n')
print(f'Variáveis Numéricas: {var_num}')
#%%
### Plot das variáveis categóricas
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 15))
axes = axes.flatten()  # simplifica a indexação
for i, var in enumerate(var_cat):
    X_train[var].value_counts().plot(kind='bar', ax=axes[i])
    axes[i].set_title(var)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Contagem')

### As variaveis Categóricas tem uma distribuição relativamente equilibrada, o que é bom para o treinamento do modelo. No entanto, é importante observar que algumas categorias podem ter uma contagem significativamente menor do que outras, o que pode afetar a capacidade do modelo de aprender padrões relevantes para essas categorias minoritárias.
### Como por exemplo a 'PhoneService' tem uma categoria 'No' com uma contagem significativamente menor do que a categoria 'Yes', o que pode dificultar a capacidade do modelo de aprender padrões relevantes para os clientes que não possuem serviço de telefone.
#%%
### Plot das variáveis numéricas
fig,axes=plt.subplots(nrows=1,ncols=3,figsize=(16,10))
axes=axes.flatten()
for i, var in enumerate(var_num):
    X_train[var].hist(ax=axes[i])
    axes[i].set_title(var)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Distribuição')
plt.tight_layout()
### Nessa plotagem ficou evidente que o 'SeniorCitizen' é na verdade uma variavel
#%%
### 4.2 Análise Bivariada
### Correlação Linear entre as variáveis numéricas
corr_matrix = X_train[var_num].corr()
corr_matrix
### A variavel TotalCharges tem um correlação forte em relação as outras variaveis numericas.
#%%
### 4.2.1 Análise bivariada — relação das features com Churn
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 10))
axes = axes.flatten()

for i, var in enumerate(var_num):
    if var == 'TotalCharges':
        sns.violinplot(x=y_train, y=np.log1p(X_train[var]), ax=axes[i])
        axes[i].set_ylabel('log(TotalCharges)')
    else:
        sns.violinplot(x=y_train, y=X_train[var], ax=axes[i])
    axes[i].set_title(f'{var} vs Churn')

# %%
### Tenure: Clientes com churn concentram-se em valores baixos de tenure — clientes recentes cancelam mais. Clientes sem churn têm distribuição mais uniforme ao longo do tempo.
### 
### MonthlyCharges: Churn associado a cobranças mensais mais altas. Clientes sem churn concentram-se em valores mais baixos.
### 
### TotalCharges (log): Clientes sem churn apresentam densidade em valores mais altos, reflexo direto do maior tempo de permanência (correlação com tenure). A transformação logarítmica foi necessária para melhorar a visualização da distribuição.

#%%
### 4.2.2 Análise bivariada — relação das features categóricas com Churn
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 15))
axes = axes.flatten()

for i, var in enumerate(var_cat):
    pd.crosstab(X_train[var], y_train, normalize='index').plot(kind='bar', stacked=True, ax=axes[i])
    axes[i].set_title(var)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Proporção')
    axes[i].legend(['No Churn', 'Churn'], fontsize=8)
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
# %%
###Análise Bivariada — Variáveis Categóricas

## Sinais fortes de churn:
## 
## Contract: clientes com contrato Month-to-month apresentam taxa de churn significativamente maior. Contratos mais longos funcionam como fator de retenção.
## InternetService: Fiber optic concentra maior proporção de churn, sugerindo insatisfação com qualidade ou custo do serviço.
## SeniorCitizen: idosos apresentam maior propensão ao cancelamento.
## OnlineSecurity / TechSupport / OnlineBackup / DeviceProtection: clientes sem esses serviços adicionais churnam mais, indicando que o engajamento com o produto está associado à retenção.
## PaymentMethod: Electronic check destoa dos demais métodos com taxa de churn consideravelmente maior.
## Pouco ou nenhum sinal preditivo:
## 
## Gender: distribuição praticamente idêntica entre os gêneros — baixo poder discriminativo esperado.
## PhoneService / MultipleLines / StreamingTV / StreamingMovies: diferenças pequenas entre categorias, provavelmente com baixa importância no modelo.
#%%
### 5.0 processamento de dados via pipeline

