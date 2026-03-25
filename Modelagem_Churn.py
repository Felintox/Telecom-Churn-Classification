#%%
### 1.0 Importação das Bibliotecas
import pandas as pd
import numpy as np
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
# %%
data.info()
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
X_train[var_cat].hist()

# %%
