import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle #para salvar e recuperar os modelos treinados e gravados

#Gerar dados para teste
dados = np.array([
    [1500],[3000],[5500],[10000]
])

#Instanciar o MinMax Scaler
scaler = MinMaxScaler()

#Criar o modelo normalizador
scaler_model = scaler.fit(dados)

#Salvar o modelo normalizador em disco
pickle.dump(scaler_model, open('scaler_model.pkl', 'wb'))

#Normalizar os dados utilizando o modelo normalizador
dados_norm = scaler_model.fit_transform(dados)

print(dados_norm)
