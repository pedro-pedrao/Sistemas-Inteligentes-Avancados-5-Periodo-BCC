"""
Este exemplo demonstra o procedimento básico para normalizar 
um novo valor, obedecendo ao modelo normalizador 
dos dados originais
"""
import pickle
import numpy as np

novo_dado = [[2200]]

#Abrir o modelo normalizador
scaler_model = pickle.load(open('scaler_model.pkl', 'rb'))

#Normalizar o novo dados obedecendo ao modelo previo
novo_dado_norm = scaler_model.transform(novo_dado)
print(novo_dado_norm)

#Atividade
#1. Receber um valor normalizado

# Reverter de Normalizado -> Natural usando o modelo scaler_model
dado_nat = scaler_model.inverse_transform(novo_dado_norm)
print(dado_nat)