import pandas as pd
from numpy import int64

df = pd.read_csv(r'C:\CDS_PyCharm\do_zero_ao_ds\datasets\kc_house_data.csv')

#convertendo object para date:

#df['date'] = pd.to_datetime(df['date'])

#print(df.dtypes)

#===================================
#Como converter os tipos de variáveis
#===================================

# # int -> float
# df['bedrooms'] = df['bedrooms'].astype(float)
#
# # float -> int
# df['bedrooms'] = df['bedrooms'].astype(int64)
#
# # int -> str
# df['bedrooms'] = df['bedrooms'].astype(str)
#
# # Object -> date
# df['date'] = pd.to_datetime(df['date'])


#===================================
#Manipulando Variáveis
#===================================

#Criar
# df['nova_coluna'] = 'Renan'

#Deletar
# df = df.drop('nova_coluna', axis = 1)

#Selecionar

#1 - Direto pelo nome da coluna
#df['bedrooms', 'price', 'bathrooms']

#2 - Pelo índice das linhas e colunas - iloc
#Seleciona por indíces das linhas e colunas
#df.iloc[0:5,0:3]

#3 - Pelo índice das linhas e nomes das colunas - loc
#df.loc[0:5, ['price', 'bathroom']

#4 - Índice Booleanos
#df.loc[0:5, [True,False,...,False]]

