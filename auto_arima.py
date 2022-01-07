


import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from statsmodels.tsa.arima_model import ARIMA
from arch import arch_model
import yfinance
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set()




#%%

# Descarga de indices bursatiñes

raw_data = yfinance.download (tickers = "^GSPC ^FTSE ^N225 ^GDAXI", start = "1994-01-07", end = "2022-01-01", 
                              interval = "1d", group_by = 'ticker', auto_adjust = True, treads = True)


df_comp = raw_data.copy()


df_comp.head()




#%%


# Editamos los nombres y seleccionamos la variable Close

df_comp['spx'] = df_comp['^GSPC'].Close[:]
df_comp['dax'] = df_comp['^GDAXI'].Close[:]
df_comp['ftse'] = df_comp['^FTSE'].Close[:]
df_comp['nikkei'] = df_comp['^N225'].Close[:]





#%%


df_comp.head()



df_comp.tail()



#%%

# Frecuencia dia de negocio


del df_comp['^N225']
del df_comp['^GSPC']
del df_comp['^GDAXI']
del df_comp['^FTSE']
df_comp=df_comp.asfreq('b')
df_comp=df_comp.fillna(method='ffill')



df_comp.head()


#%%


# Calculo de Retornos


df_comp['ret_spx'] = df_comp.spx.pct_change(1)*100
df_comp['ret_ftse'] = df_comp.ftse.pct_change(1)*100
df_comp['ret_dax'] = df_comp.dax.pct_change(1)*100
df_comp['ret_nikkei'] = df_comp.nikkei.pct_change(1)*100





#%%

# Entrenamiento y prueba


size = int(len(df_comp)*0.8)
df, df_test = df_comp.iloc[:size], df_comp.iloc[size:]




#%%

# Importamos la funcion auto arima

from pmdarima.arima import auto_arima


model_auto = auto_arima(df.ret_ftse[1:])


print('El orden del modelo es: ',model_auto.order)


# Nos da un arma (4,5)
# El metodo mira los estimadores aik y bic

model_auto.summary()

#%%

# AUTO ARIMA EDITADO::::
    
    
    

# Con este comando vamos a introducir:


# Variables exogenas: exogenous = df_comp[['ret_spx', 'ret_dax', 'ret_nikkei']][1:]

#  Con m = 5 vamos a poner una longitud del ciclo estacional, es decir, es el margen de estacionalidad que se va a tener en cuenta, al analizar dias habiles entonces ponemos 5

# Con max_order = None, determinamos el numero total de coeficientes no estacionales que va a tener el modelo ( parte AR y MA ), le ponemos none.


# Con max_p = 7 , max_d = 2 y max_q = 7 ponemos ordenes maximos


# Con max_P = 4, max_Q = 4, max_D = 2 trabajamos la S, es decir, el componente estacional SARIMA.

# Con  maxiter = 50 le ponemos un limite para que encuentre un modelo

# Con alpha = 0.05 tenemos el nivel de siignificacion

# Con n_jobs = -1 medimos cuantos modelos estamos dejando que autoarima estime al mismo tiempo.

# Con trend = 'ct' estamos considerando una tendencia constante. Si vemos una relacion cuadratica se debera poner ctt, si la tendencia es mas compleja se puede usar una matriz de valores booleanos

# Con information_criterion = 'oob' seleccionamos el criterio oob (out of back) en vez de usar el aic o el bic. Este criterio divide el conjunto de datos en test y prueba, segun los resultados escogera el mejor modelo, para este comando tenemos que usar:  out_of_sample_size = int(len(df_comp)*0.2)) que es la cantidad de datos que dejaremos fuera para validar el modelo bajo el criterio seleccionado






model_auto2 = auto_arima(df_comp.ret_ftse[1:], exogenous = df_comp[['ret_spx', 'ret_dax', 'ret_nikkei']][1:], m = 5,
                       max_order = None, max_p = 7, max_q = 7, max_d = 2, max_P = 4, max_Q = 4, max_D = 2,
                       maxiter = 50, alpha = 0.05, n_jobs = -1, trend = 'ct', information_criterion = 'oob',
                       out_of_sample_size = int(len(df_comp)*0.2))





model_auto2.summary()








#%%