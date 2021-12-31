
# Exploramos un nuevo data set de poblacion en EEUU

import pandas as pd
import numpy as no
%matplotlib inline


from statsmodels.tsa.ar_model import AR, ARResults

df = pd.read_csv('poblacioneeuu.csv', index_col = 'DATE', parse_dates = True)


# Ponemos la frecuencia en meses

df.index.freq = 'MS'


df.head()



#%%

# graficamos:

ax = df['PopEst'].plot(figsize=(12,5), title = 'US Poblacion Mensual')
ax.autoscale(axis = 'x', tight = True)
ax.set(xlabel = 'año', ylabel = 'poblacion')





#%%

# probamos los modelos

train = df.iloc[:84]

test = df.iloc[84:]


# Modelo AR(1)

import warnings
warnings.filterwarnings('ignore')


model = AR(train['PopEst'])
AR1fit = model.fit(maxlag = 1, method = 'mle')


print('lag: ',AR1fit.k_ar, '\n')
print('Coeficientes: \n',AR1fit.params)





#%%

# Prediccion:
    
# Comenzamos donde termina el conjunto de entrenamiento
    
start = len(train)
end = len(train) + len(test) -1

predictions1 = AR1fit.predict(start = start,
                              end = end,
                              dynamic = False).rename('ar1 predictions')

predictions1





#%%

test['PopEst'].plot(legend = True)
predictions1.plot(legend = True, figsize = (12,6))

# El retraso de un mes anterior no nos sirve para estimar el aumento en la poblacion




#%%


# Agregaremos complejidad:

model = AR(train['PopEst'])
AR2fit = model.fit(maxlag = 2, method = 'mle')


print('lag: ',AR2fit.k_ar, '\n')
print('Coeficientes: \n',AR2fit.params)



start = len(train)
end = len(train) + len(test) -1

predictions2 = AR2fit.predict(start = start,
                              end = end,
                              dynamic = False).rename('ar2 predictions')

predictions2



test['PopEst'].plot(legend = True)
predictions1.plot(legend = True)
predictions2.plot(legend = True, figsize = (12,6))




# con dos retrasos ha mejorado bastante


#%%

# retornos optimos


# si ponemos ic = 'bic' nos busca el AR optimo. podemos poner aci o bic como criterios para ir probando en vez de bic

model = AR(train['PopEst'])
ARfit = model.fit(ic = 'bic')

print('lag: ',ARfit.k_ar, '\n')
print('Coeficientes: \n',ARfit.params)

# nos considera hasta 8 retornos



start = len(train)
end = len(train) + len(test) -1

predictions8 = ARfit.predict(start = start,
                              end = end,
                              dynamic = False).rename('ar8 predictions')

predictions8



test['PopEst'].plot(legend = True)
predictions1.plot(legend = True)
predictions2.plot(legend = True)
predictions8.plot(legend = True, figsize = (12,6))


# ajusta casi perfecto

#%%

# Evaluamos el modelo usando el MSE error cuadratico medio

from sklearn.metrics import mean_squared_error


labels = ['ar1', 'ar2', 'ar8']

preds = [predictions1, predictions2, predictions8]


for i in range(len(labels)):
    error = mean_squared_error(test['PopEst'], preds[i])
    print(f'{labels[i]} Error: {error.round(4)}')

    
# Evaluamos el modelo usando el AIC

models = [AR1fit, AR2fit, ARfit]

for i in range(len(models)):
    print(f'{labels[i]} AIC: {models[i].aic.round(4)}')



#%%


# Ahora que tenemos el modelo podemos hacer una prediccion:
    
    
model = AR(df['PopEst'])

ARfit = model.fit(maxlag=8,method='mle')

#predictions

# Que comience desde el ultimo valor del df y termine doce meses posteriores

forecast = ARfit.predict(start=len(df), end=len(df)+12, dynamic=False).rename('Forecast')

# Plot the results
df['PopEst'].plot(legend=True)
forecast.plot(legend=True,figsize=(12,6))

#%%
