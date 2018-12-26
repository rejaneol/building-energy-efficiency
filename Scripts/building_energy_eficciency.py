
# coding: utf-8

# # Eficiencia Energetica em Edificacoes 
# ## Uma aplicacao de Machine Learning



"""
energy.py: Eficiencia energetica: Previsão de demanda de energia para aquecimento e resfriamento em edificacoes 

Avaliar a performance de métodos de previsão por Regressão Linear (incluindo tecnicas de regularizacao Lassoe Ridge),
K-Nearest Neighbor e o metodo ensemble Gradient Boosting.

@author: Rejane
@contact: re.ol.2015@gmail.com

@date	  Setembro 2018
@version 1.0
"""


# # Importing Libraries

# Importando pacotes
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# Importando pacotes
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn import ensemble
from sklearn.metrics import mean_squared_error, r2_score

# Versões dos pacotes instalados na plataforma
print("numpy version: %s" % np.__version__)
print("pandas version: %s" % pd.__version__)
#print("scikit-learn version: %s" % sklearn.__version__)

# # Loading dataset


# Carrega a base de dados (Referencia https://archive.ics.uci.edu/ml/datasets/Energy+efficiency)
print("Carrega a base de dados: Eficiencia energetica de edificacoes")
data = pd.read_excel('../Dados/ENB2012_data.xlsx')
data.info()

data.head()

data.tail()


# ### Descrição do conjunto de dados
# 
# O conjunto de dados contem oito atributos (ou 'features', indicados por X1...X8) e duas respostas (ou saídas, indicadas por y1 e y2). 
# 
# O objetivo é usar os oito atributos para prever cada uma das duas respostas.
# 
# Especificamente: 
# 
# - X1	Relative Compactness – Compactação relativa 
# - X2	Surface Area – Área de superfície 
# - X3	Wall Area –Área de parede 
# - X4	Roof Area – Área de Telhado 
# - X5	Overall Height – Altura total 
# - X6	Orientation – Orientaçao (espacial) 
# - X7	Glazing Area – Área envidraçada 
# - X8	Glazing Area Distribution – Distribuição da área envidraçada 
# - y1	Heating Load – Demanda de energia para aquecimento 
# - y2	Cooling Load – Demanda de energia para resfriamento 

# ## Data Exploration

#Estatistica descritiva
data.describe(include='all')

# Visualizacao variaveis target
plt.subplot(121)
data['Y1'].plot.hist(figsize=(12,5), alpha = 0.6, title = 'Heating Load – Demanda de energia para aquecimento')
plt.subplot(122)
data['Y2'].plot.hist(figsize=(12,5), alpha = 0.6, title = 'Cooling Load – Demanda de energia para resfriamento')
plt.show()

# Visualizacao features e target Y1
sns.pairplot(data, x_vars=['X1', 'X2', 'X3', 'X4','X5', 'X6', 'X7', 'X8'], y_vars=['Y1'], kind="reg")
plt.show()

# Visualizacao features e target Y2
sns.pairplot(data, x_vars=['X1', 'X2', 'X3', 'X4','X5', 'X6', 'X7', 'X8'], y_vars=['Y2'], kind="reg")
plt.show()

# Correlacao entre as variaveis
round(data.corr(),3)

# # Data preparation

# Separando as series para analise -  features(X) e target(y)

X = data.iloc[:,0:(data.shape[1] - 2)]
y1 = data.take([8], axis=1) 
y2 = data.take([9], axis=1) 

print(X.info())
X.head()

print(y1.info())
y1.head()

print(y2.info())
y2.head()


# Inicialmente definindo 'target' como 'Heating Load'- y como y1
# Separando o conjunto de dados de treino e de teste

X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.1, random_state=0)


# ## Modeling

# ### Previsao de Heating Load  (target Y1)

# Cria um modelo de regressão com 3 vizinhos. Os valores default são:
# KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
# metric_params=None, n_jobs=1, n_neighbors=3, p=2, weights='uniform')

knreg = KNeighborsRegressor(n_neighbors=3)
knreg.fit(X_train, y_train)

y_pred = knreg.predict(X_test)

print('Modelo KNN:')
print("Acurácia da base de treinamento: {:.2f}".format(knreg.score(X_train, y_train)))
print("Acurácia da base de testes: {:.2f}".format(knreg.score(X_test, y_test)))
print('r2: ', round(r2_score(y_test, y_pred),4))
print("MSE: ", round(mean_squared_error(y_test, y_pred),4))

# Cria um modelo de regressão linear
# Os valores default são: 
# LinearRegression(fit_intercept=True, normalize=False)

lnr = LinearRegression().fit(X_train, y_train)
y_pred = lnr.predict(X_test)

print('Modelo Regressao Linear (dados nao normalizados):')
print("Acurácia da base de treinamento: {:.2f}".format(lnr.score(X_train, y_train)))
print("Acurácia da base de testes: {:.2f}".format(lnr.score(X_test, y_test)))
print("w[0]: ",  lnr.coef_[0])
print("b: " , lnr.intercept_)
print('r2: ', round(r2_score(y_test, y_pred),4))
print("MSE: ", round(mean_squared_error(y_test, y_pred),4))


## visualizando coeficientes do modelo
coef = pd.DataFrame(X_train.columns)
coef['Coeficiente Estimado'] = pd.Series(lnr.coef_[0])
print(coef)
coef.plot(kind='bar', title='Coeficientes Regressao Linear s/norm.', sort_columns = False)
plt.show()


# Cria um modelo de regressão linear com dados normalizados

lnr = LinearRegression(normalize=True).fit(X_train, y_train)
y_pred = lnr.predict(X_test)

print('Modelo Regressao Linear (dados normalizados):')
print("Acurácia da base de treinamento: {:.2f}".format(lnr.score(X_train, y_train)))
print("Acurácia da base de testes: {:.2f}".format(lnr.score(X_test, y_test)))
print("w[0]: ",  lnr.coef_[0])
print("b: " , lnr.intercept_)
print('r2: ', round(r2_score(y_test, y_pred),4))
print("MSE: ", round(mean_squared_error(y_test, y_pred),4))


# Regularização e feature selection - Ridge

ridge = Ridge(alpha=1).fit(X_train, y_train)
y_pred = ridge.predict(X_test)

print('Modelo Regressao Linear Ridge (dados nao normalizados):')
print("Acurácia da base de treinamento: {:.2f}".format(ridge.score(X_train, y_train)))
print("Acurácia da base de testes: {:.2f}".format(ridge.score(X_test, y_test)))
print()
print("Ridge: w: {}  b: {}".format(ridge.coef_, ridge.intercept_))
print("Número de atributos usados: {}".format(np.sum(ridge.coef_ != 0)))
print('r2: ', round(r2_score(y_test, y_pred),4))
print("MSE: ", round(mean_squared_error(y_test, y_pred),4))


# Regularização e feature selection - Lasso (Least Absolute Shrinkage Selector Operator)

lasso = Lasso(alpha=1).fit(X_train, y_train)
y_pred = lasso.predict(X_test)

print('Modelo Regressao Linear Lasso (dados nao normalizados):')
print("Acurácia da base de treinamento: {:.2f}".format(lasso.score(X_train, y_train)))
print("Acurácia da base de testes: {:.2f}".format(lasso.score(X_test, y_test)))
print()
print("Lasso: w: {}  b: {}".format(lasso.coef_, lasso.intercept_))
print("Número de atributos usados: {}".format(np.sum(lasso.coef_ != 0)))
print('r2: ', round(r2_score(y_test, y_pred),4))
print("MSE: ", round(mean_squared_error(y_test, y_pred),4))


# Regularização e feature selection - Ridge - Ajuste de hiperparametros - sem normalizacao

ridge = Ridge().fit(X_train, y_train) #default alpha = 1
y_pred = ridge.predict(X_test)
print("Ridge 1")
print("Acurácia na base de treinamento: {:.2f}".format(ridge.score(X_train, y_train)))
print("Acurácia na base de teste: {:.2f}".format(ridge.score(X_test, y_test)))
print('r2: ', round(r2_score(y_test, y_pred),4))
print("MSE: ", round(mean_squared_error(y_test, y_pred),4))
print("Número de atributos usados: {}".format(np.sum(ridge.coef_ != 0)))
print()


ridge10 = Ridge(alpha=10).fit(X_train, y_train)
y_pred = ridge10.predict(X_test)
print("Ridge 10")
print("Acurácia na base de treinamento: {:.2f}".format(ridge10.score(X_train, y_train)))
print("Acurácia na base de teste: {:.2f}".format(ridge10.score(X_test, y_test)))
print('r2: ', round(r2_score(y_test, y_pred),4))
print("MSE: ", round(mean_squared_error(y_test, y_pred),4))
print("Número de atributos usados: {}".format(np.sum(ridge.coef_ != 0)))
print()

ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
y_pred = ridge01.predict(X_test)
print("Ridge .1")
print("Acurácia na base de treinamento: {:.2f}".format(ridge01.score(X_train, y_train)))
print("Acurácia na base de teste: {:.2f}".format(ridge01.score(X_test, y_test)))
print('r2: ', round(r2_score(y_test, y_pred),4))
print("MSE: ", round(mean_squared_error(y_test, y_pred),4))
print("Número de atributos usados: {}".format(np.sum(ridge.coef_ != 0)))



# Regularização e feature selection - Ridge - Ajuste de hiperparametros - com normalizacao

ridge = Ridge(normalize=True).fit(X_train, y_train) #default alpha = 1
y_pred = ridge.predict(X_test)
print("Ridge 1")
print("Acurácia na base de treinamento: {:.2f}".format(ridge.score(X_train, y_train)))
print("Acurácia na base de teste: {:.2f}".format(ridge.score(X_test, y_test)))
print('r2: ', round(r2_score(y_test, y_pred),4))
print("MSE: ", round(mean_squared_error(y_test, y_pred),4))
print("Número de atributos usados: {}".format(np.sum(ridge.coef_ != 0)))
print()


ridge10 = Ridge(alpha=10, normalize=True).fit(X_train, y_train)
y_pred = ridge10.predict(X_test)
print("Ridge 10")
print("Acurácia na base de treinamento: {:.2f}".format(ridge10.score(X_train, y_train)))
print("Acurácia na base de teste: {:.2f}".format(ridge10.score(X_test, y_test)))
print('r2: ', round(r2_score(y_test, y_pred),4))
print("MSE: ", round(mean_squared_error(y_test, y_pred),4))
print("Número de atributos usados: {}".format(np.sum(ridge.coef_ != 0)))
print()

ridge01 = Ridge(alpha=0.1, normalize=True).fit(X_train, y_train)
y_pred = ridge01.predict(X_test)
print("Ridge .1")
print("Acurácia na base de treinamento: {:.2f}".format(ridge01.score(X_train, y_train)))
print("Acurácia na base de teste: {:.2f}".format(ridge01.score(X_test, y_test)))
print('r2: ', round(r2_score(y_test, y_pred),4))
print("MSE: ", round(mean_squared_error(y_test, y_pred),4))
print("Número de atributos usados: {}".format(np.sum(ridge.coef_ != 0)))


# Regularização e feature selection - Lasso - Ajuste de hiperparametros - sem normalizacao

lasso = Lasso().fit(X_train, y_train) #default alpha = 1
y_pred = lasso.predict(X_test)
print("Lasso 1")
print("Acurácia na base de treinamento: {:.2f}".format(lasso.score(X_train, y_train)))
print("Acurácia na base de teste: {:.2f}".format(lasso.score(X_test, y_test)))
print('r2: ', round(r2_score(y_test, y_pred),4))
print("MSE: ", round(mean_squared_error(y_test, y_pred),4))
print("Número de atributos usados: {}".format(np.sum(lasso.coef_ != 0)))
print()

lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
y_pred = lasso001.predict(X_test)
print("Lasso .01")
print("Acurácia na base de treinamento: {:.2f}".format(lasso001.score(X_train, y_train)))
print("Acurácia na base de teste: {:.2f}".format(lasso001.score(X_test, y_test)))
print('r2: ', round(r2_score(y_test, y_pred),4))
print("MSE: ", round(mean_squared_error(y_test, y_pred),4))
print("Número de atributos usados: {}".format(np.sum(lasso001.coef_ != 0)))
print()

lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
y_pred = lasso00001
print("Lasso .0001")
print("Acurácia na base de treinamento: {:.2f}".format(lasso00001.score(X_train, y_train)))
print("Acurácia na base de teste: {:.2f}".format(lasso00001.score(X_test, y_test)))
#print('r2: ', round(r2_score(y_test, y_pred),4))
#print("MSE: ", round(mean_squared_error(y_test, y_pred),4))
print("Número de atributos usados: {}".format(np.sum(lasso00001.coef_ != 0)))


# Regularização e feature selection - Lasso - Ajuste de hiperparametros - com normalizacao

lasso = Lasso(normalize=True).fit(X_train, y_train) #default alpha = 1
y_pred = lasso.predict(X_test)
print("Lasso 1")
print("Acurácia na base de treinamento: {:.2f}".format(lasso.score(X_train, y_train)))
print("Acurácia na base de teste: {:.2f}".format(lasso.score(X_test, y_test)))
print('r2: ', round(r2_score(y_test, y_pred),4))
print("MSE: ", round(mean_squared_error(y_test, y_pred),4))
print("Número de atributos usados: {}".format(np.sum(lasso.coef_ != 0)))
print()

lasso001 = Lasso(alpha=0.01, max_iter=100000, normalize=True).fit(X_train, y_train)
y_pred = lasso001.predict(X_test)
print("Lasso .01")
print("Acurácia na base de treinamento: {:.2f}".format(lasso001.score(X_train, y_train)))
print("Acurácia na base de teste: {:.2f}".format(lasso001.score(X_test, y_test)))
print('r2: ', round(r2_score(y_test, y_pred),4))
print("MSE: ", round(mean_squared_error(y_test, y_pred),4))
print("Número de atributos usados: {}".format(np.sum(lasso001.coef_ != 0)))
print()

lasso00001 = Lasso(alpha=0.0001, max_iter=100000, normalize=True).fit(X_train, y_train)
y_pred = lasso00001
print("Lasso .0001")
print("Acurácia na base de treinamento: {:.2f}".format(lasso00001.score(X_train, y_train)))
print("Acurácia na base de teste: {:.2f}".format(lasso00001.score(X_test, y_test)))
#print('r2: ', round(r2_score(y_test, y_pred),4))
#print("MSE: ", round(mean_squared_error(y_test, y_pred),4))
print("Número de atributos usados: {}".format(np.sum(lasso00001.coef_ != 0)))


# Ajuste de hiperparametros Ridge sem normalizacao dos dados - busca do melhor alpha

ridge_cv = RidgeCV(alphas=(0.1, 1.0, 10.0), normalize=False, store_cv_values=True).fit(X_train, y_train)
y_pred = ridge_cv.predict(X_test)

print("Acurácia na base de treinamento: {:.2f}".format(ridge_cv.score(X_train, y_train)))
print("Acurácia na base de teste: {:.2f}".format(ridge_cv.score(X_test, y_test)))
print()
print("Os melhores parâmetros encontrados:")
print(ridge_cv.cv_values_,'\n')
print(ridge_cv.alpha_,'\n')
print("Número de atributos usados: {}".format(np.sum(ridge_cv.coef_ != 0)))
print('r2: ', round(r2_score(y_test, y_pred),4))
print("MSE: ", round(mean_squared_error(y_test, y_pred),4))


# visualizando coeficientes do modelo
coef = pd.DataFrame(X_train.columns)
coef['Coeficiente Estimado'] = pd.Series(ridge_cv.coef_[0])
print(coef)
coef.plot(kind='bar', title='Coeficientes Ridge s/norm.', sort_columns = False)
plt.show()


# Ajuste de hiperparametros Ridge com normalizacao dos dados - busca do melhor alpha

ridge_cv = RidgeCV(alphas=(0.1, 1.0, 10.0), normalize=True, store_cv_values=True).fit(X_train, y_train)
y_pred = ridge_cv.predict(X_test)

print("Acurácia na base de treinamento: {:.2f}".format(ridge_cv.score(X_train, y_train)))
print("Acurácia na base de teste: {:.2f}".format(ridge_cv.score(X_test, y_test)))
print()
print("Os melhores parâmetros encontrados:")
print(ridge_cv.cv_values_,'\n')
print(ridge_cv.alpha_,'\n')
print("Número de atributos usados: {}".format(np.sum(ridge_cv.coef_ != 0)))
print('r2: ', round(r2_score(y_test, y_pred),4))
print("MSE: ", round(mean_squared_error(y_test, y_pred),4))


# Ajuste de hiperparametros Lasso sem normalizacao dos dados - busca do melhor alpha

lasso_cv = LassoCV(normalize=False, cv=5, random_state=0).fit(X_train, y_train)
y_pred = lasso_cv.predict(X_test)

print("Acurácia na base de treinamento: {:.2f}".format(lasso_cv.score(X_train, y_train)))
print("Acurácia na base de teste: {:.2f}".format(lasso_cv.score(X_test, y_test)))
print()
print("Os melhores parâmetros encontrados:")
print(round(lasso_cv.alpha_,4),'\n')
print(lasso_cv.alphas_,'\n')
print("Número de atributos usados: {}".format(np.sum(lasso_cv.coef_ != 0)))
print('r2: ', round(r2_score(y_test, y_pred),4))
print("MSE: ", round(mean_squared_error(y_test, y_pred),4))


# visualizando coeficientes do modelo
coef = pd.DataFrame(X_train.columns)
coef['Coeficiente Estimado'] = pd.Series(lasso_cv.coef_[0])
print(coef)
coef.plot(kind='bar', title='Coeficientes Lasso s/norm.', sort_columns = False)
plt.show()


# Ajuste de hiperparametros Lasso com normalizacao dos dados - busca do melhor alpha

lasso_cv = LassoCV(normalize=True, cv=5, random_state=0).fit(X_train, y_train)
y_pred = lasso_cv.predict(X_test)

print("Acurácia na base de treinamento: {:.2f}".format(lasso_cv.score(X_train, y_train)))
print("Acurácia na base de teste: {:.2f}".format(lasso_cv.score(X_test, y_test)))
print()
print("Os melhores parâmetros encontrados:")
print(round(lasso_cv.alpha_,4),'\n')
print(lasso_cv.alphas_,'\n')
print("Número de atributos usados: {}".format(np.sum(lasso_cv.coef_ != 0)))
print('r2: ', round(r2_score(y_test, y_pred),4))
print("MSE: ", round(mean_squared_error(y_test, y_pred),4))


# visualizando coeficientes do modelo
coef = pd.DataFrame(X_train.columns)
coef['Coeficiente Estimado'] = pd.Series(lasso_cv.coef_[0])
print(coef)
coef.plot(kind='bar', title='Coeficientes Lasso c/norm.', sort_columns = False)
plt.show()


# Um modelo ensemble

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,'learning_rate': 0.01, 'loss': 'ls'}
reg = ensemble.GradientBoostingRegressor(**params).fit(X_train, y_train)
y_pred = reg.predict(X_test)

print("Acurácia na base de treinamento: {:.2f}".format(reg.score(X_train, y_train)))
print("Acurácia na base de teste: {:.2f}".format(reg.score(X_test, y_test)))
print('r2: ', round(r2_score(y_test, y_pred),4))
print("MSE: ", round(mean_squared_error(y_test, y_pred),4))
print("Número de atributos usados: {}".format(np.sum(reg.feature_importances_ != 0)))

# visualizando coeficientes do modelo
coef = pd.DataFrame(X_train.columns)
coef['Coeficiente Estimado'] = pd.Series(reg.feature_importances_)
print(coef)
coef.plot(kind='bar', title='Coeficientes Gradient Boosting', sort_columns = False)
plt.show()

