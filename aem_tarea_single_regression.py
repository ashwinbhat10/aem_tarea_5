# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 07:26:55 2021

@author: ASBHBHAT
"""


# Importar librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from scipy import stats
from statsmodels.compat import lzip
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

# %% Get the data

data = pd.read_csv(r'C:\Users\ASBHBHAT\Downloads\CarPrice_Assignment.csv')


# %% Variable Selection
# Independent Variable = carlength
X = np.array(data.carlength).reshape(-1, 1)

# Dependent Variable = Price
Y = np.array(data.price)

# %% Graficación
plt.figure()
plt.scatter(X, Y)
plt.title('Car Prices')
plt.xlabel('Length of car')
plt.ylabel('Price')
plt.show()

# %% Linear model
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
rls = linear_model.LinearRegression()
rls.fit(X_train, Y_train)
Y_pred = rls.predict(X_test)
Y_pred2 = rls.predict(X_train)

# %% Linear regression data
print("B_1:", rls.coef_)
print("B_0:",rls.intercept_)
print("rls score:",rls.score(X_train, Y_train))

# %% Parametric proof of hypothesis B1
error = Y_train - Y_pred2
ds_error = error.std()
ds_X = X_train.std()
error_st = ds_error/np.sqrt(X.size)
t1 = rls.coef_/(error_st/ds_X)
print("t1:",t1)

# %% Parametric proof of hypothesis B0
mean_X = X_train.mean()
mean_XC = mean_X**2
var_X = X_train.var()
to = rls.intercept_/(error_st*np.sqrt(1+(mean_XC/var_X)))
print("to:", to)

# %% Linear Model Graph
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred, color='r', linewidth=3)
plt.title(' Linear Regression ')
plt.xlabel('Length of the car')
plt.ylabel('Price of the car')
plt.show()

# %% Adjust the line of Regresion
plt.figure()
sns.regplot(Y_test, Y_pred, data=data, marker='+')
plt.xlabel('Actual Values')
plt.ylabel('Predicted  Values')
plt.title('Actual Values VS Predicted Value')
plt.show()

# %% Linear Regression Stats Model
# Omnibus y Jarque-Bera miden si los errores se distribuyen de manera normal.
# Durbin-Watson mide la autocorrelación de los errores.
X2 = sm.add_constant(X_train, prepend=True)
rls2 = sm.OLS(endog=Y_train, exog=X2)
rls2 = rls2.fit()
print(rls2.summary())
Y_pred3 = rls2.predict(X2)
error2 = Y_train - Y_pred3

# %% Visualizar Homocedasticidad - (La varianza de los errores es constante.)
plt.figure()
sns.regplot(Y_pred3, error2, data=data, marker='*')
plt.xlabel('Fitted Values', size=20)
plt.ylabel('Residuals', size=20)
plt.title('Fitted Values VS Residuals', size=20)
plt.show()

# %% Forma Estadística de Homocedasticidad: (Hay igualdad entre las varianzas.)
# Breusch-Pagan
# Ho: Homocedasticidad (p>0.05)
# H1: No hay homocedasticidad (p<0.05)
names = ['Lagrange multiplier statistic', 'p-value',
         'f-value', 'f p-value']
test = sms.het_breuschpagan(error2, X2)
print("Breush-pagan:",lzip(names, test))

# %% Forma gráfica de la  normalidad de los residuos
plt.figure()
plt.hist(rls2.resid_pearson)
plt.show()

# %% QQ plot
plt.figure()
ax = sm.qqplot(rls2.resid_pearson)
plt.show()


# %% Forma Estadística de la normalidad (Shaphiro-Wilk)
# Shapiro-Wilks también mide la normalidad de los errores.
names = ['Stastistic', 'p-value']
test = stats.shapiro(error2)
print("shapiro-wilks:", lzip(names, test))
