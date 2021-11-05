# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 08:49:05 2021

@author: ASBHBHAT
"""

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
#%% Get the data

data = pd.read_csv(r'C:\Users\ASBHBHAT\Downloads\CarPrice_Assignment.csv')

#%% Multiple Linear Regression
#Independent variable
X_multiple=np.array(data.iloc[:,10:13])

#Dependent Variable
Y_multiple=np.array(data.price)

#Create the model
X_train,X_test,Y_train,Y_test=train_test_split(X_multiple,Y_multiple, test_size=0.2)
X= sm.add_constant(X_train, prepend=True)
rlm = sm.OLS(endog=Y_train, exog=X,)
rlm = rlm.fit()
print(rlm.summary())
#%%Calcular el error
Y_pred=rlm.predict(X)
error=Y_train-Y_pred
#%% Forma Estadística de Homocedasticidad
#Breusch-Pagan
#H0: Homocedasticidad (p>0.05)
#H1: No homocedasticidad (p<0.05)
names=['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']
test = sms.het_breuschpagan(error, X)
lzip(names, test)
print(lzip(names, test))
#%% Forma estadística de la normalidda (Shapiro-Wilk)
#Ho: Normalidad (p>0.05)
#H1: No normalidad (p<0.05)
names=[' Statistic', 'p-value']
test=stats.shapiro(error)
lzip(names,test)
print(lzip(names, test))