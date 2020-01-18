# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 11:53:47 2020

@author: ghulam
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


dataset = pd.read_csv('monthlyexp vs incom.csv')
X = dataset.iloc[:, :1].values
Y = dataset.iloc[:, 1:2].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = regressor.predict(X_test)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 8)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, Y)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)


# Visualising the Training set results
plt.scatter(X_train, Y_train, color = 'pink')
plt.plot(X_train, regressor.predict(X_train), color = 'red')
plt.title('Mothly experience vs Income')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, Y_test, color = 'brown')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Mothly experience vs Income')
plt.xlabel('Monthly Experience')
plt.ylabel('Income')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Mothly experience vs Income (Polynomial Regression)')
plt.xlabel('Monthly Experience')
plt.ylabel('Income')
plt.show()

################# checking accuracy
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
