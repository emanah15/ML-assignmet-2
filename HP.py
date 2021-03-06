# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 11:21:14 2020

@author: ghulam
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataset = pd.read_csv('housing price.csv')
X = dataset.iloc[:, :1]
Y = dataset.iloc[:, 1:2]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, Y_train, color = 'pink')
plt.plot(X_train, regressor.predict(X_train), color = 'red')
plt.title(' (Training set)')
plt.xlabel('ID ')
plt.ylabel('SALE PRICE')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, Y_test, color = 'brown')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('(Test set)')
plt.xlabel('ID ')
plt.ylabel('SALE PRICE')
plt.show()

print('The sale price of ID 2950 is:')
print(regressor.predict([[2950]]))

print('The sale price of ID 3500 is:')
print(regressor.predict([[3500]]))

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))

