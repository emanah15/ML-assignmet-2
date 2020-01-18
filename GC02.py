# -*- coding: utf-8 -*-
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('global_co2.csv')
X = dataset.iloc[219:,:1].values
Y = dataset.iloc[219:,1:2].values

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
poly_reg = PolynomialFeatures(degree = 6)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, Y)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

# Visualising the Data set to see the relation between X and Y
plt.scatter(X_train, Y_train, color = 'pink')
plt.plot(X_train, regressor.predict(X_train), color = 'black')
plt.title(' CO2 production from 1970s (Training set)')
plt.xlabel('Years ')
plt.ylabel('CO2 Production')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, Y_test, color = 'brown')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title(' CO2 production from 1970s (Training set)')
plt.xlabel('Years ')
plt.ylabel('CO2 Production')
plt.show()

#since the data is non linear, Polynomial egression will be applied
# Visualising the Polynomial Regression results
plt.scatter(X, Y, color = 'orange')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('CO2 production from 1970s')
plt.xlabel('Years')
plt.ylabel('CO2 Production')
plt.show()

print('CO2 production in 2011 is')
print(regressor.predict([[2011]]))
print('CO2 production in 2012 is')
print(regressor.predict([[2012]]))
print('CO2 production in 2013 is')
print(regressor.predict([[2013]]))

################# checking accuracy
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))

