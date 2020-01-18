

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('annual_temp.csv')
Years = dataset.iloc[::2, 1::2].values # year 
gcag = dataset.iloc[::2, 2].values 

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
Years_train, Years_test, gcag_train, gcag_test = train_test_split(Years, gcag, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(Years_train, gcag_train)

# Predicting the Test set results
Y_pred = regressor.predict(Years_test)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
Years_poly = poly_reg.fit_transform(Years)
poly_reg.fit(Years_poly, gcag)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(Years_poly, gcag)

# Visualising the Training set results
plt.scatter(Years_train, gcag_train, color = 'yellow')
plt.plot(Years_train, regressor.predict(Years_train), color = 'red')
plt.title('Annual Temp Years (Training set)')
plt.xlabel('Years ')
plt.ylabel('MEAN TEMP')
plt.show()

# Visualising the Test set results
plt.scatter(Years_test, gcag_test, color = 'brown')
plt.plot(Years_train, regressor.predict(Years_train), color = 'blue')
plt.title('Annual Temp GCAG (Test set)')
plt.xlabel('Years ')
plt.ylabel('Mean Temp')
plt.show()

#WE NEED TO HERE SOMETHING BWFORE THIS
Years_grid = np.arange(min(Years), max(Years), 0.1)
Years_grid = Years_grid.reshape((len(Years_grid), 1))
plt.scatter(Years_train, gcag_train, color = 'purple')
plt.plot(Years_grid, lin_reg_2.predict(poly_reg.fit_transform(Years_grid)), color = 'black')
plt.title('GCAG  Annual Temperature')
plt.xlabel('Years')
plt.ylabel('GCAG  Annual Temperature')
plt.show()

print(regressor.predict([[2016]])) # double bracket matrix form MATRIX INPUT
print(regressor.predict([[2017]]))

###################################################################
A = dataset.iloc[1::2, 1::2].values # year 
gist = dataset.iloc[1::2, 2].values # mean temp

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
Years_train, Years_test, gist_train, gist_test = train_test_split(Years, gist, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()
regressor1.fit(Years_train, gist_train)

# Predicting the Test set results
Y_pred2 = regressor1.predict(Years_test)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
Years_poly = poly_reg.fit_transform(Years)
poly_reg.fit(Years_poly, gist)
lin_reg_3 = LinearRegression()
lin_reg_3.fit(Years_poly, gist)

# Visualising the Training set results
plt.scatter(Years_train, gist_train, color = 'orange')
plt.plot(Years_train, regressor1.predict(Years_train), color = 'red')
plt.title('Annual Temp Years (Training set)')
plt.xlabel('Years ')
plt.ylabel('Mean Temperature')
plt.show()


Years_grid = np.arange(min(Years), max(Years), 0.1)
Years_grid = Years_grid.reshape((len(Years_grid), 1))
plt.scatter(Years_train, gist_train, color = 'cyan')
plt.plot(Years_grid, lin_reg_3.predict(poly_reg.fit_transform(Years_grid)), color = 'black')
plt.title('GIST  Annual Temperature')
plt.xlabel('Years')
plt.ylabel('GIST  Annual Temperature')
plt.show()
print('temp of GIST in 2017')
print(regressor1.predict([[2017]]))
print('temp of GCAG in 2017')
print(regressor.predict([[2017]]))

print('temp of GIST in 2016')
print(regressor1.predict([[2016]]))
print('temp of GCAG in 2016')
print(regressor.predict([[2016]]))

#accracy of GCAG
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(gcag_test, Y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(gcag_test, Y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(gcag_test, Y_pred)))

#Accuracy of GIST
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(gist_test, Y_pred2))
print('Mean Squared Error:', metrics.mean_squared_error(gist_test, Y_pred2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(gist_test, Y_pred2)))