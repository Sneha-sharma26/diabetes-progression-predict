import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

# load dataset
diabetes = datasets.load_diabetes()

#1 Plotting using 1 feature
diabetes_X = diabetes.data[:, np.newaxis,2]   # reshaped the 3rd column (2nd index column)

# splitting into training and testing data
diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]

diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-30:]             

model = linear_model.LinearRegression()

model.fit(diabetes_X_train, diabetes_y_train)        # basically we are getting the Line of the plot to test it further
diabetes_y_predicted = model.predict(diabetes_X_test)

print("Mean squared error is: ", mean_squared_error(diabetes_y_test, diabetes_y_predicted))
# output: Mean squared error is:  3035.060115291269

# to check the value difference between predicted and orginal values
print(diabetes_y_predicted[:5])
print(diabetes_y_test[:5])

print("Weights: ", model.coef_)
print("Intercept: ", model.intercept_)

plt.scatter(diabetes_X_test, diabetes_y_test)
plt.plot(diabetes_X_test, diabetes_y_predicted)
plt.show()

# -------------------------------------------------- #

# #2 Plotting using all independent features
# diabetes_X = diabetes.data

# # splitting into training and testing data
# diabetes_X_train = diabetes_X[:-30]
# diabetes_X_test = diabetes_X[-30:]

# diabetes_y_train = diabetes.target[:-30]
# diabetes_y_test = diabetes.target[-30:]             

# model = linear_model.LinearRegression()

# model.fit(diabetes_X_train, diabetes_y_train)        # basically we are getting the Line of the plot to test it further
# diabetes_y_predicted = model.predict(diabetes_X_test)

# print("Mean squared error is: ", mean_squared_error(diabetes_y_test, diabetes_y_predicted))
# # output: Mean squared error is:  1826.4841712795046      -> decreased

# # to check the value difference between predicted and orginal values
# print(diabetes_y_predicted[:5])
# print(diabetes_y_test[:5])

# print("Weights: ", model.coef_)
# print("Intercept: ", model.intercept_)

# # plotting lines will not be used as we cannot plot because of the dissimilar sizes of features(have multiple columns now) and target data(have single column)