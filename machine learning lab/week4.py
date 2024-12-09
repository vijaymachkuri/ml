import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Train the regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Define a function to visualize results
def plot_results(X, y, title, xlabel, ylabel, regressor, X_train=None):
    plt.scatter(X, y, color='red')
    plt.plot(X_train, regressor.predict(X_train), color='blue')  # Use training data for the regression line
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# Visualize training set results
plot_results(X_train, y_train, "Salary vs Experience (Training set)", 
             "Years of Experience", "Salaries", regressor, X_train)

# Visualize test set results
plot_results(X_test, y_test, "Salary vs Experience (Test set)", 
             "Years of Experience", "Salaries", regressor, X_train)
