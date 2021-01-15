import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import SGDRegressor 

def model(X, theta):
    return X.dot(theta)

def cost_function(X, y, theta):
    n = len(y)
    return 1/(2*n) * np.sum((model(X, theta) - y)**2)

def grad(X, y, theta):
    n = len(y)
    return 1/n * X.T.dot(model(X, theta) - y)

def gradient_descent(x, y, theta, learning_rate, n_iterations):
    cost_history = np.zeros(n_iterations)
    for i in range(0, n_iterations):
        theta = theta - learning_rate * grad(x, y, theta)
        
        cost_history[i] = cost_function(x, y, theta)

    print(theta.shape)
    return theta, cost_history

def coef_determination(y, pred):
    u = ((y - pred)**2).sum()
    v = ((y - y.mean())**2).sum()
    return 1 - u/v

df = pandas.read_csv('reg_simple.csv')
x = np.array(df['heure_rev'])
y = np.array(df['note'])
plt.scatter(x,y)
plt.show()
print(x.shape)
x = x.reshape(x.shape[0], 1)
y = y.reshape(y.shape[0], 1)
print(y.shape)
print("=============================================")
x = np.hstack((x, np.ones(x.shape)))
print(x)
print("=============================================")
theta = np.random.randn(2, 1)
print(theta)
print("=============================================")
plt.scatter(x[:,0],y)
plt.plot(x[:,0], model(x,theta), c='r')
plt.show()
print("=============================================")
# print(cost_function(x, y, theta))
theta_final , cost_history= gradient_descent(x, y, theta, 0.001, 1000)
print(theta_final)
predictions = model(x, theta_final)
plt.scatter(x[:,0], y)
plt.plot(x[:,0], predictions, c='r')
plt.show()
plt.plot(range(1000), cost_history)
plt.show()
print(coef_determination(y, predictions))