import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model

# Récupération des données
# bostonHousePrices = pandas.read_csv('boston_house_prices.csv')
# positionSalaries = pandas.read_csv('Position_Salaries.csv')
# qualiteVinRouge = pandas.read_csv('qualite_vin_rouge.csv')
regSimple = pandas.read_csv('reg_simple.csv')

# Visualisation des données
print(regSimple)


# np.random.seed(0)

# n_samples = 27
# x = np.linspace(0, 100, n_samples).reshape(n_samples, 1)
# y = x + np.random.randn(n_samples, 1)

# plt.scatter(x, y)
# plt.show()

# X = np.hstack((x, np.ones(x.shape)))
# print(X.shape)

# theta = np.random.randn(2, 1)
# print(theta)

# def model(X, theta):
#     return X.dot(theta)

# def cost_function(X, y, theta):
#     n = len(y)
#     return 1/(2*n) * np.sum((model(X, theta) - y)**2)

# def grad(X, y, theta):
#     n = len(y)
#     return 1/m * X.T.dot(model(X, theta) - y)

# def gradient_descent(x, y, theta, learning_rate, n_iterations):
#     cost_history = np.zeros(n_iterations)

#     for i in range(0, n_iterations):
#         theta = theta - learning_rate * grad(x, y, theta)
#         cost_history[i] = cost_function(x, y, theta)

#     return theta, cost_history

# # Example de test :
# print(cost_function(X, y, theta)) # pas d'erreur, retourne float, ~ 1000

# n_iterations = 1000
# learning_rate = 0.01
 
# theta_final, cost_history = gradient_descent(X, y, theta, learning_rate, n_iterations)
 
# print(theta_final) # voici les parametres du modele une fois que la machine a été entrainée
 
# # création d'un vecteur prédictions qui contient les prédictions de notre modele final
# predictions = model(X, theta_final)
 
# # Affiche les résultats de prédictions (en rouge) par rapport a notre Dataset (en bleu)
# plt.scatter(x, y)
# plt.plot(x, predictions, c='r')
# plt.show()


# x = np.array(df['heure_rev'])
# y = np.array(df['note'])
# plt.scatter(x,y)
# plt.show()
# print(x.shape)
# x = x.reshape(x.shape[0], 1)
# y = y.reshape(y.shape[0], 1)
# print(y.shape)
# print("=============================================")
# x = np.hstack((x, np.ones(x.shape)))
# print(x)
# print("=============================================")
# theta = np.random.randn(2, 1)
# print(theta)
# print("=============================================")
# plt.scatter(x[:,0],y)
# plt.plot(x[:,0], model(x,theta), c='r')
# plt.show()
# print("=============================================")
# # print(cost_function(x, y, theta))
# theta_final , cost_history= gradient_descent(x, y, theta, 0.001, 1000)
# print(theta_final)
# predictions = model(x, theta_final)
# plt.scatter(x[:,0], y)
# plt.plot(x[:,0], predictions, c='r')
# plt.show()
# plt.plot(range(1000), cost_history)
# plt.show()
# print(coef_determination(y, predictions))