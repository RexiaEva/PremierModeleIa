import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy import stats

# Récupération des données
# bostonHousePrices = pandas.read_csv('boston_house_prices.csv')
# positionSalaries = pandas.read_csv('Position_Salaries.csv')
# qualiteVinRouge = pandas.read_csv('qualite_vin_rouge.csv')
regSimple = pd.read_csv('reg_simple.csv')

# Visualisation des données
print(regSimple)
#selection de la première colonne de notre dataset (la taille de la population)
X = regSimple.iloc[0:len(regSimple),0]
#selection de deuxième colonnes de notre dataset (le profit effectué)
Y = regSimple.iloc[0:len(regSimple),1] 
axes = plt.axes()
axes.grid() # dessiner une grille pour une meilleur lisibilité du graphe
plt.scatter(X,Y) # X et Y sont les variables qu'on a extraite dans le paragraphe précédent




# Création du modèle (model(X,theta))
theta = (random.randint(0, 100), random.randint(0,100))
modele = X.apply(lambda x: theta[0] * x + theta[1])
plt.plot(X, modele, c='r')




# fonction coût
M = len(regSimple)
Sigma = 0
for i in range(1, M):
   Sigma += (modele.iat[i, 0] - Y.iat[i, 0])**2
J = (1/2*M)*Sigma




# Gradient


plt.show()