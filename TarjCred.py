# -*- coding: utf-8 -*-
"""
Default of Credit Card Clients
Nombre Estudiantes:
    Víctor Manuel Arroyo Martín
    Sergio Cabezas González de Lara
"""

import numpy as np
import matplotlib.pyplot as plt
import math as m
import pandas as pd
import seaborn as sns

from mpl_toolkits import mplot3d
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.utils.fixes import loguniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Perceptron, Lasso, LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, VarianceThreshold, SelectKBest
# Fijamos la semilla
np.random.seed(1)

def readData(name):
    # Leemos los ficheros  
    df = pd.read_excel (name)
    df = df.drop([0])
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    dfy = df['Y']
    del df['Y']
    dfy=dfy.astype('int')
    return df, dfy


x, y = readData('./datos/default of credit card clients.xls')

# 70% training 30% test
# Lectura de los datos de entrenamiento
# Lectura de los datos para el test

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.30, random_state=1)

# Matriz de correlación y su gráfica
corrMatrix = x_train.astype('float64').corr()

plt.figure(figsize=(10,10))
plt.title('The Correlation', y=1.05, size=15)
sns.heatmap(corrMatrix,linewidths=0.1,vmax=1.0, square=True, cmap="YlGnBu", linecolor='white', annot=True)
plt.show()

# Matriz triangular superior de la matriz de correlación
upper = corrMatrix.where(np.triu(np.ones(corrMatrix.shape), k=1).astype(np.bool))

# Características con correlacion lineal con otras mayor que 0.8
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
# Eliminar características
for i in to_drop:
    del x_train[i]
    del x_test[i]
    

input("\n--- Pulsar tecla para continuar ---\n")
#Preprocesado de los datos
preproc=[("standardize", StandardScaler()),
         ("var", VarianceThreshold(0.1)),
         ("poly", PolynomialFeatures(2))
         ]


pipe=Pipeline([('lr', LogisticRegression())])
#CV con varios modelos
params_grid=[
        {"lr":[LogisticRegression(penalty='l1',max_iter=500)],
                "lr__C":np.logspace(-2,2,5),
                "lr__solver":['lbfgs']},
       {"lr": [RandomForestClassifier(random_state = 1,
                                       n_jobs = -1, criterion = 'entropy')],
         "lr__n_estimators": [100, 200],
         "lr__max_depth": [6, 8]},
        {"lr": [SVC(kernel='rbf', gamma='scale', max_iter=1000, degree=2)],
               "lr__C":np.logspace(-2,2,5)},
        {"lr": [MLPClassifier(random_state=1, max_iter=1000, hidden_layer_sizes=60)],
           "lr__solver":['sgd'],
           "lr__activation":['relu','logistic']}
            
]

#Mostramos el mejor usando de scoring (métrica)
best_lr=GridSearchCV(pipe,params_grid, scoring='accuracy', cv=5, n_jobs=-1)
best_lr.fit(x_train,y_train)

print("Parámetros del mejor clasificador:\n{}".format(best_lr.best_params_))
print("Accuracy en CV: {:0.3f}%".format(100.0 * best_lr.best_score_))
print("Accuracy en training: {:0.3f}%".format(
        100.0 * best_lr.score(x_train, y_train)))
print("Accuracy en test: {:0.3f}%".format(
        100.0 * best_lr.score(x_test, y_test)))


input("\n--- Pulsar tecla para continuar ---\n")
#PCA con 3 dimensiones
pca = PCA(n_components=3)
pca.fit(x_train)
x = pca.transform(x_train)

# Creamos la figura
fig = plt.figure()
# Creamos el plano 3D
ax1 = fig.add_subplot(111, projection='3d')

ax1.plot(np.squeeze(x[np.where(y_train==0),0]), np.squeeze(x[np.where(y_train == 0),1]), np.squeeze(x[np.where(y_train == 0),2]), 'o', color='green', label='0')
ax1.plot(np.squeeze(x[np.where(y_train==1),0]), np.squeeze(x[np.where(y_train== 1),1]), np.squeeze(x[np.where(y_train == 1),2]), 'o', color='red', label='1')

plt.legend()

plt.show()

input("\n--- Pulsar tecla para continuar ---\n")
#PCA con 2 dimensiones
pca = PCA(n_components=2)
pca.fit(x_train)
x = pca.transform(x_train)
# Creamos la figura
fig = plt.figure()
# Creamos el plano 3D
ax1 = fig.add_subplot(111)

ax1.plot(np.squeeze(x[np.where(y_train==0),0]), np.squeeze(x[np.where(y_train == 0),1]), 'o', color='green', label='0')
ax1.plot(np.squeeze(x[np.where(y_train==1),0]), np.squeeze(x[np.where(y_train== 1),1]), 'o', color='red', label='1')

plt.legend()
