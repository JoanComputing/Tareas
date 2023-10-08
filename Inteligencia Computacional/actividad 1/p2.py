# -*- coding: utf-8 -*-
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
def rotate_point(X, angle_degrees):
    """Rotates points around the origin by an angle.

    Args:
        X (np.ndarray): Array of size (n_samples, n_features), each point is a row and each feature
        a column.
        angle_degrees (float): Angle to rotate points. Must be defined in degrees.

    Returns:
        np.ndarray: Array of size (n_samples, n_features) with rotated points from input X.
    """
    # X tiene que venir en la forma (n_samples, 2)
    angle_rads = np.deg2rad(angle_degrees)
    rot_matrix = np.array([[np.cos(angle_rads), -np.sin(angle_rads)], [np.sin(angle_rads), np.cos(angle_rads)]])
    return np.matmul(rot_matrix, X.T).T

# Parte 1:
size=1000
media_vector1=np.array([-1,-4])
media_vector2=np.array([3,-7])
covariance_matrix=np.array([[1,0.6],[0.6,1]])
distribucion1=np.random.multivariate_normal(media_vector1,covariance_matrix,size=size)
distribucion2=np.random.multivariate_normal(media_vector2,covariance_matrix,size=size)
data=np.concatenate((distribucion1,distribucion2),axis=0)
data=rotate_point(data,45)
df=pd.DataFrame(data,columns=["Columna1","Columna2"])
zeros=np.zeros(len(df)//2,dtype=int)
ones=np.ones(len(df)//2,dtype=int)
data2=np.concatenate((zeros,ones))
df=df.assign(Clase=data2)
fig2,ax2=plt.subplots(figsize=(10,5))
sns.histplot(data=df,x='Columna1',y='Columna2',hue='Clase')
plt.title("Histograma Bidimensional con Clases rotado")
plt.show()
fig2.savefig("P2-1.png")

"""
Para esta parte notamos que al rotar los datos en 45 grados, es suficiente para ver que
podemos trazar una linea recta que separe a las clases, y que sea perpendicular a la columna1
"""

# Parte 2:
df=df.drop(columns='Columna2')
fig3 = plt.figure()

sns.histplot(data=df, x="Columna1", hue="Clase")
plt.title("Histograma unidimensional con Clases")
plt.show()

fig3.savefig('P2-2.png')
X=df['Columna1'] 
Y=df['Clase']
X=X.to_numpy()
Y=Y.to_numpy()

"""
Como se puede apreciar en el grafico, el haber rotado los datos en 45 grado genero que
estos esten bien divididos en sus clases, haciendo asi que no se solapen entre ellos
"""

# Parte 3:
def clasificar(X, umbral):
    return (X > umbral).astype(int)
umbral = 4.5
Y_pred=clasificar(X,umbral)

# Parte 4:
cm = confusion_matrix(Y, Y_pred)
fig4 = ConfusionMatrixDisplay(confusion_matrix=cm)
plt.figure(figsize=(10, 5))
fig4.plot()
plt.title("Matriz de confusión")
plt.savefig('P2-4.png')
"""
Al calcular la matriz de confusion para nuestros datos, notamos que la totalidad de los datos
se encuentran en Verdaderos positvos y verdaderos negativos, esto nos dice que al haber rotado
los datos, y haber hecho una clasificaciond de los datos, nuestras predicciones fueron correctas
en la totalidad de los datos.
"""