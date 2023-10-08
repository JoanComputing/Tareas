# -*- coding: utf-8 -*-
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
# Parte 1:

size=1000
media_vector1=np.array([-1,-4])
media_vector2=np.array([3,-7])
covariance_matrix=np.array([[1,0.6],[0.6,1]])
distribucion1=np.random.multivariate_normal(media_vector1,covariance_matrix,size=size)
distribucion2=np.random.multivariate_normal(media_vector2,covariance_matrix,size=size)
mean1=np.mean(distribucion1,axis=0)
mean2=np.mean(distribucion2,axis=0)
print(mean1)
print(mean2)
"""
Al calcular las medias de las distribuciones con np, notamos que estas son bastantes similares
con las que definimos
"""
# Parte 2:
    
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.histplot(data=distribucion1,ax=ax[0])
sns.histplot(data=distribucion2,ax=ax[1])
ax[0].set_xlabel("Columna 1"),ax[0].set_ylabel("Columna 2"),ax[0].set_title("Distribución normal bivariada con media"+" "+
    str(media_vector1[0])+" y"+" "+str(media_vector1[1]))
ax[1].set_xlabel("Columna 1"),ax[1].set_ylabel("Columna 2"),ax[1].set_title("Distribución normal bivariada con media"+" "+
    str(media_vector2[0])+" y"+" "+str(media_vector2[1]))
fig.savefig("P1-2.png")

"""
Al graficar las distribuciones, se puede ver que las medias de estas estan cerca de las que definimos
"""


# Parte 3:
    
data=np.concatenate((distribucion1,distribucion2),axis=0)
df=pd.DataFrame(data,columns=["Columna1","Columna2"])
zeros=np.zeros(len(df)//2,dtype=int)
ones=np.ones(len(df)//2,dtype=int)
data2=np.concatenate((zeros,ones))
df=df.assign(Clase=data2)

# Parte 4:
fig2,ax2=plt.subplots(figsize=(10,5))
sns.histplot(data=df,x='Columna1',y='Columna2',hue='Clase')
plt.title("Histograma Bidimensional con Clases")
plt.show()
fig2.savefig("P1-4.png")

"""
Observando el grafico podemos ver que los datos graficados se encuentran separados
por una diagonal, esto significa que con una correcta manipulacion de los datos,
estos podrian ser separados de buena manera, pues no se solapan entre si
"""