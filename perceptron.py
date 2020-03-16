import numpy as np
import scipy.linalg as la
import pdb
from sklearn.model_selection import KFold
import itertools
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA

class Classifier:
    def __init__(self):
        pass

    def fit(self):
        pass

    def predLabel(self):
        pass

class Perceptron(Classifier):
	def __init__(self):
        	"""Constructor de la clase """
        
      		pass
        def fit(self,X,y,k):
                """Entrena el clasificador
                X: matriz numpy cada fila es un dato, cada columna una medida
                """
		tags=set(y)
                W=np.zeros((len(tags),len(X[0])))
		
		mu=float(1)/float(k)
               	X_train=[]
		X_test=[]
		y_train=[]
		y_test=[]
		
		X_effective=[]
		X_internal=[]
		y_effective=[]
		y_internal=[]

	
		kf=KFold(n_splits=k, shuffle=True) #Realizamos el k-fold 10
		for train_index, test_index in kf.split(X):
			 X_train.append( X[train_index])
			 y_train.append( y[train_index])
			 X_test.append( X[test_index])
			 y_test.append( y[test_index])
		error2=0
		for u in range(1):
			kf2=KFold(n_splits=k, shuffle=True) #Tomamos el 90 porciento para aprendizaje efectivo y el 10 porciento para validacion interna
			# bucle
		
			for effective_index, internal_index in kf.split(X_train[u]):
				 X_effective.append( X_train[u][effective_index])
				 y_effective.append( y_train[u][effective_index])
				 X_internal.append( X_train[u][internal_index])
				 y_internal.append( y_train[u][internal_index])	
			
			x_axis=[]
			y_axis=[]
			y_axis2=[]
			it=0
			while it<50 : 
				
				
				x_axis.append(it)
				for i in range(len(X_effective[u])):
					alfa=[]	
					for j in range(len(tags)):
						alfa.append(np.dot(W[j],X_effective[u][i].T))				
					
					#print(alfa)
					chosen=np.argmax(alfa) #la clasificamos como aquella de valor mas alto
				
				
				
					if y_effective[u][i]!=list(tags)[chosen]:
					
						W[list(tags).index(y_effective[u][i])]+=mu*X_effective[u][i] #incrementamos usando mu la que deberia ser
					
						W[list(set(range(len(tags)))-set([list(tags).index(y_effective[u][i])]))]-=mu*X_effective[u][i] #decrementamos las incorrectas
			
			
				alfas= np.dot(W,X_internal[u].T)#actualizamos las clases discriminantes
				ch=np.argmax(alfas,axis=0)
				aprendizaje=float(sum(i == j for i, j in zip(np.array(list(tags))[ch], y_internal[u]))*100)/len(X_internal[u])
				error=float(sum(i != j for i, j in zip(np.array(list(tags))[ch], y_internal[u]))*100)/len(X_internal[u])
				y_axis.append(error)
				y_axis2.append(aprendizaje)
				it+=1
			
			fig = plt.figure()
			plt.plot(x_axis,y_axis)
			fig.suptitle('error interno', fontsize=20)
			plt.xlabel('iteracion', fontsize=18)
			plt.ylabel('error', fontsize=16)
			fig.savefig('ir.jpg')	
			fig = plt.figure()
			plt.plot(x_axis,y_axis2)
			fig.suptitle('aprendizaje', fontsize=20)
			plt.xlabel('iteracion', fontsize=18)
			plt.ylabel('acierto', fontsize=16)
			fig.savefig('ar.jpg')	
	
		
			alfas= np.dot(W,X_test[u].T)
			ch=np.argmax(alfas,axis=0)
		
			error2+=float(sum(i != j for i, j in zip(np.array(list(tags))[ch], y_test[u]))*100)/len(X_test[u])
		print("ERROR:"+str(error2/10)+"%")
				
		
				
				
				
			
				
					
			


					
					
				
                return self
	
    

 #########################################################

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
train_samples_raw = np.fromfile('train-images-idx3-ubyte/data',dtype=np.uint8)[16:]
test_samples_raw = np.fromfile('t10k-images-idx3-ubyte/data',dtype=np.uint8)[16:]
y = np.fromfile('train-labels-idx1-ubyte/data',dtype=np.uint8)[8:]
test_labels = np.fromfile('t10k-labels-idx1-ubyte/data',dtype=np.uint8)[8:]

# Reorganizo la matriz de datos para que tenga la forma esperada
# 28-filas x 28-columnas x N-datos 
train_samples = np.swapaxes(np.reshape(train_samples_raw,(28,28,60000),order='F'),0,1)
test_samples = np.swapaxes(np.reshape(test_samples_raw,(28,28,10000),order='F'),0,1)

X=  np.reshape(train_samples, (28*28,60000)).T
pca = PCA(n_components=2)
X = pca.fit(X).transform(X)

clas=Perceptron()
clas.fit(X,y,10)
