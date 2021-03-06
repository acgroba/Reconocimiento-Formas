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

class clasifKmeans(Classifier):
	def __init__(self):
        	"""Constructor de la clase """
        
      		pass
        def fitBoosting(self,X,y,k, M):
                """Entrena el clasificador
                X: matriz numpy cada fila es un dato, cada columna una medida
                """
		
		X_train=[]
		X_test=[]
		y_train=[]
		y_test=[]
		
		kf=KFold(n_splits=10, shuffle=True)
		for train_index, test_index in kf.split(X):
			X_train.append( X[train_index])
			y_train.append( y[train_index])
			X_test.append( X[test_index])
			y_test.append( y[test_index])
		err=0
		
		for u in range(10):
			X_train[u]=np.array(X_train[u])
			y_train[u]=np.array(y_train[u])
			tags=set(list(y_train[u]))
			Ws=[]
			W=np.zeros((len(tags),len(X_train[u][0])))
			for z in range(M):
				
				s = list(range(len(X_train[u])))
				random.shuffle(s)
				
				X_temp=X_train[u]
				Y_temp=y_train[u]
				X_train[u]=X_train[u][s]
				
				y_train[u]=y_train[u][s]
				
				
				
				
				
        		       
				
				mu=float(1)/float(k)
        		       
				
				X_effective=[]
				X_internal=[]
				y_effective=[]
				y_internal=[]
		
			
			
				
			
				kf2=KFold(n_splits=10, shuffle=True)
				# bucle
				
				for effective_index, internal_index in kf.split(X_train[u]):
					
					 X_effective.append( X_train[u][effective_index])
					 y_effective.append( y_train[u][effective_index])
					 X_internal.append( X_train[u][internal_index])
					 y_internal.append( y_train[u][internal_index])	
				
				
				it=0
				while it<60 :
					
					
					for i in range(len(X_effective[u])):
						alfa=[]	
						for j in range(len(tags)):
							
							alfa.append(np.dot(W[j],X_effective[u][i].T))				
						
						#print(alfa)
						chosen=np.argmax(alfa)
					
					
					
						if y_effective[u][i]!=list(tags)[chosen]:
						
							W[list(tags).index(y_effective[u][i])]+=mu*X_effective[u][i]
						
							W[list(set(range(len(tags)))-set([list(tags).index(y_effective[u][i])]))]-=mu*X_effective[u][i]
				
				
					
					it+=1
				
				
				X_train[u]=np.array(X_temp)
				y_train[u]=np.array(Y_temp)
				
				
				
			
				if z>0:	
			
					X_train[u]=X_temp
					y_train[u]=Y_temp
				else:
					Ws.append(W)
			chos=[]
			for w in Ws:
				alfas= np.dot(w,X_train[u].T)
		        	chos.append(np.argmax(alfas,axis=0))
		
			 	
			

			
			
			pred=map(lambda x: max(set(list(x)), key=list(x).count), chos)
			miss=[]
			co=0
			
			
			for i, j in zip(np.array(list(tags))[pred], y_train[u]):
							
				if i!=j:
					miss.append(co)
				co+=1
				
			
			err+=float(len(miss))/len(y_train[u])
			
			X_temp=X_train[u]
			Y_temp=y_train[u]
						
			X_train[u]=X_train[u][miss]
			
			y_train[u]=y_train[u][miss]
				
		return (err/10)
			

		
					
				
				
			
				
					
			


 #########################################################

