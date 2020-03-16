import numpy as np
import scipy.linalg as la
import pdb
from sklearn.model_selection import KFold
import itertools as it
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

class granuladas(Classifier):
	def __init__(self):
        	"""Constructor de la clase """
        
      		pass
        def fit(self,X,y,k, X_test, y_test):
                """Entrena el clasificador
                X: matriz numpy cada fila es un dato, cada columna una medida
                """
		tags=list(set(y))
              
		
		mus=np.mean(X, axis=0)
		sigma=np.std(X, axis=0)
		
		umbrales=[mus-k*sigma, mus+k*sigma]
               	self.umbrales=umbrales
		
		N=len(tags)
		n=len(X[0])
		a=list(it.product(range(3),repeat=n))
		rules=list(it.product(a,range(N)))
		
	
		acierto=[]
	
		
	
		for rule in rules:
			correct=0
			
			for i in range(len( X)):
				pertenece=True
				for j in range(len( rule[0])):
					
					if((rule[0][j]==0 and X[i][j]>umbrales[0][j]) or (rule[0][j]==2 and X[i][j]<umbrales[1][j]) or (rule[0][j]==1 and (X[i][j]<umbrales[0][j] or  X[i][j]>umbrales[1][j]))):
						
						
						pertenece=False
					
						
					
				
				if pertenece:
					
					if y[i]==tags[int(rule[1])]:

						correct+=1
			
			acierto.append(float(correct)/float(len(X)))
		
		aciertosort=sorted(acierto)
		finalrules=[]
		for u  in range(10):
			ind=acierto.index(aciertosort.pop())
			finalrules.append(rules[ind])
			acierto[ind]=0
		
		self.finalrules=finalrules	
		
		exito=0
		tr=0

		for i  in range (len(X_test)):
			param=[]
			for j in range(len(X_test[i])):
				if X_test[i][j] <= self.umbrales[0][j]:
					param.append(0)
				else:
					if X_test[i][j]>= self.umbrales[0][j] and X_test[i][j]<=self.umbrales[1][j]:
						param.append(1)
					else:	
						param.append(2)
			c=0
			for norm in self.finalrules:
				
				if list(norm[0])==param  and y_test[i]==tags[int(norm[1])] :
					exito+=1
					
					continue
					
		
		return float(exito)/float(len(X_test))
		

					
				
				
				
			
				
	

		




 #########################################################


