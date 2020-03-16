import numpy as np
import scipy.linalg as la
import pdb
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
        
        
      		pass
        def fit(self,X,y,k):
               
                clusters=[]
		clabels=[]
		optimal=True #True para iniciacion optima, false para random
               	
		if k!=1:
			if optimal==False:
			
				import random
				SEED=448
				random.seed(SEED) #mezclado inicial de los datos para evitar regularidades de la organizacion inicial
				s = list(range(len(X)))
				random.shuffle(s)
			
				X=X[s]
				y=y[s]
				lg=len(X)/k
			
				i=0
			
				for i in range(k-1): #creamos k clusters
		
					clusters.append(X[i*lg:(i*lg)+lg])
					clabels.append(y[i*lg:(i*lg)+lg])
				f=(i*lg)+lg #el ultimo cluster puede no contener el mismo numero de datos que los demas
				clusters.append(X[f:len(X)])
				clabels.append(y[f:len(X)])
			else:#INICIALIZACION OPTIMA
				X_temp=X
				X=list(X)
				centroides=[]
				centroides.append(X[0]) #Cojemos un dato al azar como primer centroide (en este caso el primero)
				X.pop(0)
				for i in range(k): #calculamos los k-1 puntos que vayan estando mas alejados del conjunto de centroides ya seleccionados
					dist=np.matrix(centroides)*np.matrix(X).transpose()-0.5*(np.matrix(np.diag(np.matrix(centroides)*np.matrix(centroides).transpose())).T*np.ones((1,len(X))))
					s=np.sum(dist,axis=1)
	
					centroides.append(X[list(s).index(max(s))])
					X.pop(list(s).index(max(s)))	
				X=X_temp
				
			
			          	
				clusters=[]
				labels=[]
		
		
			
			
				for i in range (k): #organizamos los datos restantes utilizando estos centroides
					clusters.append(X[np.where(dist.argmax(axis=0)==i)[1]][0])
					labels.append(y[np.where(dist.argmax(axis=0)==i)[1]][0])
				
		else:  #si k es igual a 1 solo hay un cluster
			clusters=[X]
			clabels=[y]
			
		
		
		cont=0
		
		while True :
			centroides=[]
			cont+=1
			
			
		
		

			for i in range(len(clusters)): #calculamos los centroides 
				centroides.append(np.mean(clusters[i], axis=0)) 
			
			dist=np.zeros((len(X),len(centroides)))
		
			
					
				#calculamos las distancias a los centroides  y las guardamos en dist	
			
			
                	dist=np.matrix(centroides)*np.matrix(X).transpose()-0.5*(np.matrix(np.diag(np.matrix(centroides)*np.matrix(centroides).transpose())).T*np.ones((1,len(X))))
			
			          	
			newClusters=[]
			newClabels=[]
		
			iguales=True
			
			
			for i in range (k): #reorganizamos los clusters conforme a las distancias obtenidas
				newClusters.append(X[np.where(dist.argmax(axis=0)==i)[1]][0])
				newClabels.append(y[np.where(dist.argmax(axis=0)==i)[1]][0])
				if np.all(newClusters[i]!=clusters[i]): #si los clusters han convergido finalizamos
					iguales=False
			
			if iguales or cont>50:
				
				break
			else:	#si no han convergido los actualizamos		
				clusters=newClusters
				clabels=newClabels
			
		
		comp=compactacion(clusters,clabels) #calculamos el indice de compactacion
		self.comp=comp
		dunn=DunnIndex(clusters) #calculamos el indice de Dunn
		fisher=self.FisherRatio(clusters) #calculamos el ratio de Fisher
		
		
		self.dunn=dunn
		self.fisher=fisher
			
		
					
					
				
                return self
	def FisherRatio(self,clusters):
	

		dists = np.ones([len(clusters), len(clusters)])*9876
		
   
    
		for i in range(len(clusters)): #calculamos las distancias entre los clusters
			for j in range(len(clusters)):
				if i!=j:
					dists[i, j] = distance(clusters[i], clusters[j])
        
		
		minim=np.min(dists)
		if minim==9876:
			minim=0
		
		fisher = minim/self.comp
		return fisher
	
	

	

		



class clasifSequential(Classifier):
	def __init__(self):
        	"""Constructor de la clase """
        
      		pass
        def fit(self,X,t,q,y):
                """Entrena el clasificador
                X: matriz numpy cada fila es un dato, cada columna una medida
                """
                print("hola")
               	m=1
		clusters={}
		clabels={}
		centroides=[]	
		
		clusters[0]=[X[0]]
		clabels[0]=[y[0]]
		centroides.append(X[0])
		
		
		for i in range(1,len(X)):
			dist=np.zeros(m)
		
			for j in range(m):
					
			#calculamos las distancias euclideas y las guardamos en dist
                        	
                		dist[j]=np.linalg.norm(np.array(centroides[j])-np.array(X[i]))
			              	
			ck=dist.argmin()
			
			
			if (dist[ck]>=t) and m<q: #si supera el umbral y no el numero maximo de clusters, creamos otro cluster
				m+=1
				
				clusters[m-1]=[X[i]]
				clabels[m-1]=[y[i]]
				centroides.append(X[i])
			else:
				clusters[ck]+=[X[i]]
				clabels[ck]+=[y[i]]
				centroides[ck]=(np.mean(clusters[ck], axis=0))
			
			
		  	
		
	
		
		
		
		comp=compactacion(clusters,clabels)
		print(comp)
		self.comp=comp
		dunn=DunnIndex(clusters)
		fisher=self.FisherRatio(clusters)
		
		
		self.dunn=dunn
		self.fisher=fisher
			
	
				
				
					
				
				
				
			

		
		
                return self

	def FisherRatio(self, clusters):
	

		dists = np.ones([len(clusters), len(clusters)])*9876
		diams = np.zeros([len(clusters), 1])
   
    
		for i in range(len(clusters)):
			for j in range(len(clusters)):
				if i!=j:
					dists[i, j] = distance(clusters[i], clusters[j])
        
		
		minim=np.min(dists)
		if minim==9876:
			minim=0
	
		fisher = minim/internalStandardVar(clusters)
		return fisher

def compactacion(clusters,clabels):
	index=0
	for j in range(len(clusters)):
		
		
		if len(clabels[j])==0:
			index+=0
		else: #calculamos el indice de compactacion como el porcentaje de datos con la etiqueta moda en el cluster
			
			index+=float(list(clabels[j]).count(max(set(clabels[j]), key=list(clabels[j]).count)))/float(len(clabels[j]))
		
			
	return float(index)/float(len(clusters))
					


def distance(c1, c2): #distancia entre clusters
	import scipy.spatial.distance as scipy

	if ( type(c1)==list and (len(c1)==0 or len(c2)==0)) or (type(c1)!=list and (c1.size==0 or c2.size==0)):
		return 99999
	dist=scipy.cdist(c1, c2, metric='euclidean')
        
	return np.min(dist)
	 
    		
    
def diameter(c): #diametro del cluster
	import scipy.spatial.distance as scipy

	if ( (type(c)==list) and (len(c)==0 )) or  (type(c)!=list and (c.size==0 )):
		return -1
	diam=scipy.cdist(c, c, metric='euclidean')
	
          
	return np.max(diam)
	
    		



    
def DunnIndex(clusters):
	
	import itertools

	dists = np.ones([len(clusters), len(clusters)])*9876
	diams = np.zeros([len(clusters), 1])
   
    
	for i in range(len(clusters)):
		for j in range(len(clusters)):
			if i!=j:
				dists[i, j] = distance(clusters[i], clusters[j])
        
		diams[i] =diameter(clusters[i])
	
	minim=np.min(dists)
	if minim==9876:
		minim=0
	dunn = minim/np.max(diams)
	return dunn

def interdistancia(X):
	dist=np.zeros((len(X),len(X)))

	for i in range(0, len(X)):
		for j in range(0, len(X)):
			dist[i, j] = np.linalg.norm(X[i]-X[j])
	return np.mean(dist)

def internalStandardVar(clusters):
	var=0
	for i in range(len(clusters)):
		if  ( type(clusters[i])==list and len(clusters[i])!=0 ) or  (type(clusters[i])!=list and clusters[i].shape!=(0,4)):
			var+=np.mean(np.std(clusters[i],axis=0))
	
	return float(var)/float(len(clusters))

    

 #########################################################

