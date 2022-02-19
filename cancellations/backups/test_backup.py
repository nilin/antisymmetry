import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import matplotlib
from sympy.utilities.iterables import multiset_permutations
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pickle
import time
import copy
import jax
import jax.numpy as jnp
import optax
import cancellation as canc
import cancellation_full as full
	



key=jax.random.PRNGKey(0)
key,*subkeys=jax.random.split(key,1000)


def estimate_var(f,X_distribution,n_samples,key):

	X=X_distribution(key,n_samples)

	Y=jax.vmap(f)(X)
	variance=jnp.var(Y)

	validate(Y)

	return variance,Y

def validate(Y):
	Xs=Y.shape[0]
	fs=Y.shape[1]
	T=jnp.take(Y,np.array([0,Xs//2]),axis=0)
	B=jnp.take(Y,np.array([Xs//2,Xs]),axis=0)
	L=jnp.take(Y,np.array([0,fs//2]),axis=1)
	R=jnp.take(Y,np.array([fs//2,fs]),axis=1)

	vars_by_f=jnp.var(Y,axis=0)
	vars_by_x=jnp.var(Y,axis=1)
	print(jnp.var(vars_by_f))
	print(jnp.var(vars_by_x))
	print('')

	print(jnp.var(T))
	print(jnp.var(B))
	print(jnp.var(L))
	print(jnp.var(R))
	print('')

	print('var of first/last Xs log-ratio '+str(jnp.log(jnp.var(T)/jnp.var(B))))
	print('var of first/last fs log-ratio '+str(jnp.log(jnp.var(L)/jnp.var(R))))
	print('\n')
	





def duality(key,dist_W,dist_X,instances,samples):
	
	key,*subkeys=jax.random.split(key,4)
	W=dist_W(subkeys[0],instances)
	X=dist_X(subkeys[1],samples)

	F=lambda X:jnp.matmul(jax.lax.collapse(W,1,3),jax.lax.collapse(X,1,3).T)
	G=canc.antisymmetrize(F)

	return W,X,G(X)



def plot_duality(key):

	instances=100
	samples=100
	params={'n':5,'d':3}
	sphere=canc.spherical(params['n'],params['d'])

	W,X,Y=duality(key,sphere,sphere,instances,samples)

	dW=jnp.repeat(jnp.expand_dims(full.mindist(W),axis=1),samples,axis=1)
	dX=jnp.repeat(jnp.expand_dims(full.mindist(X),axis=0),instances,axis=0)

	dWdX=jnp.ravel(jnp.multiply(dW,dX))
	absY=jnp.ravel(jnp.abs(Y))

	print(jnp.corrcoef(dWdX,absY))

	plt.figure()
	plt.scatter(dWdX,absY,s=2)
	sns.kdeplot(dWdX,absY,color='r',bw=.2)
	plt.show()


	

def plots():

	instances=1000
	samples=1000
	d=3
	variances=[]

	for n in range(2,20):

		paramstring=''

		W_distribution=canc.Gaussian(n,d)
		X_distribution=canc.Gaussian(n,d)

		W,X,Y=duality(subkeys[n],W_distribution,X_distribution,instances,samples)
		print(Y)

		variances.append(jnp.var(Y))
		savedata(variances,'variances'+paramstring)	

		plt.figure()
		plt.plot(range(2,n+1),jnp.log(jnp.array(variances)),color='r')
		plt.plot(range(2,n+1),jnp.log(jnp.array([math.factorial(i) for i in range(2,n+1)])),color='b')
		plt.savefig('plots/vars'+paramstring+str(n)+'.pdf')




#def plots():
#
#	variances=[]
#
#	for n in range(2,20):
#
#		paramstring=''
#
#		params={'d':3,'n':n,'instances':1000}
#		X_distribution=canc.Gaussian(params['n'],params['d'])
#
#		simple=canc.Simple(params,subkeys[2*n])
#
#		f=simple.evaluate
#		g=canc.antisymmetrize(f)
#
#		var,_=estimate_var(g,X_distribution,1000,subkeys[2*n+1])
#
#		variances.append(var)
#		savedata(variances,'variances'+paramstring)	
#
#		plt.figure()
#		plt.plot(range(2,n+1),jnp.log(jnp.array(variances)),color='r')
#		plt.plot(range(2,n+1),jnp.log(jnp.array([math.factorial(i) for i in range(2,n+1)])),color='b')
#		plt.savefig('plots/vars'+paramstring+str(n)+'.pdf')


def savedata(thedata,filename):
        filename='data/'+filename
        with open(filename,'wb') as file:
                pickle.dump(thedata,file)

#plot_duality(key)	
plots()
