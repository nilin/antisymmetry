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
import DPP
	


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
	



def lipschitz(f,Xdist,samples,eps,key):

	key,*subkeys=jax.random.split(key,4)
	X0=Xdist(subkeys[0],samples)
	s=X0.shape
	dXdist=canc.spherical(s[-2],s[-1],radius=eps)

	dX=dXdist(subkeys[1],samples)
	X1=X0+dX

	dY=f(X1)-f(X0)
	return jnp.max(jnp.abs(dY)/eps)
	

def pairwisedistprop(X,loss):
	n=X.shape[-2]
	stacked_x_1=jnp.repeat(jnp.expand_dims(X,-2),n,axis=-2)
	stacked_x_2=jnp.swapaxes(stacked_x_1,-2,-3)
	diffs=stacked_x_1-stacked_x_2
	dists_=jnp.sum(jnp.square(diffs),axis=-1)
	dists=jnp.take(np.partition(dists_,1,axis=-1),jnp.array([i for i in range(1,n)]),axis=-1)
	return jax.vmap(loss)(dists)

mindistsquared=lambda X:pairwisedistprop(X,jnp.min)
mindist=lambda X:jnp.sqrt(pairwisedistprop(X,jnp.min))
inverseloss=lambda x: 1/jnp.sum(1/x)




def savedata(thedata,filename):
        filename='data/'+filename
        with open(filename,'wb') as file:
                pickle.dump(thedata,file)


def plot(x,y):
	plt.figure()
	plt.yscale('log')
	plt.plot(x,y,color='b')
	plt.scatter(x,y,color='r')
	plt.show()

