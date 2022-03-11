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
	


def flatten_nd(x):
	s=x.shape
	newshape=s[:-2]+(s[-2]*s[-1],)
	return jnp.reshape(x,newshape)

def separate_n_d(x,n,d):
	s=x.shape
	newshape=s[:-1]+(n,d)
	return jnp.reshape(x,newshape)
	


#def L2(functions,X_dist,X_density,n_samples,key):
#	X_dist()
#
#def L2_from_data(Y,X,X_density):
#
#	densities=jnp.repeat(jnp.expand_dims(X_density(X),axis=0),Y.shape[0],axis=0)
#	return jnp.average(Y.square/densities,axis=1)
#	
#
#def check_L2(X,X_density,key,n_centers=5):
#
#	(n_samples,dim)=X.shape
#	
#	centers=jax.random.normal(n_centers,dim)
#	centers_=jnp.repeat(jnp.expand_dims(centers,axis=1),n_samples,axis=1)
#	X_=jnp.repeat(jnp.expand_dims(X,axis=0),n_centers,axis=0)
#
#	displacements=X_-centers_
#	Y=jnp.exp(-jnp.sum(displacements.square,axis=-1)/2)/jnp.sqrt(2*math.pi)**dim
#
#	l2=L2_from_data(Y,X,X_density)
#	np.testing.assert_allclose(l2,jnp.ones(n_centers),rtol=.01)


pwr=lambda x,p:jnp.power(x,p*jnp.ones(x.shape))



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


def pairwisediffs(X):
	n=X.shape[-2]
	stacked_x_1=jnp.repeat(jnp.expand_dims(X,-2),n,axis=-2)
	stacked_x_2=jnp.swapaxes(stacked_x_1,-2,-3)
	return stacked_x_1-stacked_x_2

def pairwisedists(X):
	return jnp.sum(jnp.square(pairwisediffs(X)),axis=-1)



def Coulomb(X):
	energies=jnp.triu(1/pairwisedists(X),k=1)
	return jnp.sum(energies,axis=(-2,-1))


def mindist(X):
	energies=jnp.triu(1/pairwisedists(X),k=1)
	return 1/jnp.max(energies,axis=(-2,-1))
	



def savedata(data,filename):
        filename='data/'+filename
        with open(filename,'wb') as file:
                pickle.dump(data,file)

def getdata(filename):
	with open('data/'+filename,"rb") as file:
		data=pickle.load(file)
	return data

def rangevals(_dict_):
	range_vals=jnp.array([[k,v] for k,v in _dict_.items()]).T
	return range_vals[0],range_vals[1]

def saveplot(datanames,savename):
	plt.figure()
	plt.yscale('log')
	for filename in datanames:
		data=getdata(filename)
		plot_dict(data)
	plt.savefig('plots/'+savename+'.pdf')
			
def plot_dict(_dict_):
	_range,sqnorms=rangevals(_dict_)
	plt.scatter(_range,sqnorms,color='r')
	plt.plot(_range,sqnorms,color='r')



"""
def plot(x,y):
	plt.figure()
	plt.yscale('log')
	plt.plot(x,y,color='b')
	plt.scatter(x,y,color='r')
	plt.show()
"""


def intersect(x,y):
	return range(max(x[0],y[0]),min(x[-1],y[-1])+1)




def compare(x,y):
	rel_err=jnp.linalg.norm(y-x,axis=-1)/jnp.linalg.norm(x,axis=-1)
	print('maximum relative error')
	print(jnp.max(rel_err))
	print()
