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
import util
import jax
import jax.numpy as jnp
import optax
	



#############################################################

odd_angle=lambda x:(jnp.abs(x-1)-abs(x+1)+2*x)/4
ReLU=lambda x:(jnp.abs(x)+x)/2
activation=ReLU
oscillating=lambda x:jnp.sin(100*x)
#activation=lambda x:x

odd_angle_leaky=lambda x:odd_angle(x)+.1*x
ReLU_leaky=lambda x:ReLU(x)+.1*x



box=lambda X:jnp.product(jnp.product(jnp.heaviside(1-jnp.square(X),0),axis=-1),axis=-1)

envelope=lambda X:jnp.exp(-jnp.sum(jnp.square(X)))
envelope_FN=envelope

#envelope=box
#envelope_FN=box





#apply_tau=lambda W,X:ReLU(jnp.matmul(jax.lax.collapse(W,1,3),jax.lax.collapse(X,1,3).T))
apply_tau_=lambda W,X,activation=ReLU:activation(jnp.matmul(util.flatten_nd(W),util.flatten_nd(X).T))
apply_tau=apply_tau_

def w_to_alpha(W,activation):
	F=lambda X:apply_tau_(W,X,activation)
	return antisymmetrize(F)

def apply_alpha(W,X,activation=ReLU):
	alpha_w=w_to_alpha(W,activation)
	return alpha_w(X)






def antisymmetrize(f):
	def antisymmetric(X):
		y=jnp.zeros(f(X).shape)
		n=X.shape[-2]
		for P in itertools.permutations(jnp.identity(n)):
			sign=jnp.linalg.det(P)
			PX=jnp.swapaxes(jnp.dot(jnp.array(P),X),0,-2)
			y+=sign*f(PX)
		return y/jnp.sqrt(math.factorial(n))
	return antisymmetric
	




class Simple:
	def __init__(self,params,randomness_key,normalize=False):
		self.params=params
		d,n,instances=params['d'],params['n'],params['instances']
		key,*subkeys=jax.random.split(randomness_key,4)

		self.W=jax.random.normal(subkeys[0],shape=(instances,n*d))*jnp.sqrt(2/(d*n))
		self.a=jax.random.normal(subkeys[1],shape=(instances,))*jnp.sqrt(2)
		if(normalize):
			self.W=normalize_rows(self.W)
			self.a=jnp.ones(shape=(instances,))

	def evaluate(self,X):
		X_vec=jnp.ravel(X)
		return jnp.multiply(self.a,activation(jnp.dot(self.W,X_vec)))


def distribution(f,X_distribution,samples=100):
	key=jax.random.PRNGKey(np.random.randint(100))
	
	X_list=X_distribution(key,samples)
	y_list=jax.vmap(f)(X_list)

	return y_list




def normalize(X):
	return lambda x:x/jnp.sqrt(jnp.sum(jnp.square(x)))(X)

def normalize_rows(X):
	return jax.vmap(lambda x:x/jnp.sqrt(jnp.sum(jnp.square(x))))(X)

	


def Gaussian(n,d):
	return (lambda key,samples:jax.random.normal(key,shape=(samples,n,d)))

def spherical(n,d,radius=1): 
	g=Gaussian(n,d)
	return (lambda key,samples:normalize_rows(g(key,samples))*radius)



class TwoLayer:
	def __init__(self,params,randomness_key):
		self.params=params
		d,n,m=params['d'],params['n'],params['m']
		key,*subkeys=jax.random.split(randomness_key,4)

		self.W=jax.random.normal(subkeys[0],shape=(m,n*d))*jnp.sqrt(1/(d*n))
		self.a=jax.random.normal(subkeys[1],shape=(m,))*jnp.sqrt(1/m)

	def evaluate(self,X):
		X_vec=jnp.ravel(X)
		layer1=activation(jnp.dot(self.W,X_vec))
		return jnp.dot(self.a,layer1)
