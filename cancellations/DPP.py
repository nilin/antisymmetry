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
import antisymmetry.mcmc as mcmc



key=jax.random.PRNGKey(0)
key,*subkeys=jax.random.split(key,1000)


def GaussianDensity(var):
	C=1/(var*jnp.sqrt(2*math.pi))
	rho=lambda x:C**(x.shape[-1])*jnp.exp(-jnp.sum(jnp.square(x),axis=-1)/(2*var))
	return rho

"""
for translation-invariant kernel k, return a function x -> matrix K(x_i,x_j)_ij
"""
def x_to_Kmatrix_TI(k):
	def Kmatrixfunction(x):
		n=x.shape[-2]
		xs=jnp.repeat(jnp.expand_dims(x,axis=-2),n,axis=-2)
		ys=jnp.swapaxes(xs,-3,-2)
		return k(xs-ys)
	return Kmatrixfunction


"""
construct density function
"""	
def DPP_density(x_to_K,envelope):
	def rho(x):
		K=x_to_K(x)
		env=jnp.product(envelope(x),axis=-1)
		return env*jnp.linalg.det(K)
	return rho


"""
construct density function with Gaussian translation-invariant kernel and Gaussian envelope
"""
def GG_DPP_density(var_kernel,var_envelope):

	x_to_K=x_to_Kmatrix_TI(GaussianDensity(var_kernel))
	envelope=GaussianDensity(var_envelope)
	
	rho=DPP_density(x_to_K,envelope)
	return rho



class Sampler(mcmc.Metropolis_batch):
	def __init__(self,rho,n,d,walkers,key):
		key1,key2=jax.random.split(key)
		X=jax.random.normal(key1,shape=(walkers,n,d))
		super().__init__(rho,X)
		self.key=key2

	def sample(self,steps,bsteps):
		self.key,subkey=jax.random.split(self.key)
		return super().sample(subkey,steps,bsteps)


"""
__________________________________________________tests__________________________________________________
"""

"""
def test_det():
	rho=GG_DPP_density(1,1)
	x=jnp.array([[1,2],[3,4],[1,4]])
	X=jnp.repeat(jnp.expand_dims(x,0),10,axis=0)
	print(rho(X))
"""



"""
__________________________________________________plots__________________________________________________
"""	

