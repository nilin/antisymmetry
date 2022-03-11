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
import spherical



key=jax.random.PRNGKey(0)
key,*subkeys=jax.random.split(key,1000)






"""
construct density function
"""	
def DPP_density(x_to_K,envelope,from_orbitals=False):
	def rho(x):
		K=x_to_K(x)
		env=jnp.product(envelope(x),axis=-1)
		detfactor=jnp.linalg.det(K)
		if(from_orbitals):
			detfactor=jnp.square(detfactor)
		return env*detfactor
	return rho



def x_to_K_2d(x):
	n=x.shape[-2]
	K=(n-1)//2
	_,h=spherical.circular_harmonics(x,K)
	return jnp.take(h,jnp.array(range(n)),axis=-1)


def circular_harmonics_DPP_density():
	envelope=lambda x:jnp.sum(jnp.exp(-jnp.square(x)/2),axis=-1)
	return DPP_density(x_to_K_2d,envelope,from_orbitals=True)
	



"""
__________________________________________________displacement kernels__________________________________________________
__________________________________________________displacement kernels__________________________________________________
__________________________________________________displacement kernels__________________________________________________
__________________________________________________displacement kernels__________________________________________________
__________________________________________________displacement kernels__________________________________________________
__________________________________________________displacement kernels__________________________________________________
"""

def GaussianDensity(dev):
	C=1/(dev*jnp.sqrt(2*math.pi))
	rho=lambda x:C**(x.shape[-1])*jnp.exp(-jnp.sum(jnp.square(x),axis=-1)/(2*(dev**2)))
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
construct density function with Gaussian translation-invariant kernel and Gaussian envelope
"""
def Gaussian_kernel_DPP_density(dev_kernel):

	x_to_K=x_to_Kmatrix_TI(GaussianDensity(dev_kernel))
	one=lambda x:jnp.ones(x.shape[:-2])
	
	rho=DPP_density(x_to_K,one)
	return rho

"""
__________________________________________________displacement kernels over__________________________________________________
__________________________________________________displacement kernels over__________________________________________________
__________________________________________________displacement kernels over__________________________________________________
__________________________________________________displacement kernels over__________________________________________________
__________________________________________________displacement kernels over__________________________________________________
__________________________________________________displacement kernels over__________________________________________________

"""











def move_on_sphere(r,eps):
	def sampler(key,x):
		std_dev=eps/jnp.sqrt(x.shape[-2]*x.shape[-1])
		return r*canc.normalize_rows(x+std_dev*jax.random.normal(key,x.shape)) 
	return sampler
	

class Sampler(mcmc.Metropolis_batch):
	def __init__(self,rho,n,d,walkers,key,r=1):
		key1,key2=jax.random.split(key)
		X=canc.spherical(n,d,radius=r)(key1,walkers)
	
		super().__init__(rho,X,move_on_sphere(r,.1*r))
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

