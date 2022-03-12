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
import util
import cancellation as canc
import antisymmetry.mcmc as mcmc
import DPP
import test_cancellation as test
import opt
	












key=jax.random.PRNGKey(0)
key,*keys=jax.random.split(key,1000)

n=6
d=4
instances=1000
samples=1000


pwr=lambda x,p:jnp.power(x,p*jnp.ones(x.shape))
#opt.gen_W(keys[0],(instances,n,d))



energies=lambda W:util.Coulomb(W)+jnp.sum(jnp.square(W),axis=(-2,-1))
#energies=lambda W:util.Coulomb(W)+jnp.sum(pwr(W,4),axis=(-2,-1))
#energies=lambda W:util.mindist(W)+jnp.sum(pwr(W,4),axis=(-2,-1))
energy=lambda W:jnp.sum(energies(W))



def normalize(W):
	norms=jnp.sqrt(jnp.sum(jnp.square(W),axis=(-2,-1)))
	norms_=jnp.tile(jnp.expand_dims(norms,axis=(1,2)),(n,d))
	return W/norms_
	

def gen_W(key,shape,lossfunction=lambda w:1/util.mindist(w)):

	instances,n,d=shape
	W=normalize(jax.random.normal(key,shape=(1000*instances,n,d)))
	loss=lossfunction(W)	
	_,indices=jax.lax.top_k(-loss,instances)
	return W[indices]
	



	

#test.plotphi()

"""
def gen_W_(key,shape):
	
	instances,n,d=shape
	W=jax.random.normal(key,shape=shape)/jnp.sqrt(n*d)

	rho=lambda w:jnp.exp(-10*energies(w))
	#rho=lambda w:jnp.multiply(jnp.heaviside(1-jnp.sum(jnp.square(w),axis=(-2,-1)),1),jnp.square(util.mindist(w)))
	#rho=lambda w:jnp.multiply(jnp.heaviside(1-jnp.sum(jnp.square(w),axis=(-2,-1)),1),jnp.exp(-util.Coulomb(w)))
	sampler=mcmc.Metropolis_batch(rho,W)
	W_=sampler.sample(keys[1],200,10000)
	W_subsamples=[W_[i] for i in range(0,len(W_),100)]
	W=jnp.concatenate(W_subsamples,axis=0)

	return W




W=jax.random.normal(key,shape=(instances,n,d))/jnp.sqrt(n*d)
X=jax.random.normal(keys[2],shape=(samples,n,d))
test.E_inv_vs_var_(W,X,msg='E_inv')

"""



W=gen_W(keys[0],shape=(instances,n,d))
X=jax.random.normal(keys[2],shape=(samples,n,d))

print(jnp.average(util.mindist(W)))

test.d_vs_var_(W,X,msg='d')
test.E_inv_vs_var_(W,X,msg='d')











"""

W=jax.random.normal(keys[2],shape=(instances,n,d))

f=lambda W:jnp.sum(util.Coulomb(W))

print(jax.grad(f)(W))

"""



"""
rho=DPP.circular_harmonics_DPP_density()
sampler=mcmc.Metropolis_batch(rho,W)
W_=sampler.sample(keys[1],1000,4000)
W=W_[-1]
"""






def normalize_last_dim(x):
	norms=jnp.sqrt(jnp.sum(jnp.square(x),axis=-1))
	norms_=jnp.repeat(jnp.expand_dims(norms,axis=-1),x.shape[-1],axis=-1)
	return x/norms_

def normof_last_dim(x):
	norms=jnp.sqrt(jnp.sum(jnp.square(x),axis=-1))
	return jnp.expand_dims(norms,axis=-1)
	
#test.d_vs_var_(W,X,msg='d')
#test.d_vs_var_(W,X,transform=normalize_last_dim,msg='spherical')
#test.d_vs_var_(W,X,transform=normof_last_dim,msg='radial')
#test.d_vs_var_3d(W,X,normalize_last_dim,normof_last_dim,msg='')

"""
def test():
	W=jax.random.normal(keys[2],shape=(instances,n,2))
	Wn=normalize_last_dim(W)
	x=jnp.take(Wn,0,axis=-1)
	y=jnp.take(Wn,1,axis=-1)
	plt.figure()
	plt.scatter(jnp.ravel(x),jnp.ravel(y),s=2)
	plt.show()	
test()
"""
#test.test_DPP(key)





#plot_dsquare()
#plotphi()
#plot_duality(key,250,100,by='W',bw_=.1,figname='by_w.pdf')	
#d_vs_var(key,10000,100)	
#plots()
