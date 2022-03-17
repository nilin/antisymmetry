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
import DPP
import antisymmetry.mcmc as mcmc
import test_cancellation as test
	



key=jax.random.PRNGKey(0)
key,*subkeys=jax.random.split(key,1000)




def gaussian(means,X):
	d=means.shape[-1]

	means_=jnp.repeat(jnp.expand_dims(means,1),X.shape[0],axis=1)
	X_=jnp.repeat(jnp.expand_dims(X,0),means.shape[0],axis=0)

	s=jnp.sum(jnp.square(X_-means_),axis=-1)

	return jnp.exp(-s/2)/(jnp.sqrt(2*math.pi)**d)



def GMM(centers):
	k=centers.shape[0]
	f=lambda X:jnp.sum(gaussian(centers,X),axis=0)/k
	return f
	

walkers=100
	
centers=10*jax.random.normal(subkeys[0],shape=(3,1))
X=10*jax.random.normal(subkeys[1],shape=(walkers,1))
rho=GMM(centers)

print(centers.shape)
print(X.shape)
print(gaussian(centers,X))

S=mcmc.Metropolis_batch(rho,X)
samples=S.sample(subkeys[2],1000,5000)
samples=[jnp.squeeze(s) for s in samples]
samples=jnp.concatenate(samples)

print(samples[0].shape)

plt.figure()
sns.kdeplot(samples,color='r',bw=.1)
I=jnp.arange(-10,10,.05)
I_=jnp.expand_dims(I,-1)
plt.plot(I,jnp.squeeze(rho(I_)),color='b')
#plt.scatter(dWdX,absY,s=2)

plt.show()
