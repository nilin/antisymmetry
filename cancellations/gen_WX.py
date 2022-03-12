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
import bookkeep
import copy
import jax
import jax.numpy as jnp
import optax
import util
import spherical
import cancellation as canc
import antisymmetry.mcmc as mcmc
import DPP
import opt
import thetests as test	



def gen_W(key,shape,lossfunction=lambda w:1/util.mindist(w)):

	instances,n,d=shape
	W=test.normalize(jax.random.normal(key,shape=(1000*instances,n,d)))
	loss=lossfunction(W)	
	_,indices=jax.lax.top_k(-loss,instances)
	return W[indices]

	

def gen_WXs(instances,samples,n_range,d,key,savename='separated'):
	key,*subkeys=jax.random.split(key,1000)

	W_={}
	for n in n_range:
		print(n)
		if savename=='separated':
			W_[int(n)]=gen_W(subkeys[2*n],shape=(instances,n,d),lossfunction=util.Coulomb)
		else:
			W_[int(n)]=jax.random.normal(subkeys[2*n],shape=(instances,n,d))/jnp.sqrt(n*d)
	X_={int(n):jax.random.normal(subkeys[2*n+1],shape=(samples,n,d)) for n in n_range}
	bookkeep.savedata((W_,X_,instances,samples,n_range,d),savename+' d='+str(d))


key=jax.random.PRNGKey(0)
gen_WXs(1000,1000,jnp.arange(2,9),3,key,'separated')
#genWXs(1000,1000,jnp.arange(2,9),3,key,'trivial')
gen_WXs(1,10000,jnp.arange(2,9),3,key,'trivial 1 10000')


