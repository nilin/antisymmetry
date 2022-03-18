import numpy as np
import sys
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
import cancellation as canc
import antisymmetry.mcmc as mcmc
import bookkeep as bk





def genXs(key,n_,d,samples):
	_,*keys=jax.random.split(key,100)
	return {int(n):jax.random.normal(keys[n],shape=(samples,n,d)) for n in n_}

def genWs(Wtype,key,n_,d,instances):
	_,*keys=jax.random.split(key,100)
	return {int(n):globals()['gen_W_'+Wtype](keys[n],shape=(instances,n,d)) for n in n_}

def gen_W_normal(key,shape):
	(_,n,d)=shape
	return jax.random.normal(key,shape)/jnp.sqrt(n*d)

def gen_W_separated(key,shape):
	return util.normalize(separate(gen_W_normal(key,shape)))

def separate(W,iterations=100,optimizer=optax.rmsprop(.01),smoothingperiod=25):
	(instances,n,d)=W.shape
	print('\n'+str(n))
	state=optimizer.init(W)

	losses=[]
	for i in range(iterations):
		loss,grads=jax.value_and_grad(energy)(W)
		updates,_=optimizer.update(grads,state,W)
		W=optax.apply_updates(W,updates)
		losses.append(loss)
		bk.printbar(loss/losses[0],'')
	return W

def softCoulomb(X):
	eps=.001
	dists=jnp.sqrt(util.pairwisesquaredists(X)+eps)
	energies=jnp.triu(1/dists,k=1)
	return jnp.sum(energies,axis=(-2,-1))

energy=lambda W:jnp.average(softCoulomb(W))+jnp.sum(jnp.square(W))



key=jax.random.PRNGKey(0)
key1,key2=jax.random.split(key)
Wtype={'n':'normal','s':'separated'}[sys.argv[1]]
nmax=int(sys.argv[2])
n_=range(2,nmax+1)
d=3

def gen(instances,samples,suffix=''):
	Ws=genWs(Wtype,key1,n_,d,instances)
	deltas={int(n):util.L2norm(util.mindist(Ws[n])) for n in n_}
	Xs=genXs(key2,n_,d,samples)
	bk.savedata({'Ws':Ws,'Xs':Xs,'Wtype':Wtype,'instances':instances,'samples':samples,'d':d,'n_':n_,'deltas':deltas},Wtype+'/WX'+suffix)

if len(sys.argv)>3 and sys.argv[3]=='small':
	print('Generating small')
	gen(100,100,' small')
else:
	gen(1000,1000)
