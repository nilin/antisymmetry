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
import bookkeep as bk
import copy
import jax
import jax.numpy as jnp
import optax
import util
import sys
import os
import cancellation as canc
import antisymmetry.mcmc as mcmc



l_=jnp.arange(-2.9,3,.1)
mid=len(l_)//2

def overlaps(W,X,activations):
	k=len(activations)
	instances=W.shape[0];samples=X.shape[0]
	Ys=jnp.array([canc.apply_alpha(W,X,activations[i]) for i in range(k)])
	return jnp.tensordot(Ys,Ys,axes=((-2,-1),(-2,-1)))/(instances*samples)

def scale(M):
	D=jnp.diag(jnp.sqrt(1/jnp.diag(M)))
	return jnp.linalg.multi_dot([D,M,D])
	

def plot_overlaps(W,X,activations):
	one_position=np.where(round(l_*1000)==1000)[0][0]

	M=scale(overlaps(W,X,activations))
	plt.matshow(M)
	plt.show()
	plt.figure()
	plt.plot(l_[jnp.arange(mid)],jnp.take(M,one_position,axis=-1)[jnp.arange(mid)],'b')
	plt.plot(l_[jnp.arange(mid+1,len(l_))],jnp.take(M,one_position,axis=-1)[jnp.arange(mid+1,len(l_))],'b')
	plt.plot(l_,jnp.exp(-(1/2)*jnp.square(l_-1)),'r')
	plt.show()
		

def exponential(l):
	f=lambda x:jnp.exp(l*x)
	return f

activations=[exponential(l) for l in l_]

Wtype={'s':'separated','n':'normal'}[sys.argv[1]]
n=int(sys.argv[2])

Ws,Xs=[bk.getdata(Wtype+'/WX')[k] for k in ('Ws','Xs')]
plot_overlaps(Ws[n],Xs[n],activations)

