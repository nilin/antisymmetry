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
	







key=jax.random.PRNGKey(0)
key,*keys=jax.random.split(key,1000)



#opt.gen_W(keys[0],(instances,n,d))



energies=lambda W:util.Coulomb(W)+jnp.sum(jnp.square(W),axis=(-2,-1))
energy=lambda W:jnp.sum(energies(W))



def dT_test(W,X,apply_tau):

	ijs=util.argmindist(W)
	Ws=util.transpositions(W,ijs)

	Y=apply_tau(W,X)
	Ys=apply_tau(Ws,X)
	dY=Y-Ys

	return jnp.sqrt(jnp.average(jnp.square(dY),axis=-1))
	
def dT_vs_AT(W,X,apply_tau,apply_alpha):
	dT_by_w=dT_test(W,X,apply_tau)
	AT_by_w=jnp.sqrt(jnp.average(jnp.square(apply_alpha(W,X)),axis=-1))
	plt.figure()
	plt.scatter(dT_by_w,AT_by_w)
	plt.show()

def test(apply_tau=canc.apply_tau,nmax=0,WXname='trivial d=3'):

	L2={}
	W_,X_,instances,samples,n_range,d=bookkeep.getdata(WXname)

	if nmax==0:
		nmax=n_range[-1]

	for n in range(2,nmax):

		X=X_[n]
		W=W_[n]

		norms=dT_test(W,X,apply_tau)
		L2[n]=jnp.sqrt(jnp.average(jnp.square(norms)))

		fn='pairwise '+WXname+' n='+str(n)
		bookkeep.savedata(L2,fn)
		bookkeep.saveplot([fn],fn,colors=['r'])

		print(n)


WXname='trivial d=3'
W_,X_,instances,samples,n_range,d=bookkeep.getdata(WXname)
dT_vs_AT(W_[4],X_[4],canc.apply_tau,canc.apply_alpha)
