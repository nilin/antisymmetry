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



def normalize(W):
	norms=jnp.sqrt(jnp.sum(jnp.square(W),axis=(-2,-1)))
	norms_=jnp.tile(jnp.expand_dims(norms,axis=(1,2)),W.shape[-2:])
	return W/norms_
	


def evalWs(W_,f=util.Coulomb,name=''):
	plt.figure()
	plt.yscale('log')
	avg=[]
	mx=[]
	_range=[]
	for i in range(len(W_)):
		MD=f(W_[i])				
		avg.append(jnp.average(MD))
		mx.append(jnp.max(MD))
		_range.append(W_[i].shape[-2])

	print(jnp.array(mx)/jnp.array(avg))
	
	d=W_[0].shape[-1]
	plt.plot(_range,avg)
	plt.plot(_range,mx)
#	plt.plot(_range,2*util.pwr(jnp.array(_range),-1/(d-1)))
	plt.savefig('plots/evalW'+name+'.pdf')
	plt.show()
		




def test(apply_alpha=canc.apply_alpha,filename='alpha',nmax=0,WXname='WX'):

	L2={}
	W_,X_,instances,samples,n_range,d=bookkeep.getdata(WXname)

	L2_nonsymmetrized=[jnp.sqrt(jnp.average(jnp.square(canc.apply_tau(W_[n],X_[n])))) for n in n_range]
	bookkeep.savedata(L2_nonsymmetrized,'nonsym')
		
	if nmax==0:
		nmax=n_range[-1]

	for n in range(2,nmax):

		W=W_[n]
		X=X_[n]

		Y=apply_alpha(W,X)
		L2[n]=jnp.sqrt(jnp.average(jnp.square(Y)))

		fn=filename+'_'+WXname+'_n='+str(n)
		bookkeep.savedata(L2,fn)
		bookkeep.saveplot([fn],fn,colors=['r'])

		print(n)


def test_2d_harmonic_slater(key,samples):
	key,*subkeys=jax.random.split(key,1000)

	n_range=range(1,12)
	X_={n:jax.random.normal(subkeys[2*n+1],shape=(samples,n,2)) for n in n_range}
	L2={}

	h={n:harmonics_2d(X_[n]) for n in n_range}

	h_to_prod=lambda h: jnp.product(jnp.diagonal(h,axis1=-2,axis2=-1),axis=-1)
	nonsym={n:jnp.sqrt(jnp.average(jnp.square(h_to_prod(h[n])))) for n in n_range}
	bookkeep.savedata(nonsym,'2dharmonicprod')

	_det={n:jnp.sqrt(jnp.average(jnp.square(jnp.linalg.det(h[n])/jnp.sqrt(math.factorial(n))))) for n in n_range}
	bookkeep.savedata(_det,'2dharmonicdet')


def harmonics_2d(X):
	samples=X.shape[0]
	n=X.shape[-2]
	K=n//2
	X_=jnp.reshape(X,(samples*n,2))
	_,h=spherical.circular_harmonics(X_,K)
	h=jnp.take(h,jnp.array(range(n)),axis=-1)
	h=jnp.reshape(h,(samples,n,n))
	return h
	

def polynomialerror(f,deg,n_mu,fname=''):
	errors=[]
	for i in range(len(deg)):
		x=util.sample_mu(n_mu[i],10000,key)
		err=util.bestpolyfit(f,deg[i],x)[0]
		errors.append(err)
	return jnp.array(errors)
