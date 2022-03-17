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
import spherical
import cancellation as canc
import antisymmetry.mcmc as mcmc




key=jax.random.PRNGKey(0)
key,*keys=jax.random.split(key,1000)

subfolder='abbrev/'

activationnames=['osc','HS','ReLU','exp','tanh','DReLU']
proxynames=['Z','OP','polyOP','polyZ','OCP','polyOCP','polyOCP_proxy','extendedpolyOP']
defaultstyles={'Z':'g--','OP':'k--','polyOP':'k','polyZ':'g','OCP':'r--','polyOCP':'r','polyOCP_proxy':'y:','extendedpolyOP':'m'}



####################################################################################################
def Znorm(key,activation,W,X):
	z=jax.random.normal(key,shape=(10000,))
	return util.L2norm(activation(z))

def OPnorm(key,activation,W,X):
	n,d=W.shape[-2:]
	x=util.sample_mu(n*d,10000,key)
	return util.L2norm(activation(x))

def polyOPnorm(key,activation,W,X):
	n,d=W.shape[-2:]
	x=util.sample_mu(n*d,10000,key)
	a,dist=util.polyfit(x,activation(x),n-2)
	#p=util.poly_as_function(a)
	#r=lambda x:activation(x)-p(x)
	return dist

def polyZnorm(key,activation,W,X):
	n=W.shape[-2]
	z=jax.random.normal(key,shape=(10000,))
	a,dist=util.polyfit(z,activation(z),n-2)
	#p=util.poly_as_function(a)
	#r=lambda x:activation(x)-p(x)
	return dist

def OCPnorm(key,activation,W,X):
	r_squared=jnp.sum(jnp.square(W),axis=(-2,-1))
	eps_squared=2*jnp.square(util.mindist(W))
	return jnp.sqrt(jnp.average(util.variations(key,activation,r_squared,eps_squared)))

def polyOCPnorm(key,activation,W,X):
	n=W.shape[-2]
	r_squared=jnp.sum(jnp.square(W),axis=(-2,-1))
	eps_squared=2*jnp.square(util.mindist(W))
	key1,key2=jax.random.split(key)
	a,dist=util.poly_fit_variations(key1,activation,n-2,r_squared,eps_squared)
	return util.L2norm(dist)
		
def polyOCP_proxynorm(key,activation,W,X):
	n,d=W.shape[-2:]
	key1,key2=jax.random.split(key)
	r_squared=jnp.sum(jnp.square(W),axis=(-2,-1))
	eps_squared=2*jnp.square(util.mindist(W))

	x=util.sample_mu(n*d,10000,key1)
	a,dist=util.polyfit(x,activation(x),n-2)
	p=util.poly_as_function(a)
	r=lambda x:activation(x)-p(x)

	return jnp.sqrt(jnp.average(util.variations(key2,r,r_squared,eps_squared)))

def extendedpolyOPnorm(key,activation,W,X):
	n,d=W.shape[-2:]
	x=util.sample_mu(n*d,10000,key)
	functionbasis=util.prepfunctions([util.monomials(n-2)],[lambda x:jnp.exp(-2*x),lambda x:jnp.exp(-x),jnp.exp,lambda x:jnp.exp(2*x)])
	a,dist=util.functionfit(x,activation(x),functionbasis)
	return dist

####################################################################################################

def plotproxies(activation,proxychoices,nmax,styles='default'):
	WXname='trivial d=3'
	W_,X_,instances,samples,n_range,d=bk.getdata(WXname)

	if styles=='default':
		styles=[defaultstyles[c] for c in proxychoices]

	n_=range(2,nmax+1)

	i=0
	for proxyname in proxychoices:	
		compnorm=globals()[proxyname+'norm']
		norms=[compnorm(keys[n],activation,W_[n],X_[n]) for n in n_]
		plt.plot(n_,norms,styles[i],zorder=50-i)	
		i=i+1

def plotdata(filenames,colors,alpha=.2,subfolder=subfolder):
	i=0
	for filename in filenames:
		range_,vals=bk.getplotdata(subfolder+filename)
		plt.scatter(range_,vals,color=colors[i],zorder=100)
		plt.plot(range_,vals,color=colors[i],alpha=alpha)
		i=i+1

def data_only(files,colors,subfolder):
	activations=[util.activations[f] for f in files]
	plt.figure()
	plt.yscale('log')
	plotdata(files,colors,alpha=1,subfolder=subfolder)
	plt.plot(range(1,10),1/jnp.sqrt(jnp.array([math.factorial(n) for n in range(1,10)])),'r:')
	savename=' '.join(files)
	plt.savefig('plots/'+subfolder+savename+'.pdf')

def data_and_proxy(files,proxies,colors,markers,**kwargs):
	def plots(activations,proxychoices,colors,markers,nmax):
		i=0
		for activation in activations:
			colormarkers=[colors[i]+marker for marker in markers]
			plotproxies(activation,proxychoices,nmax,colormarkers)
			i=i+1
	activations=[util.activations[f] for f in files]
	plt.figure()
	plt.yscale('log')
	if 'ylim' in kwargs:
		plt.ylim(kwargs.get('ylim'))
	plots(activations,proxies,colors,markers,8)
	plotdata(files,colors)
	savename=' '.join(files)+' _ '+' '.join(proxies)
	plt.savefig('plots/'+savename+'.pdf')

def data_and_proxy_separate(files,proxies,styles='default',subfolder='',same_ylim=False,**kwargs):
	activations=[util.activations[f] for f in files]
	n=len(files)
	if same_ylim:
		ylim=ylimfromfiles(files)
	plt.figure(figsize=(10,10))
	for i in range(n):
		plt.figure()
		plt.yscale('log')
		if same_ylim:
			plt.ylim(ylim)
		plotproxies(activations[i],proxies,8,styles=styles)
		plotdata([files[i]],['b'])
		savename=files[i]+' _ '+' '.join(proxies)
		plt.savefig('plots/'+subfolder+savename+'.pdf')

def ylimfromfiles(files):
	mins=[]
	maxs=[]
	for filename in files:
		_,vals=bk.getplotdata(subfolder+filename)
		mins.append(jnp.min(vals))
		maxs.append(jnp.max(vals))
	padratio=1.1#jnp.power(max(maxs)/min(mins),.2)
	return min(mins),max(maxs)*padratio



data_only(activationnames,['k','g','b','r','m','c'],'abbrev_separated/')
data_only(activationnames,['k','g','b','r','m','c'],'abbrev/')


### test all proxies on all functions ###
data_and_proxy_separate(activationnames,proxynames,subfolder='all/')

### main proxies ###
data_and_proxy_separate(activationnames,['OP','polyOP','OCP','polyOCP'],subfolder='mainproxies/')

### polyOCP improvement by direct minimization ###
data_and_proxy_separate(activationnames,['polyOCP','polyOCP_proxy'],['r','k'],'polyOCP/')


