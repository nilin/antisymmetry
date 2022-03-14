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


def plot(activation,proxychoices,markers,nmax):
	WXname='trivial d=3'
	W_,X_,instances,samples,n_range,d=bk.getdata(WXname)

	proxynames={'Z','OP','polyOP','polyZ','OCP','polyOCP','polyOCP_proxy'}
	proxies={name:[] for name in proxynames}

	_range=range(2,nmax+1)
	for n in _range:

		z=jax.random.normal(keys[n],shape=(10000,))
		proxies['Z'].append(util.L2norm(activation(z)))

		x=util.sample_mu(n*d,10000,keys[n])
		proxies['OP'].append(util.L2norm(activation(x)))

		p=util.bestpolyfunctionfit(activation,n-2,x)
		r=lambda x:activation(x)-p(x)
		proxies['polyOP'].append(util.L2norm(r(x)))
		proxies['polyZ'].append(util.L2norm(r(z)))
		
		W=W_[n]
		X=X_[n]
		r_squared=jnp.sum(jnp.square(W),axis=(-2,-1))
		eps_squared=2*jnp.square(util.mindist(W))

		proxies['OCP'].append(jnp.sqrt(jnp.average(util.variations(keys[n],activation,r_squared,eps_squared)/2)))
		proxies['polyOCP'].append(jnp.sqrt(jnp.average(util.variations(keys[n],r,r_squared,eps_squared)/2)))

		r_sq=jnp.atleast_1d(jnp.average(r_squared))
		eps_sq=jnp.atleast_1d(jnp.array(jnp.average(eps_squared)))
		proxies['polyOCP_proxy'].append(jnp.sqrt(util.variations(keys[n],r,r_sq,eps_sq)/2))

	i=0
	for name in proxychoices:	
		plt.plot(_range,proxies[name],markers[i],zorder=50-i)	
		i=i+1

def plots(activations,proxychoices,colors,markers,nmax):
	i=0
	for activation in activations:
		colormarkers=[colors[i]+marker for marker in markers]
		plot(activation,proxychoices,colormarkers,nmax)
		i=i+1


def plotdata(filenames,colors,alpha=.2):
	i=0
	for filename in filenames:
		range_,vals=bk.getplotdata(subfolder+filename)
		plt.scatter(range_,vals,color=colors[i],zorder=100)
		plt.plot(range_,vals,color=colors[i],alpha=alpha)
		i=i+1

def data_only(files,colors,**kwargs):
	activations=[util.activations[f] for f in files]
	plt.figure()
	plt.yscale('log')
	plotdata(files,colors,alpha=1)
	savename=' '.join(files)
	plt.savefig('plots/'+savename+'.pdf')

def data_and_proxy(files,proxies,colors,markers,**kwargs):
	activations=[util.activations[f] for f in files]
	plt.figure()
	plt.yscale('log')
	if 'ylim' in kwargs:
		plt.ylim(kwargs.get('ylim'))
	plots(activations,proxies,colors,markers,8)
	plotdata(files,colors)
	savename=' '.join(files)+' _ '+' '.join(proxies)
	plt.savefig('plots/'+savename+'.pdf')

def data_and_proxy_separate(files,proxies,colormarkers,**kwargs):
	activations=[util.activations[f] for f in files]
	n=len(files)
	if 'ylim' in kwargs:
		ylim=kwargs.get('ylim')
	else:
		ylim=ylimfromfiles(files)
	plt.figure(figsize=(10,10))
	for i in range(n):
		plt.figure()
		plt.yscale('log')
		plt.ylim(ylim)
		plot(activations[i],proxies,[m for m in colormarkers],8)
		plotdata([files[i]],['b'])
		savename=files[i]+' _ '+' '.join(proxies)
		plt.savefig('plots/'+savename+'.pdf')

def ylimfromfiles(files):
	mins=[]
	maxs=[]
	for filename in files:
		_,vals=bk.getplotdata(subfolder+filename)
		mins.append(jnp.min(vals))
		maxs.append(jnp.max(vals))
	padratio=1.1#jnp.power(max(maxs)/min(mins),.2)
	return min(mins),max(maxs)*padratio

#colors=['b','r']
#plots([util.ReLU,jnp.exp],['OP','polyOP','OCP','polyOCP'],colors,[':','-.','--','-'],10)
#plotdata(['ReLU','exp'],colors)

proxies=['polyOCP','OCP','polyZ']
colors=['r','g','b','m']
colormarkers=['r','r:','k:']
markers=['-','--',':']


data_only(['osc','HS','ReLU','exp','tanh','DReLU'],['k','g','b','r','m','c'])
#data_and_proxy([],activations,['polyOCP','polyOCP_proxy'],4*['r'],['',':'])


#data_and_proxy_separate(files,activations,proxies,markers)
data_and_proxy_separate(['tanh'],['polyOCP','polyOCP_proxy'],['r','r:'])
data_and_proxy_separate(['HS','ReLU'],proxies,colormarkers)
data_and_proxy_separate(['HS'],['polyOCP','polyOCP_proxy'],['r','r:'])
data_and_proxy_separate(['ReLU'],['polyOCP','polyOCP_proxy'],['r','r:'])
data_and_proxy_separate(['exp'],['polyOCP','polyOCP_proxy'],['r','r:'])
data_and_proxy_separate(['HS'],proxies,colormarkers)
data_and_proxy_separate(['osc','exp'],proxies,colormarkers)
#data_and_proxy_separate(files,activations,proxies,markers,ylim=(1/100,10))
#data_and_proxy(files,activations,proxies,colors,markers)

