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
import sys
import jax
import jax.numpy as jnp
import optax
import util
import cancellation as canc
import antisymmetry.mcmc as mcmc

from proxies import *


key=jax.random.PRNGKey(0)
key,*keys=jax.random.split(key,1000)


activationnames=['osc','HS','ReLU','exp','tanh','DReLU']
proxynames=['Z','OP','polyOP','polyZ','OCP','polyOCP','polyOCP_proxy','extendedpolyOP']
defaultstyles={'Z':'g--','OP':'k--','polyOP':'k:','polyZ':'g:','OCP':'r--','polyOCP':'r:','polyOCP_proxy':'y:','extendedpolyOP':'m'}




def evalproxies(activation,proxychoices,n_,datafolder):
	Ws,Xs=[bk.getdata(datafolder+'/WX')[k] for k in ['Ws','Xs']]
	norms_table={}
	for proxyname in proxychoices:	
		compnorm=globals()[proxyname+'norm']
		norms_table[proxyname]=[compnorm(keys[n],activation,Ws[n],Xs[n]) for n in n_]
	return norms_table	


"""
color by data, style by estimate
"""
def multiple_activations(ac_name_color,proxy_name_style,WXfolder,datafolder):

	plt.figure()
	plt.yscale('log')
	for ac_name,color in ac_name_color.items():
		n_,norms=bk.getdata(datafolder+'/'+ac_name)
		plt.plot(n_,norms,color=color,marker='o')
	
		norms_table=evalproxies(util.activations[ac_name],proxy_name_style.keys(),n_,WXfolder)	
		[plt.plot(n_,norms_table[p],color=color,ls=ls) for p,ls in proxy_name_style.items()]
		
	savename=' '.join(ac_name_color)
	plt.savefig('plots/'+savename+'.pdf')


"""
each estimate a different color and style
"""
def one_plot_per_activation(ac_names,proxychoices,WXfolder,datafolder,**kwargs):

	for ac_name in ac_names:
		plt.figure()
		plt.yscale('log')
		if 'ylim' in kwargs:
			plt.ylim(kwargs.get('ylim'))

		n_,norms=bk.getdata(datafolder+'/'+ac_name)
		plt.plot(n_,norms,'bo-')

		norms_table=evalproxies(util.activations[ac_name],proxychoices,n_,WXfolder)
		[plt.plot(n_,norms_table[p],defaultstyles[p]) for p in proxychoices]

		savename=ac_name+' _ '+' '.join(proxychoices)
		plt.savefig('plots/singledata/'+savename+'.pdf')


def make_colors(l):
	l=list(l)
	palette=sns.color_palette(None,len(l))
	return {l[i]:palette[i] for i in range(len(l))}	

def cutrange(n_,vals,nmax):
	n_vals=zip(n_,vals)
	n_vals=list(filter(lambda nv:nv[1]<=nmax,n_vals))
	n_,vals=zip(*n_vals)
	return n_,vals


Wtype={'n':'normal','s':'separated'}[sys.argv[1]]
WXfolder=Wtype

nmax=sys.argv[2]
datafolder=WXfolder+'/'+str(nmax)

multiple_activations(make_colors(util.activations.keys()),{'polyOCP':'dotted'},WXfolder,datafolder)
#multiple_activations({'ReLU':'red','HS':'blue'},{'polyOCP':'dotted'},WXfolder,datafolder)


### test all proxies on all functions ###
#one_plot_per_activation(activationnames,proxynames,datafolder)

#### main proxies ###
one_plot_per_activation(util.activations.keys(),['OP','polyOP','OCP','polyOCP'],WXfolder,datafolder)
#
#### polyOCP improvement by direct minimization ###
#one_plot_per_activation(activationnames,['polyOCP','polyOCP_proxy'],['r','k'],'polyOCP/')
#
#
