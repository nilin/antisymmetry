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
import DPP
import thetests as test
	








activations={'DReLU':util.DReLU,'tanh':jnp.tanh,'osc':util.osc,'exp':jnp.exp,'ReLU':util.ReLU,'HS':util.heaviside}


def getWX(Wname):
	W_,X_,_,_,n_range,_=bk.getdata('trivial')
	if Wname=='separated':
		W_=bk.getdata('W separated')
	return W_,X_,n_range


def gen_magnitudes(Wname,activation):
	W_,X_,n_range=getWX(Wname)
	dev=[]
	delta=[]
	n_range=range(2,9)
	for n in n_range:
		print(n)
		W=W_[n]
		X=X_[n]
		dev.append(util.L2norm(canc.apply_alpha(W,X,activation)))
		print(dev[-1])
		delta.append(util.L2norm(util.mindist(W)))
	return n_range,dev

def gen_dists(Wname):
	W_,X_,n_range=getWX(Wname)
	delta=[]
	n_range=range(2,9)
	for n in n_range:
		W=W_[n]
		X=X_[n]
		delta.append(util.L2norm(util.mindist(W)))
	return n_range,delta
		

for Wname in ['separated','trivial']:
	bk.savedata(gen_dists(Wname),Wname+' delta')

	for ac_name,activation in activations.items():
		
		print('W:'+Wname+', activation:'+ac_name)
		rangedev=gen_magnitudes(Wname,activation)
		print(rangedev)
		bk.savedata(rangedev,Wname+' '+ac_name)
