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
import shutil
import cancellation as canc
import antisymmetry.mcmc as mcmc
import proxies
import permutations



blocksize=720

def compute_term(W,X,activation,p):
	I,n,d=W.shape
	S=X.shape[0]

	P=permutations.perm_as_matrix(p)
	pX=jax.vmap(jnp.dot,in_axes=(None,0))(P,X)

	W_=jnp.reshape(W,(I,n*d))	
	pX_=jnp.reshape(pX,(S,n*d))
	return permutations.sign(p)*activation(jnp.inner(W_,pX_))


def zeros(W,X):
	instances,samples=W.shape[0],X.shape[0]
	return jnp.zeros(shape=(instances,samples))


def partial_sum(W,X,activation,start):

	S=zeros(W,X)
	n=W.shape[-2]
	p0=permutations.k_to_perm(start,n)
	p=p0
	for i in range(blocksize):
		S=S+compute_term(W,X,activation,p)
		p=permutations.nextperm(p)	
		if(i%24==0):
			bk.printbar(i/blocksize,'')
	return S,{'start':p0,'next':p}
		

def register_partial_sums(W,X,activation,prefix,start,stop=-1):
	
	a=start
	cumulative_sum=zeros(W,X)
	blocksums=[]
	timer=bk.Stopwatch()
	n=W.shape[-2]
	N=math.factorial(n)
	if stop==-1:
		stop=N

	prevfilepath='nonexistent'
	while a<stop:


		S,_=partial_sum(W,X,activation,a)
		cumulative_sum=cumulative_sum+S
		blocksums.append(S)
		a=a+blocksize
	
		filepath=prefix+str(start)+' '+str(a)			
		bk.savedata({'result':cumulative_sum,'interval':(start,a),'W':W,'X':X},filepath)
		if os.path.exists('data/'+prevfilepath):
			removepath='data/'+prevfilepath
			os.remove(removepath)
		prevfilepath=filepath
		print('\n\n'+str(a)+' terms. '+str(round(blocksize/timer.tick()))+' terms per second.'); bk.printbar(a/N,''); print('\n\n')

	print('Reached '+str(stop))


		
"""
gen_partial_sum.py ReLU n 10 0
"""

ac_name=sys.argv[1]
Wtype={'s':'separated','n':'normal'}[sys.argv[2]]
n=int(sys.argv[3])
start=int(sys.argv[4])
activation=util.activations[ac_name]
if len(sys.argv)>5:
	stop=int(sys.argv[5])
else:
	stop=math.factorial(n)

dirpath='partialsums/'+Wtype
bk.mkdir('data/'+dirpath)

W,X=[bk.getdata(Wtype+'/WX')[k][n] for k in ('Ws','Xs')]


print('Computing partial sum for '+ac_name+' activation, '+Wtype+' weights, and n='+str(n)+'.')
print('Starting at the (including) k(permutation)='+str(start)+' term.')


register_partial_sums(W,X,activation,dirpath+'/'+ac_name+' n='+str(n)+' range=',start,stop)
