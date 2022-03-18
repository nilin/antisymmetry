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
import multiprocessing as mp
import permutations





def compute_unsigned_term(W,X,activation,p):
	I,n,d=W.shape
	S=X.shape[0]

	P=permutations.perm_as_matrix(p)
	pX=jax.vmap(jnp.dot,in_axes=(None,0))(P,X)

	W_=jnp.reshape(W,(I,n*d))	
	pX_=jnp.reshape(pX,(S,n*d))
	return activation(jnp.inner(W_,pX_))

@jax.jit
def compute_unsigned_term_ReLU(W,X,p):
	return compute_unsigned_term(W,X,util.ReLU,p)

@jax.jit
def compute_unsigned_term_HS(W,X,p):
	return compute_unsigned_term(W,X,util.HS,p)

@jax.jit
def zeros(W,X):
	instances,samples=W.shape[0],X.shape[0]
	return jnp.zeros(shape=(instances,samples))

def partial_sum(W,X,ac_name,start,smallblocksize):
	compute_unsigned_term_activation=globals()['compute_unsigned_term_'+ac_name]
	S=zeros(W,X)
	n=W.shape[-2]
	p0=permutations.k_to_perm(start,n)
	p=p0
	sgn=permutations.sign(p0)
	for i in range(smallblocksize):
		S=S+sgn*compute_unsigned_term_activation(W,X,p)
		p,ds=permutations.nextperm(p)	
		sgn=sgn*ds
		#if(i%24==0):
		#	bk.printbar(i/smallblocksize,'')
	return S




"""
gen_partial_sum.py ReLU n 10 0
"""

ac_name=sys.argv[1]
Wtype={'s':'separated','n':'normal','ss':'separated small','ns':'normal small'}[sys.argv[2]]
n=int(sys.argv[3])
start=int(sys.argv[4])
if len(sys.argv)>5:
	stop=int(sys.argv[5])
else:
	stop=math.factorial(n)

dirpath='partialsums/'+Wtype
bk.mkdir('data/'+dirpath)

W,X=[bk.getdata(Wtype+'/WX')[k][n] for k in ('Ws','Xs')]
#
#
#print('Computing partial sum for '+ac_name+' activation, '+Wtype+' weights, and n='+str(n)+'.')
#print('Starting at the (including) k(permutation)='+str(start)+' term.')





tasks=8
smallblocksize=630
largeblocksize=tasks*smallblocksize



def partialsumfunction(k):
	return partial_sum(W,X,ac_name,k,smallblocksize)

def testfun(x):
	return -x

if __name__=='__main__':

	assert(5040%largeblocksize==0)

	prefix=dirpath+'/'+ac_name+' n='+str(n)+' range='
	a=start
	cumulative_sum=zeros(W,X)
	timer=bk.Stopwatch()
	n=W.shape[-2]
	N=math.factorial(n)
	prevfilepath='nonexistent'

	with mp.Pool(tasks) as pool:
		while a<stop:
			parallelsmallsums=pool.map(partialsumfunction,[a+smallblocksize*t for t in range(tasks)])
			#parallelsmallsums=[partialsumfunction(a+smallblocksize*t) for t in range(tasks)]
			cumulative_sum=cumulative_sum+sum(parallelsmallsums)
			a=a+largeblocksize
		
			filepath=prefix+str(start)+' '+str(a)			
			bk.savedata({'result':cumulative_sum,'interval':(start,a),'W':W,'X':X},filepath)
			if os.path.exists('data/'+prevfilepath):
				removepath='data/'+prevfilepath
				os.remove(removepath)
			prevfilepath=filepath
			bk.printbar(a/N,str(a)+' terms. '+str(round(largeblocksize/timer.tick()))+' terms per second.')

	print('Reached '+str(stop))
