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




def combine(ac_name,Wtype,n):

	activation=util.activations[ac_name]
	dirpath='partialsums/'+Wtype
	prefix=dirpath+'/'+ac_name+' n='+str(n)+' range='

	a=0
	b=0
	N=math.factorial(n)

	checkpoints=[]
	blocksums=[]
	while b<N:
		b=b+720

		filepath=prefix+str(a)+' '+str(b)
		if os.path.exists('data/'+filepath):
			print(filepath)
			data=bk.getdata(filepath)
			S=data['result']
			blocksums.append(S)
			checkpoints.append(b)
			a=b	

	assert(checkpoints[-1]==N)
	return sum(blocksums)/jnp.sqrt(N)



def test_combine(n):
	
	norm_=util.L2norm(combine('ReLU','normal',n))
	n_,norms=bk.getdata('normal/'+str(n)+'/ReLU')

	print(100*'-')	
	print(norm_)
	print(norms)
	assert(jnp.abs(jnp.log(norm_/norms[-1]))<.001)
	print(100*'-')	



"""
combine_partial_sums.py ReLU n 10
"""
test_combine(7)


ac_name=sys.argv[1]
Wtype={'s':'separated','n':'normal'}[sys.argv[2]]
n=int(sys.argv[3])

print(jnp.average(combine(ac_name,Wtype,n)))
