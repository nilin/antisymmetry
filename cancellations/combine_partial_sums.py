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



		

def register_partial_sums(W,X,activation,prefix,start,stop=-1):
	
	a=start
	cumulative_sum=zeros(W,X)
	blocksums=[]
	t=bk.Stopwatch()
	n=W.shape[-2]
	if stop==-1:
		stop=math.factorial(n)

	while a<stop:
		bk.printbar((a-start)/(stop-start),str(round(blocksize/t.tick()))+' terms per second.')

		S,_=partial_sum(W,X,activation,a)
		cumulative_sum=cumulative_sum+S
		blocksums.append(S)
		a=a+blocksize
	
		if a%5040==0:
			print('\n'+50*'-'+' saved interval ['+str(start)+','+str(a)+') '+50*'-')
			bk.savedata({'result':cumulative_sum,'interval':(start,a),'blocksums':blocksums,'blocksize':blocksize,'W':W,'X':X},prefix+str(start)+' '+str(a))

	print('Reached '+str(stop))


		
"""
combine_partial_sums.py ReLU n 10 0
"""

ac_name=sys.argv[1]
Wtype={'s':'separated','n':'normal'}[sys.argv[2]]
n=int(sys.argv[3])
activation=util.activations[ac_name]

dirpath='partialsums/'+Wtype
prefix=dirpath+'/'+ac_name+' n='+str(n)+' range='

a=0
b=0
N=math.factorial(n)

while b<N:
	filename=prefix+str(a)+' '+str(b)
	
	b=b+120




