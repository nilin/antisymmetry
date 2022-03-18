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




def gen_magnitudes(Wtype,compute_from_WX,nmax=8):
	Ws,Xs=[bk.getdata(Wtype+'/WX')[k] for k in ('Ws','Xs')]
	norms=[]
	n_range=range(2,nmax+1)
	for n in n_range:
		print(n)
		W=Ws[n]
		X=Xs[n]
		norms.append(util.L2norm(compute_from_WX(W,X)))
	return n_range,norms


def collectmaximaldata():
	for Wtype in ['separated','normal']:
		maxpath='data/'+Wtype+'/maximal'
		bk.mkdir('data/'+Wtype+'/maximal')
		for n in range(20):
			for ac_name,activation in util.activations.items():	
				path='data/'+Wtype+'/'+str(n)+'/'+ac_name
				if os.path.exists(path):
					shutil.copyfile(path,maxpath+'/'+ac_name)




		


Wtype={'s':'separated','n':'normal'}[sys.argv[1]]
nmax=int(sys.argv[2])

bk.mkdir('data/'+Wtype+'/'+str(nmax))

for ac_name,activation in util.activations.items():	
	path=Wtype+'/'+str(nmax)+'/'+ac_name
	print(path)
	if os.path.exists('data/'+path):
		print('exists, skipping\n')
		continue
	compute_from_wx=lambda W,X:canc.apply_alpha(W,X,activation)
	if ac_name=='exp':
		compute_from_wx=proxies.exactexp
	bk.savedata(gen_magnitudes(Wtype,compute_from_wx,nmax),Wtype+'/'+str(nmax)+'/'+ac_name)
	collectmaximaldata()
