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
import cancellation as canc
import antisymmetry.mcmc as mcmc




def gen_magnitudes(Wtype,activation,nmax=8):
	Ws,Xs=[bk.getdata(Wtype+'/WX')[k] for k in ('Ws','Xs')]
	norms=[]
	n_range=range(2,nmax+1)
	for n in n_range:
		print(n)
		W=Ws[n]
		X=Xs[n]
		norms.append(util.L2norm(canc.apply_alpha(W,X,activation)))
	return n_range,norms
		


Wtype={'s':'separated','n':'normal'}[sys.argv[1]]
print(Wtype)
nmax=int(sys.argv[2])

bk.mkdir('data/'+Wtype+'/'+str(nmax))

for ac_name,activation in util.activations.items():	
	print('activation:'+ac_name)
	bk.savedata(gen_magnitudes(Wtype,activation,nmax),Wtype+'/'+str(nmax)+'/'+ac_name)
