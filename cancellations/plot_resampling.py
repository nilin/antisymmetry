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
import copy
import jax
import jax.numpy as jnp
import optax
import bookkeep
import cancellation as canc
	


key=jax.random.PRNGKey(0)
key,*keys=jax.random.split(key,1000)

#############################################################

ReLU=lambda x:(jnp.abs(x)+x)/2
osc=lambda x:jnp.sin(100*x)





def sample_symmetrized_sum(l,terms,n_samples,key):
	
	key,*subkeys=jax.random.split(key,2*n_samples+2)
	samples=[]

	for i in range(n_samples):
		vals=jax.random.choice(subkeys[2*i],l,shape=(terms,))
		signs=jax.random.rademacher(subkeys[2*i+1],shape=(terms,))
		sample=jnp.dot(vals,signs)/jnp.sqrt(terms)
		samples.append(sample)
		
	return jnp.array(samples)



def plot_resample(activation,name='',nmax=7):
	
	W_,X_,instances,samples,n_range,d=bookkeep.getdata('trivial d=3')

	for n in n_range:
		if n<=nmax:
			print(W_)
			W=W_[int(n)]
			X=X_[int(n)]
			nonsym=canc.apply_tau_(W,X,activation)
			antisym=canc.apply_alpha(W,X,activation)
			resamples=sample_symmetrized_sum(nonsym[0],math.factorial(n),1000,keys[n])

			plt.figure()
			plt.xlim((-3,3))
			sns.kdeplot(antisym[0],bw=.1,color='b')
			sns.kdeplot(resamples,bw=.1,color='r')
			plt.savefig('plots/resampling/'+name+' '+str(n)+'.pdf')


plot_resample(osc,'osc')
plot_resample(ReLU,'ReLU')
