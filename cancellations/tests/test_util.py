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
import util
import jax
import jax.numpy as jnp
import jax.scipy.special as spc
import optax
import cancellation as canc
import antisymmetry.mcmc as mcmc
import spherical
import bookkeep



key=jax.random.PRNGKey(0)

W_,X_,instances,samples,n_range,d=bookkeep.getdata('trivial d=3')

for n in [2]:
	x=util.sample_mu(n*d,100000,key)

	W=W_[n]
	X=X_[n]
	Y=jnp.ravel(canc.apply_tau_(W,X,lambda x:x))

	print(x)
	print(Y)
	
	plt.figure()
	sns.kdeplot(x,bw=.1)
	sns.kdeplot(Y,bw=.1)
	plt.show()
	
	print(jnp.average(x))
	print(jnp.var(x))
	print()



