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
	







#activations={'osc':util.osc,'exp':jnp.exp,'ReLU':util.ReLU,'HS':util.heaviside}
activations=util.activations

colors={'trivial':'b','separated':'r'}

for ac_name,activation in activations.items():
	plt.figure()
	plt.yscale('log')

	for WXname in ['trivial','separated']:
		range1,var=bk.getplotdata(WXname+' '+ac_name)
		range2,delta=bk.getplotdata(WXname+' delta')
		plt.plot(range1,var,colors[WXname]+'o-')
		plt.plot(range2,delta,colors[WXname]+':')
		plt.savefig('plots/trivial_vs_separated '+ac_name+'.pdf')

		
plt.figure()
plt.yscale('log')
for WXname in ['trivial','separated']:
	range2,delta=bk.getplotdata(WXname+' delta')
	plt.plot(range2,delta,colors[WXname])
plt.savefig('plots/trivial_vs_separated delta.pdf')
