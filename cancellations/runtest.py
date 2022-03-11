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
import bookkeep
import copy
import jax
import jax.numpy as jnp
import optax
import util
import spherical
import cancellation as canc
import antisymmetry.mcmc as mcmc
import DPP
import test_cancellation as test
import thetests
import opt
	







key=jax.random.PRNGKey(0)
key,*keys=jax.random.split(key,1000)





print('test')	

#thetests.genWXs(1000,1000,range(1,20),2,key,'trivial')

#thetests.test_2d_harmonic_slater(key,100)




#genWXs(1000,1000,range(1,15),3,key)
	
apply_abs=lambda W,X:canc.apply_alpha(W,X,jnp.abs)
apply_osc=lambda W,X:canc.apply_alpha(W,X,canc.oscillating)
apply_exp=lambda W,X:canc.apply_alpha(W,X,jnp.exp)
def apply_pwr(p):
	return lambda W,X:canc.apply_alpha(W,X,lambda x:util.pwr(x,p))
apply_halfspace=lambda W,X:canc.apply_alpha(W,X,lambda x:jnp.heaviside(x,1))

def apply_exp_det(W,X):
	dots=jnp.tensordot(W,X,((-1),(-1)))
	exps=jnp.exp(jnp.swapaxes(dots,1,2))
	return jnp.linalg.det(exps)




thetests.test(apply_halfspace,'HS',nmax=8,WXname='trivial d=3')
thetests.test(canc.apply_alpha,'ReLU',nmax=8,WXname='trivial d=3')
thetests.test(apply_abs,'abs',nmax=8,WXname='trivial d=3')
thetests.test(apply_osc,'osc',nmax=8,WXname='trivial d=3')
thetests.test(apply_exp_det,'exp_det',nmax=12,WXname='trivial d=3')
thetests.test(apply_exp,'exp',nmax=8,WXname='trivial d=3')	



#thetests.test(apply_pwr(2),'pwr2',nmax=6,WXname='trivial d=3')
#thetests.test(apply_pwr(3),'pwr3',nmax=6,WXname='trivial d=3')
#thetests.test(apply_pwr(5),'pwr5',nmax=6,WXname='trivial d=3')


#thetests.test(apply_halfspace,'HS',nmax=8,WXname='trivial d=2')
#thetests.test(nmax=8,WXname='trivial d=2')
"""
test(apply_osc,'osc',nmax=8)
test(apply_exp_det,'exp_det',nmax=12)
test(apply_exp,'exp',nmax=8)	
"""

#thetests.test(apply_halfspace,'HS',nmax=8,WXname='WX d=2')
#thetests.test(nmax=8,WXname='WX d=2')
"""
test(apply_osc,'osc',nmax=8,WXname='WX_')
test(apply_exp_det,'exp_det',nmax=12,WXname='WX_')
test(apply_exp,'exp',nmax=8,WXname='WX_')	
"""



