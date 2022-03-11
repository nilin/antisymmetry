import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import matplotlib
from sympy.utilities.iterables import multiset_permutations
from matplotlib.gridspec import GridSpec
import seaborn as sns
import scipy
import pickle
import time
import copy
import bookkeep
import jax
import jax.numpy as jnp
import optax
import cancellation as canc
import DPP
import thetests
import util
	



key=jax.random.PRNGKey(0)
key,*keys=jax.random.split(key,1000)

W_=[jax.random.normal(keys[n],shape=(10000,n,3))/jnp.sqrt(n*3) for n in range(2,20)]
thetests.evalWs(W_,util.mindist,'mindist')
#evalWs(W_,lambda x:1/util.Coulomb(x),'Coulomb')





datanames1=['alpha7','osc7','exp_det11','exp7','HSWX7']
datanames2=['alphaWX_7','oscWX_7','exp_detWX_11','expWX_7','HSWX_7']
_colors=['r','g','b','c']
#_colors=_colors+_colors
#datanames=datanames1+datanames2
datanames=['ReLU_trivial d=3_n=7','exp_det_trivial d=3_n=10','exp_trivial d=3_n=7','osc_trivial d=3_n=7']



Bernstein=0.28
_range=jnp.arange(2,11)
ReLUguess=.5*Bernstein/(_range-2)
expguess=1/jnp.array([math.factorial(n-1) for n in _range])
#expguess=1/scipy.math.factorial(_range-1)
ReLUguessplot=(_range,jnp.square(ReLUguess),'k')
expguessplot=(_range,jnp.square(expguess),'k')

bookkeep.saveplot(datanames,'all',_colors,[ReLUguessplot,expguessplot])

bookkeep.saveplot(['2dharmonicprod','2dharmonicdet'],'2d_prod_det',['r','b'])





