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
import thetests as test
import util
	



key=jax.random.PRNGKey(0)
key,*keys=jax.random.split(key,1000)

"""
error,a,p,x,y=util.bestpolyfit(canc.ReLU,3)
plt.figure()
plt.plot(x,y,'r')
plt.plot(x,p,'b')
plt.show()
"""


#W_=[jax.random.normal(keys[n],shape=(10000,n,3))/jnp.sqrt(n*3) for n in range(2,20)]
#thetests.evalWs(W_,util.mindist,'mindist')
#evalWs(W_,lambda x:1/util.Coulomb(x),'Coulomb')






d=3
Bernstein=0.28
_range=jnp.arange(2,11)
ReLUguess=.5*Bernstein/(_range-2)/jnp.sqrt(2)
ReLUguess_=test.polynomialerror(canc.ReLU,_range-2,d*_range)
HS_guess_=test.polynomialerror(lambda x:jnp.heaviside(x,1),_range-2,d*_range)
expguess=1/jnp.array([math.factorial(n-1) for n in _range])
expguess_=test.polynomialerror(jnp.exp,_range-2,d*_range)

#expguess=1/scipy.math.factorial(_range-1)
ReLUguessplot=(_range,ReLUguess,'r')
ReLUguess_plot=(_range,ReLUguess_,'r')
HSguess_plot=(_range,HS_guess_,'g')
expguessplot=(_range,expguess,'r')
expguess_plot=(_range,expguess_,'g')
const_plot=(_range,jnp.ones(len(_range))/jnp.sqrt(2),'r')
_range1=jnp.arange(3,11)
e_plot0=(_range1,jnp.ones(len(_range1))*jnp.exp(1),'r--')
_range0=jnp.arange(2,6)
e_bound=[jnp.sqrt(jnp.average(jnp.square(jnp.exp(util.sample_mu(n*d,10000,keys[n]))))) for n in _range0]
e_plot=(_range0,e_bound,'r')



bookkeep.saveplot(['osc_trivial d=3_n=8','ReLU_trivial d=3_n=8'],'osc_ReLU',['b','b'],[const_plot])
bookkeep.saveplot(['osc_trivial d=3_n=8','ReLU_trivial d=3_n=8'],'osc_ReLU_',['b','b'],connect=True)
bookkeep.saveplot(['osc_trivial d=3_n=8','ReLU_trivial d=3_n=8'],'osc_ReLU_both',['b','b'],[const_plot],connect=True)

bookkeep.saveplot(['ReLU_trivial d=3_n=8','HS_trivial d=3_n=8','exp_trivial d=3_n=8','osc_trivial d=3_n=8'],'all_',['b','g','r','k'],scatter=True,connect=True)
bookkeep.saveplot(['exp_det_trivial d=3_n=10'],'exp_',['b'],[e_plot0,e_plot],connect=True)

bookkeep.saveplot(['ReLU_trivial d=3_n=8','HS_trivial d=3_n=8','osc_trivial d=3_n=8'],'all',['b','b','b'],[ReLUguessplot,ReLUguess_plot,HSguess_plot,const_plot])
bookkeep.saveplot(['ReLU_trivial d=3_n=8'],'ReLUplot',['b'],[ReLUguessplot,ReLUguess_plot])
bookkeep.saveplot(['HS_trivial d=3_n=8'],'HSplot',['b'],[HSguess_plot])
bookkeep.saveplot(['exp_det_trivial d=3_n=10'],'expplot',['b'],[expguessplot,expguess_plot])

bookkeep.saveplot(['2dharmonicprod','2dharmonicdet'],'2d_prod_det',['r','b'])




