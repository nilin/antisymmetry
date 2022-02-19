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
import util
import cancellation as canc
import DPP
import test_cancellation as test
	



key=jax.random.PRNGKey(0)
key,*subkeys=jax.random.split(key,1000)

#plot_dsquare()
#plotphi()
#plot_duality(key,250,100,by='W',bw_=.1,figname='by_w.pdf')	
#d_vs_var(key,10000,100)	
#plots()

test.test_DPP(key)

