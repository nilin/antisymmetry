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



key=jax.random.PRNGKey(0)


for n in range(10):
	x=util.sample_mu(n,100000,key)

	print(jnp.average(x))
	print(jnp.var(x))
	print()



