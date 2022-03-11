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
key,*subkeys=jax.random.split(key,1000)




def test_spherical_high_dim():
	x=jax.random.normal(key,shape=(1000,10))
	t,r=spherical.to_spherical(x)
	y=spherical.from_spherical(t,r)
	util.compare(x,y)

def test_circular():
	X=jax.random.normal(key,shape=(10000,2))
	t,r=spherical.to_circle_2d(X)
	x=jnp.multiply(r,jnp.cos(t))
	y=jnp.multiply(r,jnp.sin(t))
	util.compare(X,jnp.array([x,y]).T)
	print('approx range')
	print(jnp.min(t))
	print(jnp.max(t))
	print()
	

def test_2d_harmonics():
	n=1000000
	x=jax.random.normal(key,shape=(n,2))
	h=spherical.circular_harmonics(x,3)

	gram=jnp.dot(h,h.T)/n
	print(round(gram*100)/100)
	


test_spherical_high_dim()
test_circular()
test_2d_harmonics()


