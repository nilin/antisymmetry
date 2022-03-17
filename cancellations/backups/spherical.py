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



key=jax.random.PRNGKey(0)
key,*subkeys=jax.random.split(key,1000)


arccot=lambda x:jnp.arctan(1/x)


def to_spherical_north_hemisphere(x):
	d=x.shape[-1]
	N=x.shape[:-1]
	s=jnp.concatenate([jnp.zeros(shape=N+(1,)),jnp.square(x)],axis=-1)
	cs=jnp.take(jnp.cumsum(s,axis=-1),jnp.array(range(d-1)),axis=-1)
	rsquare=jnp.sum(jnp.square(x),axis=-1)
	rsquare_=jnp.repeat(jnp.expand_dims(rsquare,axis=-1),d-1,axis=-1)
	revsum=rsquare_-cs
	x_=jnp.take(x,jnp.array(range(d-1)),axis=-1)	
	t=jnp.arccos(x_/jnp.sqrt(revsum))

	return t,jnp.sqrt(rsquare)


def to_spherical(x):
	d=x.shape[-1]
	t,r=to_spherical_north_hemisphere(x)
	x_last=jnp.take(x,-1,axis=-1)
	t_=jnp.take(t,jnp.array(range(d-2)),axis=-1)
	t_last=jnp.take(t,-1,axis=-1)

	northmask=jnp.heaviside(x_last,1)
	southmask=1-northmask
	t_last_new=jnp.multiply(northmask,t_last)+jnp.multiply(southmask,2*math.pi-t_last)
	t_new=jnp.concatenate([t_,jnp.expand_dims(t_last_new,axis=-1)],axis=-1)
	return t_new,r
	

def from_spherical(t,r):
	d=t.shape[-1]+1
	sinprod=jnp.cumprod(jnp.sin(t),axis=-1)
	ones=jnp.expand_dims(jnp.ones(shape=(t.shape[:-1])),axis=-1)
	sinprod=jnp.concatenate([ones,sinprod],axis=-1)
	cosfac=jnp.concatenate([jnp.cos(t),ones],axis=-1)
	x=jnp.multiply(sinprod,cosfac)
	R=jnp.repeat(jnp.expand_dims(r,axis=-1),d,axis=-1)
	return jnp.multiply(R,x)
	

def to_circle_2d(X):
	x=jnp.take(X,0,axis=-1)
	y=jnp.take(X,1,axis=-1)
	angle=jnp.arctan(y/x)
	W_mask=1-jnp.heaviside(x,1)
	SE_mask=jnp.multiply(jnp.heaviside(x,1),1-jnp.heaviside(y,1))
	return angle+math.pi*W_mask+2*math.pi*SE_mask,jnp.sqrt(jnp.square(x)+jnp.square(y))


def circular_harmonics(x,K):
	t,_=to_circle_2d(x)
	t_repeat=jnp.repeat(jnp.expand_dims(t,axis=0),K,axis=0)
	multiples=jnp.cumsum(t_repeat,axis=0)
	
	firstharmonic=jnp.ones((1,)+t.shape)

	cos_sin=jnp.sqrt(2)*jnp.array([jnp.cos(multiples),jnp.sin(multiples)])
	higher_harmonics=jnp.swapaxes(cos_sin,0,1).reshape((2*K,)+t.shape)

	harmonics=jnp.concatenate([firstharmonic,higher_harmonics],axis=0)

	return harmonics,jnp.moveaxis(harmonics,0,-1)







	




