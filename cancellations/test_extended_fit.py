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




key=jax.random.PRNGKey(0)
key,*keys=jax.random.split(key,1000)


activation=jnp.tanh

x=jnp.sort(jax.random.normal(key,shape=(10000,)))

#functionbasis=lambda x:jnp.array([jnp.ones(x.shape),x,jnp.exp(x),jnp.exp(-x)])
#functionbasis=util.prepfunctions([util.monomials(0)],[jnp.exp,lambda x:jnp.exp(-x)])
functionbasis1=util.prepfunctions([],[lambda x:jnp.exp(-2*x),lambda x:jnp.exp(-x),lambda x:jnp.exp(x),lambda x:jnp.exp(2*x)])
functionbasis2=util.prepfunctions([lambda x:jnp.exp(jnp.array([-2,-1,1,2])*x)],[])
print(functionbasis1(1))
print(functionbasis2(1))

functionbasis=functionbasis2
a_=util.functionfit(x,activation(x),functionbasis)
p_=util.as_function(a_,functionbasis)
r_=lambda x:activation(x)-p_(x)



print(a_)

plt.figure()
plt.plot(x,activation(x),'b')
plt.plot(x,p_(x),'r')
plt.show()
