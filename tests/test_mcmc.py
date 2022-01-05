import antisymmetry.learning as learning
import antisymmetry.mcmc as mcmc
import antisymmetry.plotdata as plotdata

import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import sys
import os


def test_mcmc():

	randkey0=jax.random.PRNGKey(np.random.randint(100))

	key,*subkeys=jax.random.split(randkey0,10)

	n_walkers=10000
	n_burn=1000
	n_steps=2000
	n=3
	d=2

	position=lambda x:x
	squared_x=lambda x:jnp.square(x)

	start_positions=jax.random.uniform(subkeys[1],shape=(n_walkers,n,d))

	p=3*jax.random.normal(subkeys[2],shape=(n,d))
	density=lambda x:jnp.exp(jnp.sum(-(1/2)*jnp.square(x-p)))
	walkers=mcmc.Metropolis(density,start_positions)

	
	means_output=walkers.evaluate_observables(position,n_burn,n_steps,subkeys[3])
	np.testing.assert_allclose(p,means_output,rtol=1/100)

	print('\ntest means')
	print('expected:\n'+str(p))
	print('output:\n'+str(means_output))


	squares_true=jnp.square(p)+jnp.ones(p.shape)
	squares_output=walkers.evaluate_observables(squared_x,n_burn,n_steps,subkeys[3])
	np.testing.assert_allclose(squares_true,squares_output,rtol=1/100)

	print('\ntest squares (bias^2+variance)')
	print('expected:\n'+str(squares_true))
	print('output:\n'+str(squares_output))


test_mcmc()
