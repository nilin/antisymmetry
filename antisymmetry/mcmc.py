# author: Nilin 

import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import matplotlib
from sympy.utilities.iterables import multiset_permutations
from matplotlib.gridspec import GridSpec
import pickle
import time
import copy
import jax
import jax.numpy as jnp
import optax
	




apply_mask=jax.vmap(jnp.multiply,in_axes=(0,0))
least_first=lambda x,y:jnpheaviside(y-x,jnp.zeros(x.shape))

def gaussian_move_function(std_dev):
	return (lambda key,x:x+std_dev*jax.random.normal(key,x.shape))

class Metropolis:
	def __init__(self,amplitude,starting_points,quantum=False,proposal_function=gaussian_move_function(.5)):

		if(quantum):
			self.densities=lambda X:jnp.square(jax.vmap(amplitude)(X))
		else:
			self.densities=jax.vmap(amplitude)

		self.acceptance_ratios=lambda X,Y:jnp.divide(self.densities(Y),self.densities(X))

		self.proposal_function=proposal_function
		self.X=starting_points

	def take_step(self,key):

		key1,key2=jax.random.split(key)

		X=self.X
		Y=self.proposal_function(key1,X)

		alphas=self.acceptance_ratios(X,Y)

		list_u=jax.random.uniform(key2,alphas.shape)
		accept_list=least_first(list_u,alphas)
		reject_list=jnp.ones(len(accept_list))-accept_list

		self.X=apply_mask(accept_list,Y)+apply_mask(reject_list,X)

		

	def walk(self,key,steps):
		key,*subkeys=jax.random.split(key,steps+2)
		
		print('\n'+str(len(self.X))+' walkers')
		for i in range(steps):
			self.take_step(subkeys[i])
			print('burn: step '+str(i)+'/'+str(steps),end='\r')
		
	def evaluate_observables_here(self,local_energy_functions):
		local_energies=jax.vmap(local_energy_functions)(self.X)
		return jnp.average(local_energies,axis=0)

	def evaluate_observables(self,local_energy_functions,n_burn,n_steps,key):
		key,*subkeys=jax.random.split(key,n_steps+2)
		self.walk(key,n_burn)
		observable_estimates=[]
		print('')
		for i in range(n_steps):
			self.take_step(subkeys[i])
			estimates=self.evaluate_observables_here(local_energy_functions)
			observable_estimates.append(estimates)
			print('step '+str(i)+'/'+str(n_steps),end='\r')
		print('')
			
		return jnp.average(jnp.array(observable_estimates),axis=0)


