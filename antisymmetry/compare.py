import antisymmetry.learning as learning
import antisymmetry.mcmc as mcmc
import antisymmetry.plotdata as plotdata
import math
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
import copy
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
import jax.numpy as jnp
import jax
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import itertools
import sys
			
	
#filename="most_recent"




def single_particle_moments(K):
	def momentsfunction(X):
		x=jnp.concatenate([jnp.array([1]),X[0]],axis=0)
		tensor=jnp.array([1])
		for i in range(K):
			tensor=jnp.kron(tensor,x)
		return tensor
	return momentsfunction




def compare(truth,ansatz,params,observables,rtol=1/10):

	n=params['n']
	d=params['d']
	randkey=jax.random.PRNGKey(0); key,*subkeys=jax.random.split(randkey,10)

	n_walkers=1000
	n_burn=250
	n_steps=250

	start_positions=jax.random.uniform(subkeys[1],shape=(n_walkers,n,d))

	amplitude_truth=truth.evaluate	
	amplitude_ansatz=ansatz.evaluate	

	walkers_truth=mcmc.Metropolis(amplitude_truth,start_positions,quantum=True)
	walkers_ansatz=mcmc.Metropolis(amplitude_ansatz,start_positions,quantum=True)
	
	observables_truth=walkers_truth.evaluate_observables(observables,n_burn,n_steps,subkeys[3])
	observables_ansatz=walkers_ansatz.evaluate_observables(observables,n_burn,n_steps,subkeys[3])
	#np.testing.assert_allclose(observables_truth,observables_ansatz,rtol=1/100)

	#np testing: if ansatz gives observable within 10% of truth function
	#print('observables true function:\n'+str(observables_truth))
	#print('observables Ansatz:\n'+str(observables_ansatz))
	np.testing.assert_allclose(observables_truth,observables_ansatz,rtol=rtol) 

def get_max(truth,ansatz,params,observables):
	#source: antisymmetry/compare.py/compare()
	#modification: returns the maximum-relative-error by the given parameters through a mcmc.

	n=params['n']
	d=params['d']
	randkey=jax.random.PRNGKey(0); key,*subkeys=jax.random.split(randkey,10)

	n_walkers=1000
	n_burn=250
	n_steps=250

	start_positions=jax.random.uniform(subkeys[1],shape=(n_walkers,n,d))

	amplitude_truth=truth.evaluate	
	amplitude_ansatz=ansatz.evaluate	

	walkers_truth=mcmc.Metropolis(amplitude_truth,start_positions,quantum=True)
	walkers_ansatz=mcmc.Metropolis(amplitude_ansatz,start_positions,quantum=True)
	
	observables_truth=walkers_truth.evaluate_observables(observables,n_burn,n_steps,subkeys[3])
	observables_ansatz=walkers_ansatz.evaluate_observables(observables,n_burn,n_steps,subkeys[3])

	rel_diff_matrix = np.abs(observables_truth - observables_ansatz) / observables_truth
	max_rel_err = float(max(rel_diff_matrix))
	return max_rel_err

if __name__=='__main__':

	observables=single_particle_moments(3)
		
	args=sys.argv[1:]
	if len(args)==0:
		filename='most_recent'
	else:
		filename=args[0]

	with open('data/'+filename,"rb") as file:
		data=pickle.load(file)
		
		truth=data["true_f"]
		ansatz=data["Ansatz"]
		params=data["params"]

	compare(truth,ansatz,params,observables)


