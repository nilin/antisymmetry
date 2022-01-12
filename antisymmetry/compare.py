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
			
"""
filename=input("type name of file to plot or press enter for most recent. ")
if(filename==""):
	filename="most_recent"
"""		
filename="most_recent"

with open('data/'+filename,"rb") as file:
	alldata=pickle.load(file)
	
	truth=alldata["true_f"]
	ansatz=alldata["Ansatz"]
	params=alldata["params"]

	n=params['n']
	d=params['d']





def compare_observables(observables):
		

	randkey0=jax.random.PRNGKey(np.random.randint(100))

	key,*subkeys=jax.random.split(randkey0,10)

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

	#print('observables true function:\n'+str(observables_truth))
	#print('observables Ansatz:\n'+str(observables_ansatz))

	diff_matrix = jnp.abs(observables_truth-observables_ansatz)
	rel_err_matrix = jnp.divide(diff_matrix,observables_truth)
	rel_err = jnp.abs(jnp.mean(rel_err_matrix))
	print(rel_err, params)



##################################################################################################################################################################


def single_particle_moments(K):
	def momentsfunction(X):
		x=jnp.concatenate([jnp.array([1]),X[0]],axis=0)
		tensor=jnp.array([1])
		for i in range(K):
			tensor=jnp.kron(tensor,x)
		return tensor
	return momentsfunction

	



compare_observables(single_particle_moments(3))
	
