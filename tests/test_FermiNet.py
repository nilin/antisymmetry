import antisymmetry.learning as learning
import antisymmetry.mcmc as mcmc
import antisymmetry.plotdata as plotdata
import antisymmetry.train as train
import antisymmetry.compare as compare
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import sys
import os
import optax




def initialize(randkey):

	params=train.get_params('f/heavy')	
	#params=train.get_params('f/default')	

	randkey1,randkey2=jax.random.split(randkey)
	X_distribution=lambda key,samples:jax.random.normal(key,shape=(samples,params['n'],params['d']))

	truth_params={'d':params['d'],'n':params['n'],'m':params['m_truth']}
	truth=learning.GenericAntiSymmetric(truth_params,randkey1)
	truth.normalize(X_distribution)
			
	ansatz=learning.FermiNet(params,randkey2)

	train.print_params(truth,ansatz,params)
	return truth,ansatz,params,X_distribution



def test_FermiNet():

	randkey=jax.random.PRNGKey(0)
	randkey1,randkey2=jax.random.split(randkey)

	truth,ansatz,params,X_distribution=initialize(randkey1)
	learning.learn(truth,ansatz,params['training_batch_size'],params['batch_count'],randkey2,X_distribution)

	thedata={'true_f':truth,'Ansatz':ansatz,'params':params}
	train.savedata(thedata)

	return truth,ansatz,params



truth,ansatz,params=test_FermiNet()

plots=plotdata.Plots("data/most_recent")
plots.allplots()
plt.show()


observables=compare.single_particle_moments(2)
compare.compare(truth,ansatz,params,observables,.2)
