import antisymmetry.learning as learning
import antisymmetry.mcmc as mcmc
import antisymmetry.plotdata as plotdata
import antisymmetry.train as train
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import sys
import os
import optax





def test_FermiNet_on_self():

	params_t={'n':3,'d':2,'internal_layer_width':5,'layers':2,'ndets':5}
	params_a={'n':3,'d':2,'internal_layer_width':20,'layers':3,'ndets':20}
	params={'n':3,'d':2,'true':params_t,'Ansatz':params_a,'training_batch_size':1000,'batch_count':500}

	randkey=jax.random.PRNGKey(0); randkey1,randkey2=jax.random.split(randkey)

	X_distribution=lambda key,samples:jax.random.normal(key,shape=(samples,params['n'],params['d']))

	truth=learning.FermiNet(params_t,randkey1)
	truth.normalize(X_distribution)

	ansatz=learning.FermiNet(params_a,randkey2)
	ansatz.normalize(X_distribution)

	train.print_params(truth,ansatz,params)


	learning.learn(truth,ansatz,params['training_batch_size'],params['batch_count'],randkey2,X_distribution,optimizer=optax.rmsprop(.01))

	thedata={'true_f':truth,'Ansatz':ansatz,'params':params}
	train.savedata(thedata)
	
	plots=plotdata.Plots("data/most_recent")
	return plots



def test_Antisatz_on_self():

	params_t={'n':3,'d':2,'m':5,'p':5}
	params_a={'n':3,'d':2,'m':20,'p':20}
	params={'n':3,'d':2,'true':params_t,'Ansatz':params_a,'training_batch_size':1000,'batch_count':200}
	randkey=jax.random.PRNGKey(0); randkey1,randkey2=jax.random.split(randkey)

	X_distribution=lambda key,samples:jax.random.normal(key,shape=(samples,params['n'],params['d']))

	truth=learning.Antisatz(params_t,randkey1)
	truth.normalize(X_distribution)

	ansatz=learning.Antisatz(params_a,randkey2)
	ansatz.normalize(X_distribution)

	train.print_params(truth,ansatz,params)


	learning.learn(truth,ansatz,params['training_batch_size'],params['batch_count'],randkey2,X_distribution,optimizer=optax.rmsprop(.01))

	thedata={'true_f':truth,'Ansatz':ansatz,'params':params}
	train.savedata(thedata)


	plots=plotdata.Plots("data/most_recent")
	return plots



f_plots=test_FermiNet_on_self()
a_plots=test_Antisatz_on_self()

f_plots.allplots()
a_plots.allplots()

plt.show()
