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

	params=train.get_params('test_FermiNet_on_self')
	randkey=jax.random.PRNGKey(0); randkey1,randkey2=jax.random.split(randkey)

	X_distribution=lambda key,samples:jax.random.normal(key,shape=(samples,params['n'],params['d']))

	truth=learning.FermiNet(params,randkey1)
	truth.normalize(X_distribution)

	ansatz=learning.FermiNet(params,randkey2)
	ansatz.normalize(X_distribution)

	train.print_params(truth,ansatz,params)


	learning.learn(truth,ansatz,params['training_batch_size'],params['batch_count'],randkey2,X_distribution,optimizer=optax.rmsprop(.001))

	thedata={'true_f':truth,'Ansatz':ansatz,'params':params}
	train.savedata(thedata)



def test_Antisatz_on_self():

	params=train.get_params('test_Antisatz_on_self')
	randkey=jax.random.PRNGKey(0); randkey1,randkey2=jax.random.split(randkey)

	X_distribution=lambda key,samples:jax.random.normal(key,shape=(samples,params['n'],params['d']))

	truth=learning.Antisatz(params,randkey1)
	truth.normalize(X_distribution)

	ansatz=learning.Antisatz(params,randkey2)
	ansatz.normalize(X_distribution)

	train.print_params(truth,ansatz,params)


	learning.learn(truth,ansatz,params['training_batch_size'],params['batch_count'],randkey2,X_distribution,optimizer=optax.rmsprop(.001))

	thedata={'true_f':truth,'Ansatz':ansatz,'params':params}
	train.savedata(thedata)
	plots=plotdata.Plots("data/most_recent")
	plots.allplots()





test_Antisatz_on_self()
#test_FermiNet_on_self()
