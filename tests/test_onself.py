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




params_f_small={'n':3,'d':2,'internal_layer_width':5,'layers':2,'ndets':5}
params_f_large={'n':3,'d':2,'internal_layer_width':20,'layers':3,'ndets':20}

params_a_small={'n':3,'d':2,'m':5,'p':5}
params_a_large={'n':3,'d':2,'m':25,'p':25}

params={'n':3,'d':2,'training_batch_size':1000,'batch_count':200}

X_distribution=lambda key,samples:jax.random.normal(key,shape=(samples,params['n'],params['d']))



def print_train_getplot(randkey,truth,ansatz,optimizer):

	train.print_params(truth,ansatz,{})
	learning.learn(truth,ansatz,params['training_batch_size'],params['batch_count'],randkey,X_distribution,optimizer)

	thedata={'true_f':truth,'Ansatz':ansatz,'params':params}
	train.savedata(thedata)
	
	plots=plotdata.Plots("data/most_recent")
	return plots


########################################################################################################################################################################################################

opt_a=optax.rmsprop(.01)
opt_f=optax.rmsprop(.01)


def test_FermiNet_on_self(randkey):

	randkey,*subkeys=jax.random.split(randkey,4)

	truth=learning.FermiNet(params_f_small,subkeys[0])
	truth.normalize(X_distribution)

	ansatz=learning.FermiNet(params_f_large,subkeys[1])

	return print_train_getplot(subkeys[2],truth,ansatz,opt_f)



def test_Antisatz_on_self(randkey):

	randkey,*subkeys=jax.random.split(randkey,4)

	truth=learning.Antisatz(params_a_small,subkeys[0])
	truth.normalize(X_distribution)

	ansatz=learning.Antisatz(params_a_large,subkeys[1])

	return print_train_getplot(subkeys[2],truth,ansatz,opt_a)


def test_F_on_A(randkey):

	randkey,*subkeys=jax.random.split(randkey,4)

	truth=learning.Antisatz(params_a_small,subkeys[0])
	truth.normalize(X_distribution)

	ansatz=learning.FermiNet(params_f_large,subkeys[1])

	return print_train_getplot(subkeys[2],truth,ansatz,opt_f)


def test_A_on_F(randkey):

	randkey,*subkeys=jax.random.split(randkey,4)

	truth=learning.FermiNet(params_f_small,subkeys[0])
	truth.normalize(X_distribution)

	ansatz=learning.Antisatz(params_a_large,subkeys[1])

	return print_train_getplot(subkeys[2],truth,ansatz,opt_a)


########################################################################################################################################################################################################


randkey=jax.random.PRNGKey(1)
randkey,*subkeys=jax.random.split(randkey,5)

a_plots=test_Antisatz_on_self(subkeys[0])
f_plots=test_FermiNet_on_self(subkeys[1])
f_on_a_plots=test_F_on_A(subkeys[2])
a_on_f_plots=test_A_on_F(subkeys[3])

a_on_f_plots.allplots()
f_on_a_plots.allplots()
f_plots.allplots()
a_plots.allplots()

plt.show()
