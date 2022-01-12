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









def initialize_learn_self(ID,randkey,args):

	paramsfile='params/test_Antisatz_on_self'
	Ansatztype='a'

	params={}
	for line in open(paramsfile):
		key,val=line.split()
		params[key]=train.cast_type(val,key)


	randkey1,randkey2=jax.random.split(randkey)
	X_distribution=lambda key,samples:jax.random.normal(key,shape=(samples,params['n'],params['d']))

			
	truth=learning.Antisatz(params,randkey1)
	truth.normalize(X_distribution)

	ansatz=learning.Antisatz(params,randkey2)
	#ansatz.normalize(X_distribution)


	train.print_params(params,paramsfile,train.descriptions,Ansatztype)
	return Ansatztype,truth,ansatz,params,X_distribution


train.run(initialize_learn_self,optimizer=optax.rmsprop(.01))


plots=plotdata.Plots("data/most_recent")
plots.allplots()
