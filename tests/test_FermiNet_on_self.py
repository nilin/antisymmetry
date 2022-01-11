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









def initialize_learn_self(ID,randkey,args):

	paramsfolder='params/f'
	Ansatztype='f'

	params={}
	for line in open(paramsfolder+'/default'):
		key,val=line.split()
		params[key]=train.cast_type(val,key)


	randkey1,randkey2=jax.random.split(randkey)
	X_distribution=lambda key,samples:jax.random.normal(key,shape=(samples,params['n'],params['d']))

			
	truth=learning.FermiNet(params,randkey1)
	truth.normalize(X_distribution)

	ansatz=learning.FermiNet(params,randkey2)
	ansatz.normalize(X_distribution)


	train.print_params(params,paramsfolder,train.descriptions,Ansatztype)
	return Ansatztype,truth,ansatz,params,X_distribution


train.run(initialize_learn_self)
