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










def input_remaining_params(params,paramsfolder):

	for line in open(paramsfolder+'/default'):
		key,defaultval=line.split()
		if key not in params:
			if key in train.descriptions:
				desc=train.descriptions[key]
			else:
				desc=key
			i=input('\nInput '+desc+' '*(30-len(desc))+'(or press ENTER for default value).      '+key+'=')
			if i=='':
				params[key]=train.cast_type(defaultval,key)
			else:
				params[key]=train.cast_type(i,key)

def initialize_interactive(ID,randkey,args):

	if(len(args)==0 or args[0] not in {'a','f','s'}):
		print('='*100+'\n\nMissing Ansatz type argument. Please run as\n\n>>python antisymmetry/train_interactive.py a default\n\nfor Antisatz or\n\n>>python antisymmetry/train_interactive.py f default\n\nfor FermiNet or\n\n>>python antisymmetry/train_interactive.py s default\n\nfor symmetric. For custom parameters omit \'default\' for prompt or replace by name of parameter file.\n\n'+'='*100)
		quit()
		
	Ansatztype=args[0]
	paramsfolder='params/'+Ansatztype
	antistring='' if Ansatztype=='s' else 'anti'


	if(len(args)>1):
		firstinput=args[1]
	else:
		firstinput=input('\n'+\
		'='*100+'\nPress ENTER to generate '+antistring+'symmetric function from default parameters.\n'+'='*100+'\n'+\
		'or\nType name of parameter file. Type \'m\' to input parameters manually.\n'+\
		'To import true '+antistring+'symmetric function from saved data, type \'i\'.\n'+'='*100)

	params={}
	loaded={}
	if(firstinput=='i'):
		filename=input('Type name of file with true '+antistring+'symmetrized function. Press enter for most recent ')
		if filename=='':
			filename='most_recent_true_f'
		with open('data/'+filename,'rb') as file:
			data_true_f=pickle.load(file)
		truth=data_true_f['f']
		loaded['true_f']=True
		params=params|data_true_f['params']

	elif firstinput!='m':
		if(firstinput==''):
			firstinput='default'
		for line in open(paramsfolder+'/'+firstinput):
			key,val=line.split()
			params[key]=train.cast_type(val,key)

	input_remaining_params(params,paramsfolder)

	randkey1,randkey2=jax.random.split(randkey)
	X_distribution=lambda key,samples:jax.random.normal(key,shape=(samples,params['n'],params['d']))

	if 'true_f' not in loaded:
		truth_params={'d':params['d'],'n':params['n'],'m':params['m_truth']}
		truth=learning.GenericSymmetric(truth_params,randkey1) if Ansatztype=='s' else learning.GenericAntiSymmetric(truth_params,randkey1)

		truth.normalize(X_distribution)
				
	if 'Ansatz' not in loaded:
		ansatz=learning.SymAnsatz(params,randkey2) if Ansatztype=='s' else learning.Antisatz(params,randkey2) if Ansatztype=='a' else learning.FermiNet(params,randkey2)

	train.print_params(params,paramsfolder+'/default',train.descriptions,Ansatztype)
	return Ansatztype,truth,ansatz,params,X_distribution


train.run(initialize_interactive)
