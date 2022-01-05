import antisymmetry.learning as learning
import antisymmetry.mcmc as mcmc
import antisymmetry.plotdata as plotdata
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import sys
import os







descriptions={'training_batch_size':'training batch sizes','d':'spatial dimension','n':'number of particles','m_truth':'number of features in true function','threshold':'test error at which to stop training','p':'size of layer 1 in Ansatz','m':'size of layer 2 in Ansatz','L':'number of layers in FermiNet Ansatz'}


cast_type=lambda val,key:float(val) if key=='threshold' else int(val)

def input_remaining_params(params,paramsfolder):
	for line in open(paramsfolder+'/default'):
		key,defaultval=line.split()
		if key not in params:
			if key in descriptions:
				desc=descriptions[key]
			else:
				desc=key
			i=input('\nInput '+desc+' '*(30-len(desc))+'(or press ENTER for default value).      '+key+'=')
			if i=='':
				params[key]=cast_type(defaultval,key)
			else:
				params[key]=cast_type(i,key)

def initialize(ID,randkey,args):

	if(len(args)==0 or args[0] not in {'a','s'}):
		print('='*100+'\n\nMissing symmetry type argument. Please run as\n\n>>python3 train.py a default\n\nfor antisymmetric or\n\n>>python3 train.py s default\n\nfor symmetric. For custom parameters omit \'default\' for prompt or replace by name of parameter file.\n\n'+'='*100)
		quit()
		
	symtype=args[0]
	paramsfolder='params/'+symtype
	antistring='anti' if symtype=='a' else ''


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
			params[key]=cast_type(val,key)

	input_remaining_params(params,paramsfolder)
	paramtext=''.join(['_'+key+'='+str(val) for key,val in params.items()])

	randkey1,randkey2=jax.random.split(randkey)

	if 'true_f' not in loaded:
		truth_params={'d':params['d'],'n':params['n'],'m':params['m_truth']}
		truth=learning.GenericSymmetric(truth_params,randkey1) if symtype=='s' else learning.GenericAntiSymmetric(truth_params,randkey1)
		savedata={'f':truth,'params':{key:params[key] for key in {'n','d','m_truth'}}}
		with open('data/most_recent_true_f','wb') as file:
			pickle.dump(savedata,file)
		with open('data/true_f'+paramtext+'_ID='+ID,'wb') as file:
			pickle.dump(savedata,file)
				
	if 'Ansatz' not in loaded:
		ansatz=learning.SymAnsatz(params,randkey2) if symtype=='s' else learning.Antisatz(params,randkey2)
		#ansatz=learning.SymAnsatz(params,randkey2) if symtype=='s' else learning.FermiNet(params,randkey2)
		with open('data/initial_guess'+paramtext+'_ID='+ID,'wb') as file:
			pickle.dump({'Ansatz':ansatz,'params':{key:params[key] for key in {'n','d'}}},file)

	print_params(params,paramsfolder,descriptions,antistring)
	return symtype,truth,ansatz,params,paramtext

def print_params(params,paramsfolder,descriptions,antistring):
	print('\n'+'='*100+'\nParameters chosen, '+antistring+'symmetric case\n')
	for line in open(paramsfolder+'/default'):
		key,_=line.split()
		if key in descriptions:
			desc=descriptions[key]
		else:
			desc=key
		print(desc+': '+key+'='+str(params[key]))
	print('='*100+'\n')


####################################################################################################################################################################################################
####################################################################################################################################################################################################
####################################################################################################################################################################################################


foldernames=['theplots','data','params/a','params/s']
for folder in foldernames:
	if not os.path.exists(folder):
		os.makedirs(folder)



args=sys.argv[1:]

ID=str(time.time())

randkey0=jax.random.PRNGKey(0)

randkey1,subkey=jax.random.split(randkey0)
symtype,truth,ansatz,params,paramtext=initialize(ID,subkey,args)

losslist=[]
rate=.0001
training_batch_size=params['training_batch_size']
test_error=float('inf')
true_variance=1

randkey=randkey1


losses=learning.learn(truth,ansatz,.01,training_batch_size,params['batch_count'],randkey)


savedata={'symtype':symtype,'true_f':truth,'Ansatz':ansatz,'params':params,'losslist':losslist}
with open('data/true_f_and_learned_ansatz'+paramtext+'_ID='+ID,'wb') as file:
	pickle.dump(savedata,file)
with open('data/most_recent','wb') as file:
	pickle.dump(savedata,file)


print('\n\n'+'='*100+'\nDone. \n'+100*'='+'\nTo view plots, run\n>>python plotdata.py\n and press enter when prompted\n\nTo compare observables, run\n>>python compare.py\n'+100*'=')
#if(input('\n\n'+'='*100+'\nDone after '+str(training_batch_size*(len(losslist)+1))+' samples. Plot data? (y/n): ')=='y'):
#	plots=plotdata.Plots('data/most_recent')
#	plots.allplots()
