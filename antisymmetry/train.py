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
import optax







cast_type=lambda val,key:float(val) if key=='threshold' else val if key=='Ansatz' else int(val)

bar=150*'='

def prepdirs():
	foldernames=['theplots','data','params/a','params/s','params/f']
	for folder in foldernames:
		if not os.path.exists(folder):
			os.makedirs(folder)
	counterfile='data/counter'
	if os.path.exists(counterfile):
		ID=int(open(counterfile).read())+1
	else:
		ID=0
	open(counterfile,'w').write(str(ID))
	return ID


def savedata(thedata):
	ID=prepdirs()
	filename='data/ID='+str(ID)

	with open(filename,'wb') as file:
		pickle.dump(thedata,file)
	with open('data/most_recent','wb') as file:
		pickle.dump(thedata,file)
	
	print('\n\n'+bar+'\nData saved as '+filename+'\n'+bar+'\nTo view plots, run\n>>python antisymmetry/plotdata.py\n and press enter when prompted\n\nTo compare observables, run\n>>python antisymmetry/compare.py\n'+bar)

def get_params(paramsfile):
	params={}
	for line in open('params/'+paramsfile):
		key,val=line.split()
		params[key]=cast_type(val,key)
	return params

def print_params(truth,ansatz,params):
	
	print('\n'+bar+'\nTrue function type: '+str(type(truth))+'\nAnsatz type: '+str(type(ansatz))+'\n'+bar)
	print('\nParameters:')
	for key,item in params.items():
		print(str(key)+'='+str(item))
	print('\n')





def initialize(randkey):

	args=sys.argv[1:]
	if(len(args)==0 or args[0] not in {'a','f','s'}):
		print(bar+'\n\nMissing Ansatz type argument. Please run as\n\n>>python antisymmetry.train.py a default\n\nfor Antisatz or\n\n>>python antisymmetry.train.py f default\n\nfor FermiNet or\n\n>>python antisymmetry.train.py s default\n\nfor symmetric. For custom parameters omit \'default\' for prompt or replace by name of parameter file.\n\n'+bar)
		quit()
		
	Ansatztype=args[0]
	params=get_params(Ansatztype+'/default')	

	randkey1,randkey2=jax.random.split(randkey)
	X_distribution=lambda key,samples:jax.random.normal(key,shape=(samples,params['n'],params['d']))

	truth_params={'d':params['d'],'n':params['n'],'m':params['m_truth']}
	truth=learning.GenericSymmetric(truth_params,randkey1) if Ansatztype=='s' else learning.GenericAntiSymmetric(truth_params,randkey1)
	truth.normalize(X_distribution)
			
	ansatz=learning.SymAnsatz(params,randkey2) if Ansatztype=='s' else learning.Antisatz(params,randkey2) if Ansatztype=='a' else learning.FermiNet(params,randkey2)

	print_params(truth,ansatz,params)
	return truth,ansatz,params,X_distribution




if __name__=='__main__':

	randkey=jax.random.PRNGKey(0)
	randkey1,randkey2=jax.random.split(randkey)

	truth,ansatz,params,X_distribution=initialize(randkey1)
	learning.learn(truth,ansatz,params['training_batch_size'],params['batch_count'],randkey2,X_distribution)


	thedata={'true_f':truth,'Ansatz':ansatz,'params':params}
	savedata(thedata)
