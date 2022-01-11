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


cast_type=lambda val,key:float(val) if key=='threshold' else val if key=='Ansatz' else int(val)





def run(initialize):

	######## create data folders ############

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

	#########################################
	
	

	args=sys.argv[1:]
	randkey=jax.random.PRNGKey(0)
	randkey1,randkey2=jax.random.split(randkey)

	Ansatztype,truth,ansatz,params,X_distribution=initialize(str(ID),randkey1,args)

	losslist=[]
	rate=.0001
	training_batch_size=params['training_batch_size']
	test_error=float('inf')
	true_variance=1

	losses=learning.learn(truth,ansatz,.01,training_batch_size,params['batch_count'],randkey2,X_distribution)


	print('\n\n'+'='*100+'\nDone. \n'+100*'='+'\nTo view plots, run\n>>python plotdata.py\n and press enter when prompted\n\nTo compare observables, run\n>>python compare.py\n'+100*'=')




	######### save ###############################

	savedata={'Ansatztype':Ansatztype,'true_f':truth,'Ansatz':ansatz,'params':params,'losslist':losslist}
	with open('data/ID='+str(ID),'wb') as file:
		pickle.dump(savedata,file)
	with open('data/most_recent','wb') as file:
		pickle.dump(savedata,file)
	
	##############################################






def print_params(params,paramsfile,descriptions,Ansatztype):

	antistring='' if Ansatztype=='s' else 'anti'
	Azdict={'a':'Antisatz','f':'FermiNet','s':'symmetric'}
	print('\n'+'='*100+'\nParameters chosen, '+antistring+'symmetric case, '+'Ansatz type '+Azdict[Ansatztype]+'\n')
	for line in open(paramsfile):
		key,_=line.split()
		if key in descriptions:
			desc=descriptions[key]
		else:
			desc=key
		print(desc+': '+key+'='+str(params[key]))
	print('='*100+'\n')





def initialize(ID,randkey,args):

	if(len(args)==0 or args[0] not in {'a','f','s'}):
		print('='*100+'\n\nMissing Ansatz type argument. Please run as\n\n>>python3 train.py a default\n\nfor Antisatz or\n\n>>python3 train.py f default\n\nfor FermiNet or\n\n>>python3 train.py s default\n\nfor symmetric. For custom parameters omit \'default\' for prompt or replace by name of parameter file.\n\n'+'='*100)
		quit()
		
	Ansatztype=args[0]
	paramsfolder='params/'+Ansatztype
	paramsfile=paramsfolder+'/default'
	
	params={}
	for line in open(paramsfile):
		key,val=line.split()
		params[key]=cast_type(val,key)


	randkey1,randkey2=jax.random.split(randkey)
	#X_distribution=lambda key,samples:jax.random.uniform(key,shape=(samples,params['n'],params['d']),minval=-1,maxval=1)
	X_distribution=lambda key,samples:jax.random.normal(key,shape=(samples,params['n'],params['d']))

	truth_params={'d':params['d'],'n':params['n'],'m':params['m_truth']}
	truth=learning.GenericSymmetric(truth_params,randkey1) if Ansatztype=='s' else learning.GenericAntiSymmetric(truth_params,randkey1)
	truth.normalize(X_distribution)
			
	ansatz=learning.SymAnsatz(params,randkey2) if Ansatztype=='s' else learning.Antisatz(params,randkey2) if Ansatztype=='a' else learning.FermiNet(params,randkey2)

	print_params(params,paramsfile,descriptions,Ansatztype)
	return Ansatztype,truth,ansatz,params,X_distribution




if __name__=='__main__':
	run(initialize)
