from ast import Assert
import jax
import jax.numpy as jnp
import numpy as np
import antisymmetry.learning as learning
import antisymmetry.mcmc as mcmc
import antisymmetry.plotdata as plotdata
import antisymmetry.train as train
import antisymmetry.compare as compare
#import sys
import pickle
#import os

"""
#VERSION ONE: create a list of param file names, train with each parameter set, store plots, and compare their observables.
#set d=2 and n=3, find optimal parameter for l and d_ for ferminet, p and m for antisatz

cast_type=lambda val,key:float(val) if key=='threshold' else int(val)
def write_paramfile(params,model):
	#params should be a dictionary containing the params needed to create a param file.
	filename = ""
	for k,v in params.items():
		filename = filename + str(k)+str(v)
	filepath = "params/"+model+"/"+filename
	for line in open("params/"+model+"/default"):
		key,defaultval=line.split()
		if key not in params:
			params[key]=cast_type(defaultval,key)
	file = open(filepath,"w")
	for k,v in params.items():
		file.write(str(k)+" "+str(v)+"\n")
	file.close()
	return filename


command = ""

#antisatz parameters
model_list = ["a"]
optimal = {'d':2,'n':3,'training_batch_size':1000,'batch_count':1000}
p_list = [80,85]
m_list = [30,50]
#start with these parameters
for p_val in p_list:
	for m_val in m_list:
		dic = optimal.copy()
		dic['p'] = p_val
		dic['m'] = m_val
		for model in model_list:
			param_filename = write_paramfile(dic,model)
			command = command + "python3 antisymmetry/train.py "+ model +" "+param_filename+"; python3 antisymmetry/compare.py; "


#ferminet parameters
optimal = {'d':2,'n':3,'training_batch_size':1000,'batch_count':1000}
model = "f"
optimal = {'layers':4,'internal_layer_width':50,'training_batch_size':1000,'batch_count':1000}
ndets_list = [1,10,16]
for ndets in ndets_list:
	dic = optimal.copy()
	dic['ndets'] = ndets
	param_filename = write_paramfile(dic,model)
	command = command + "python3 antisymmetry/train.py "+ model +" "+param_filename+"; python3 antisymmetry/compare.py; "


os.system(command)
"""

#VERSION TWO: instead of command line calls, move __main__ commands to call here.
cast_type=lambda val,key:float(val) if key=='threshold' else int(val)
def write_paramfile(params,model):
	#params should be a dictionary containing the params needed to create a param file.
	#TODO: create write file function to put these params into a file with space and lines for main.py parsing 
	filename = ""
	for k,v in params.items():
		filename = filename + str(k)+str(v)
	filepath = "params/"+model+"/"+filename
	for line in open("params/"+model+"/default"):
		key,defaultval=line.split()
		if key not in params:
			params[key]=cast_type(defaultval,key)
	file = open(filepath,"w")
	for k,v in params.items():
		file.write(str(k)+" "+str(v)+"\n")
	file.close()
	return filename

def smaller(truth,ansatz,params,observables,cur_min):
	#source: antisymmetry/compare.py/compare()
	#modification: returns whether the current params give a maximum-relative-error smaller than the current minimum.

	n=params['n']
	d=params['d']
	randkey=jax.random.PRNGKey(0); key,*subkeys=jax.random.split(randkey,10)

	n_walkers=1000
	n_burn=250
	n_steps=250

	start_positions=jax.random.uniform(subkeys[1],shape=(n_walkers,n,d))

	amplitude_truth=truth.evaluate	
	amplitude_ansatz=ansatz.evaluate	

	walkers_truth=mcmc.Metropolis(amplitude_truth,start_positions,quantum=True)
	walkers_ansatz=mcmc.Metropolis(amplitude_ansatz,start_positions,quantum=True)
	
	observables_truth=walkers_truth.evaluate_observables(observables,n_burn,n_steps,subkeys[3])
	observables_ansatz=walkers_ansatz.evaluate_observables(observables,n_burn,n_steps,subkeys[3])

	rel_diff_matrix = np.abs(observables_truth - observables_ansatz) / observables_truth
	max_rel_err = float(max(rel_diff_matrix))
	return min(max_rel_err, cur_min)

def initial(randkey,param_file,Ansatztype):
	#source: antisymmetry/train.py/initialize()
	
	params = train.get_params(Ansatztype+"/"+param_file)

	randkey1,randkey2=jax.random.split(randkey)
	X_distribution=lambda key,samples:jax.random.normal(key,shape=(samples,params['n'],params['d']))
	#X_distribution=lambda key,samples:jax.random.uniform(key,shape=(samples,params['n'],params['d']),minval=-1,maxval=1)

	truth_params={'d':params['d'],'n':params['n'],'m':params['m_truth'],'batch_count':params['batch_count']}
	truth=learning.GenericSymmetric(truth_params,randkey1) if Ansatztype=='s' else learning.GenericAntiSymmetric(truth_params,randkey1)
	truth.normalize(X_distribution)
			
	ansatz=learning.SymAnsatz(params,randkey2) if Ansatztype=='s' else learning.Antisatz(params,randkey2) if Ansatztype=='a' else learning.FermiNet(params,randkey2)

	train.print_params(truth,ansatz,params)
	return truth,ansatz,params,X_distribution

def training(paramfile, cur_min, ansatztype):
	randkey=jax.random.PRNGKey(0)
	randkey1,randkey2=jax.random.split(randkey)

	truth,ansatz,params,X_distribution=initial(randkey1,paramfile,ansatztype)
	learning.learn(truth,ansatz,params['training_batch_size'],params['batch_count'],randkey2,X_distribution)
	thedata={'true_f':truth,'Ansatz':ansatz,'params':params}
	train.savedata(thedata)

	observables=compare.single_particle_moments(3)

	with open('data/most_recent',"rb") as file:
		data=pickle.load(file)
		truth=data["true_f"]
		ansatz=data["Ansatz"]
		params=data["params"]
	return smaller(truth,ansatz,params,observables,cur_min)

if __name__=='__main__':
	optimal = {'d':2,'n':3,'training_batch_size':1000,'batch_count':1000}
	cur_min = 10e7
	stack = []
	params_dict = {}

	l_list = [50,100]
	width_list = [50,100]
	#l_list = jnp.arange(50,200,50)
	#width_list = jnp.arange(10,200,50)
	#dets_list = [2**k for k in jnp.arange(0,9)]
	dets_list = [1, 10, 256]
	for l_val in l_list:
		for w_val in width_list:
			for n_val in dets_list:
				dic = optimal.copy()
				dic['layers'] = l_val
				dic['internal_layer_width'] = w_val
				dic['ndets'] = n_val
				param_filename = write_paramfile(dic,"f")
				if len(stack) >= 5:
					params_dict.pop(stack[0])
					stack = stack[1:]
				new_min = training(param_filename,cur_min, "f")
				stack.append(new_min)
				params_dict[new_min] = dic
	print(params_dict)
					
				