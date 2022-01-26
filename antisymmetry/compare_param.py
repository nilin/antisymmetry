from ast import Assert
import jax
import jax.numpy as jnp
import numpy as np
import antisymmetry.learning as learning
import antisymmetry.train as train
import antisymmetry.compare as compare
import pickle

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

def training(paramfile, ansatztype):
	randkey=jax.random.PRNGKey(0)
	randkey1,randkey2=jax.random.split(randkey)

	truth,ansatz,params,X_distribution=train.initial(randkey1,paramfile,ansatztype)
	learning.learn(truth,ansatz,params['training_batch_size'],params['batch_count'],randkey2,X_distribution)
	thedata={'true_f':truth,'Ansatz':ansatz,'params':params}
	train.savedata(thedata)

	observables=compare.single_particle_moments(3)

	with open('data/most_recent',"rb") as file:
		data=pickle.load(file)
		truth=data["true_f"]
		ansatz=data["Ansatz"]
		params=data["params"]
	return compare.get_max(truth,ansatz,params,observables)

def put_in_stack(stack, new_item):
	#given a list (assume sorted from min to max) and a new item, add the new item to the list to maintain sorting
	end = len(stack)
	look = 0
	while look < end and new_item > stack[look]:
		look += 1
	if look >= end:
		stack.append(new_item)
	else:
		stack = stack[:look] +[new_item] + stack[look:]

if __name__=='__main__':
	optimal = {'d':2,'n':3,'training_batch_size':1000,'batch_count':1000}
	cur_min = 10e7
	stack = []
	params_dict = {}

	l_list = [4,10,20]
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
				new_item = training(param_filename,"f")
				put_in_stack(stack, new_item)
				params_dict[new_item] = dic
	for i in range(5):
		print(stack[i])
		print(params_dict[stack[i]])
					
				