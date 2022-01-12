import jax.numpy as jnp
import numpy as np
import sys
import os





#goal: to read a list of param file names and run main.py on each file, store plots, and [after implementation] compare their observables.
cast_type=lambda val,key:float(val) if key=='threshold' else int(val)
def write_paramfile(params,model):
	#params should be a dictionary containing the params needed to create a param file.
	#TODO: create write file function to put these params into a file with space and lines for main.py parsing 
	filename = ""
	for k,v in params.items():
		filename = filename + str(k)+str(v)
	filepath = "params/"+model+"/"+filename
	for line in open("params/"+model+"/"+'/default'):
		key,defaultval=line.split()
		if key not in params:
			params[key]=cast_type(defaultval,key)
	file = open(filepath,"w")
	for k,v in params.items():
		file.write(str(k)+" "+str(v)+"\n")
	file.close()
	return filename



#set d=2 and n=3, find optimal parameter for l and d_ for ferminet, p and m for antisatz
optimal = {'d':2, 'n':3}
command = ""

"""
#antisatz parameters
#model_list = ["a"]
#p_list = jnp.arange(48, 52)
#m_list = jnp.arange(48, 52)
#p_list = [20, 50]
#m_list = [25,50,75]
for p_val in p_list:
	for m_val in m_list:
		dic = optimal.copy()
		dic['p'] = p_val
		dic['m'] = m_val
		for model in model_list:
			param_filename = write_paramfile(dic,model)
			command = command + "python3 train.py "+ model +" "+param_filename+"; python3 compare.py; "
"""

#ferminet parameters
model_list = ["f"]
l_list = jnp.arange(2,6)
d__list = jnp.arange(70,120,10)

for l_val in l_list:
	for d_val in d__list:
		dic = optimal.copy()
		dic['l'] = l_val
		dic['d_'] = d_val
		for model in model_list:
			param_filename = write_paramfile(dic,model)
			command = command + "python3 train.py "+ model +" "+param_filename+"; python3 compare.py; "



os.system(command)

#TODO:
#Write results to file 
#pull new changes, create branch, commit to branch.