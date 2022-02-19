import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import matplotlib
from sympy.utilities.iterables import multiset_permutations
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pickle
import time
import copy
import jax
import jax.numpy as jnp
import optax
import cancellation as canc
import cancellation_full as full
	



def plots():

	m_list=[1,10,100]
	antisymvars={m:[] for m in m_list}

	for n in range(2,15):
		plt.figure()


		for m in m_list:
		
			params={'d':3,'n':n,'m':m}
			k=25
			X_distribution=canc.Gaussian(params['n'],params['d'])
			key=jax.random.PRNGKey(n)
			key,*subkeys=jax.random.split(key,k+2)
			antisymvar=[]
			for i in range(k):
				tl=canc.TwoLayer(params,subkeys[i])
				f=tl.evaluate
				g=canc.antisymmetrize(f)

				asd=canc.distribution(g,X_distribution)

				antisymvar.append(jnp.var(asd))
				print('variance of antisymmetrized: '+str(antisymvar[-1]))
				print('')
			
			print('average variance of antisymmetrized: '+str(jnp.average(jnp.array(antisymvar))))
			print(str(n)+100*'_')

			antisymvars[m].append(jnp.average(jnp.array(antisymvar)))
			plt.plot(range(2,n+1),jnp.array(antisymvars[m]),color='r')
			plt.scatter(range(2,n+1),jnp.array(antisymvars[m]),color='r')

		savedata(antisymvars,'vars_old')	

		plt.yscale('log')
		plt.plot(range(2,n+1),jnp.array([math.factorial(i) for i in range(2,n+1)]),color='b')
		plt.savefig('plots/vars_old_'+str(n)+'.pdf')



def plot_non_sym():

	m_list=[1,10,100]
	_vars={m:[] for m in m_list}

	for n in range(2,100):

		for m in m_list:
		
			params={'d':3,'n':n,'m':m}
			k=25
			X_distribution=canc.Gaussian(params['n'],params['d'])
			key=jax.random.PRNGKey(n)
			key,*subkeys=jax.random.split(key,k+2)
			_var=[]
			for i in range(k):
				tl=canc.TwoLayer(params,subkeys[i])
				f=tl.evaluate

				nsd=canc.distribution(f,X_distribution)

				_var.append(jnp.var(nsd))
				print('variance of _metrized: '+str(_var[-1]))
				print('')
			
			print('average variance of _metrized: '+str(jnp.average(jnp.array(_var))))
			print(str(n)+100*'_')

			_vars[m].append(jnp.average(jnp.array(_var)))
	plt.figure()
	for m in m_list:
		plt.plot(range(2,n+1),jnp.array(_vars[m]),color='b')
	plt.ylim(bottom=0)
	plt.savefig('plots/vars_old_nonsym_'+str(n)+'.pdf')
	savedata(_vars,'vars_old')	


def savedata(thedata,filename):
        filename='data/'+filename
        with open(filename,'wb') as file:
                pickle.dump(thedata,file)

plot_non_sym()
plots()
