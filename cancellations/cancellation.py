# author: Nilin 

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
	



#############################################################

odd_angle=lambda x:(jnp.abs(x-1)-abs(x+1)+2*x)/4
ReLU=lambda x:(jnp.abs(x)+x)/2
activation=ReLU
#activation=lambda x:jnp.sin(100*x)
#activation=lambda x:x

odd_angle_leaky=lambda x:odd_angle(x)+.1*x
ReLU_leaky=lambda x:ReLU(x)+.1*x



box=lambda X:jnp.product(jnp.product(jnp.heaviside(1-jnp.square(X),0),axis=-1),axis=-1)

envelope=lambda X:jnp.exp(-jnp.sum(jnp.square(X)))
envelope_FN=envelope

#envelope=box
#envelope_FN=box




class TwoLayer:
	def __init__(self,params,randomness_key):
		
		self.params=params
		d,n,m=params['d'],params['n'],params['m']
		key,*subkeys=jax.random.split(randomness_key,4)

		self.W=jax.random.normal(subkeys[0],shape=(m,n*d))*jnp.sqrt(2/(d*n))
		self.a=jax.random.normal(subkeys[2],shape=(m,))*jnp.sqrt(2/m)

	def eval_raw(self,X):
		X_vec=jnp.ravel(X)
		layer1=activation(jnp.dot(self.W,X_vec))
		return jnp.dot(self.a,layer1)

def antisymmetrize(f):
	def antisymmetric(X):
		y=0
		for P in itertools.permutations(jnp.identity(len(X))):
			sign=jnp.linalg.det(P)
			PX=jnp.matmul(jnp.array(P),X)	
			y+=sign*f(PX)
		return y
	return antisymmetric
	
	
class GenericAntiSymmetric(TwoLayer):
	def __init__(self,params,randomness_key):
		self.activationchoice=1
		super().__init__(params,randomness_key)

	def eval(self,X):
		y=0
		for P in itertools.permutations(jnp.identity(len(X))):
			sign=jnp.linalg.det(P)
			PX=jnp.matmul(jnp.array(P),X)	
			y+=sign*self.eval_raw(PX)
		return y



def distribution(f,X_distribution,samples=100):
	key=jax.random.PRNGKey(np.random.randint(100))
	
	X_list=X_distribution(key,samples)
	y_list=jax.vmap(f)(X_list)

	return y_list




def normalize(X):
	return jax.vmap(lambda x:x/jnp.sqrt(jnp.sum(jnp.square(x))))(X)
	


def Gaussian(n,d):
	return (lambda key,samples:jax.random.normal(key,shape=(samples,n,d)))

def spherical(n,d,radius=1): 
	g=Gaussian(n,d)
	return (lambda key,samples:normalize(g(key,samples))*radius)


"""
def sample_symmetrized_sum(l,terms,n_samples,key):
	
	key,*subkeys=jax.random.split(key,2*n_samples+2)
	samples=[]

	for i in range(n_samples):
		vals=jax.random.choice(subkeys[2*i],l,shape=(terms,))
		signs=jax.random.rademacher(subkeys[2*i+1],shape=(terms,))
		sample=jnp.dot(vals,signs)
		samples.append(sample)
		
	return jnp.array(samples)

def pairwisedistprop(X,loss):
	n=X.shape[-2]
	stacked_x_1=jnp.repeat(jnp.expand_dims(X,-2),n,axis=-2)
	stacked_x_2=jnp.swapaxes(stacked_x_1,-2,-3)
	diffs=stacked_x_1-stacked_x_2
	dists_=jnp.sum(jnp.square(diffs),axis=-1)
	dists=jnp.take(np.partition(dists_,1,axis=-1),jnp.array([i for i in range(1,n)]),axis=-1)
	return jax.vmap(loss)(dists)

mindistsquared=lambda X:pairwisedistprop(X,jnp.min)
inverseloss=lambda x: 1/jnp.sum(1/x)




def plotxy(f,X_distribution,n_samples):

	key=jax.random.PRNGKey(0)
	key,*subkeys=jax.random.split(key,10)

	X=X_distribution(subkeys[1],n_samples)
	X=normalize(X)

	Y_guess=jnp.sqrt(mindistsquared(X))
	#Y_guess=pairwisedistprop(X,inverseloss)
	Y=jnp.sqrt(jax.vmap(f)(X))

	plt.figure()
	plt.scatter(Y_guess,Y,s=2)
	sns.kdeplot(Y_guess,Y,color='r',bw=.2)
	plt.savefig('plots/corr.pdf')
	plt.show()

	print(jnp.corrcoef(Y_guess,Y))


	
def plotxcorr(g,X_distribution,n_samples):
	key=jax.random.PRNGKey(0)
	key,*subkeys=jax.random.split(key,10)
	X=X_distribution(subkeys[1],n_samples)
	X=normalize(X)
	
	#project=antisymmetrize(lambda x:x)
	#pl=lambda x:jnp.sum(jnp.square(project(x)))
	f=lambda x:g(x)**2

	Y=jnp.sum(jnp.square(jnp.average(X,axis=-2)),axis=-1)

	Z=jax.vmap(f)(X)

	print('test:'+str(jnp.corrcoef(Y,Z)))
	
	plt.figure()
	plt.scatter(Y,Z,s=2)
	sns.kdeplot(Y,Z,color='r')
	plt.show()

	
	
	

params={'d':3,'n':6,'m':100}
key=jax.random.PRNGKey(0)
g=GenericAntiSymmetric(params,key)
f=lambda x:g.eval(x)**2/(g.eval_raw(x)**2+.01)
#f=lambda x:g.eval(x)**2
plotxy(f,Gaussian(params['n'],params['d']),10000)	

#plotxcorr(f,Gaussian(params['n'],params['d']),1000)	

def plots123():

	nonsymvars=[]

	for n in range(1,100):
		params={'d':3,'n':n,'m':10}
		k=20
		#X_distribution=lambda key,samples:jax.random.normal(key,shape=(samples,params['n'],params['d']))
		X_distribution=spherical(params['n'],params['d'],jnp.sqrt(params['n']*params['d']))
		key=jax.random.PRNGKey(n)
		key,*subkeys=jax.random.split(key,k+2)
		nonsymvar=[]
		antisymvar=[]
		for i in range(k):
			g=GenericAntiSymmetric(params,subkeys[i])

			nsd=distribution(g.eval_raw,X_distribution)

			nonsymvar.append(jnp.var(nsd))
		
		print('average variance of non-symmetrized: '+str(jnp.average(jnp.array(nonsymvar))))
		print(str(n)+100*'_')

		nonsymvars.append(jnp.median(jnp.array(nonsymvar)))

	
	plt.figure()
	plt.plot(range(2,n+1),nonsymvars[1:n],color='b')
	plt.plot(np.array([1]),np.array([0]))
	plt.savefig('plots/vars_unsym'+str(n)+'.pdf')

	plt.figure()
	plt.plot(range(2,n+1),jnp.log(jnp.array(nonsymvars[1:n])),color='b')
	plt.plot(np.array([1]),np.array([0]))
	plt.savefig('plots/vars_unsym_log'+str(n)+'.pdf')



#plots123()

"""

def plots():

	nonsymvars=[]
	antisymvars=[]

	for n in range(2,9):
		params={'d':3,'n':n,'m':1}
		k=10
		#X_distribution=lambda key,samples:jax.random.normal(key,shape=(samples,params['n'],params['d']))
		#X_distribution=Gaussian(params['n'],params['d'])
		X_distribution=spherical(params['n'],params['d'],jnp.sqrt(params['n']*params['d']))
		key=jax.random.PRNGKey(n)
		key,*subkeys=jax.random.split(key,k+2)
		nonsymvar=[]
		antisymvar=[]
		for i in range(k):
			g=GenericAntiSymmetric(params,subkeys[i])

			nsd=distribution(g.eval_raw,X_distribution)
			asd=distribution(g.eval,X_distribution)

		#		
		#	if(i==0):
		#		plt.figure()
		#		#plt.hist(nsd.tolist(),bins=20)
		#		sns.kdeplot(nsd,bw=.1)
		#		plt.savefig('plots/q'+str(n)+'nsd.pdf')
		#		plt.figure()
		#		#plt.hist(asd.tolist(),bins=20)
		#		sns.kdeplot(asd,bw=.1)
		#		resamples=sample_symmetrized_sum(nsd,math.factorial(n),500,subkeys[i])
		#		sns.kdeplot(resamples,bw=.1,color='r')
		#		plt.savefig('plots/q'+str(n)+'asd.pdf')

			nonsymvar.append(jnp.var(nsd))
			antisymvar.append(jnp.var(asd))
			print('variance of non-symmetrized: '+str(nonsymvar[-1]))
			print('variance of antisymmetrized: '+str(antisymvar[-1]))
			print('')
		
		print('average variance of non-symmetrized: '+str(jnp.average(jnp.array(nonsymvar))))
		print('average variance of antisymmetrized: '+str(jnp.average(jnp.array(antisymvar))))
		print(str(n)+100*'_')

		nonsymvars.append(jnp.median(jnp.array(nonsymvar)))
		antisymvars.append(jnp.median(jnp.array(antisymvar)))

	
		plt.figure()
		plt.plot(range(2,n+1),nonsymvars,color='b')
		plt.savefig('plots/vars_unsym'+str(n)+'.pdf')


		#X=X_distribution(key,100)
		#mindists=mindistsquared(X)
		#ds_estimate=jnp.average(mindists)
		#print(ds_estimate)

		plt.figure()
		plt.plot(range(2,n+1),jnp.log(jnp.array(antisymvars)),color='r')
		plt.plot(range(2,n+1),jnp.log(jnp.array([math.factorial(i) for i in range(2,n+1)])),color='b')
		plt.savefig('plots/vars'+str(n)+'.pdf')



plots()
