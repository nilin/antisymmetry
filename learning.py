# author: Nilin 

import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import matplotlib
from sympy.utilities.iterables import multiset_permutations
from matplotlib.gridspec import GridSpec
import pickle
import time
import copy
import jax
import jax.numpy as jnp
	


#loss: y[0]=ansatz(X), y[1]=truth(X) 
#loss=lambda y:abs(y[0]-y[1])
loss=lambda y:jnp.sqrt((y[0]-y[1])**2+.01)



#############################################################

odd_angle=lambda x:(abs(x-1)-abs(x+1)+2*x)/4
ReLU=lambda x:(abs(x)+x)/2

odd_angle_leaky=lambda x:odd_angle(x)+.1*x
ReLU_leaky=lambda x:ReLU(x)+.1*x

############## true function ################################# 

activation_f=jnp.tanh


############## antisymmetric Ansatz ##########################

activation0=ReLU_leaky
#activation0=jnp.tanh
#activation1=ReLU_leaky
activation1=jnp.tanh
odd_activation=jnp.tanh

PARAM_names=['T','c','V','b','W','a']


class Antisatz:
	def __init__(self,params,randomness_key):
	
		self.params=params
		d,n,d_,p,m=params['d'],params['n'],params['d_'],params['p'],params['m']
		key,*subkeys=jax.random.split(randomness_key,7)

		T=jax.random.uniform(subkeys[0],shape=(d_,d),minval=-1,maxval=1)
		c=jax.random.uniform(subkeys[1],shape=(d_,),minval=-1,maxval=1)
		V=jax.random.uniform(subkeys[2],shape=(p,n,d_),minval=-1,maxval=1)
		b=jax.random.uniform(subkeys[3],shape=(p,n),minval=-1,maxval=1)
		W=jax.random.uniform(subkeys[4],shape=(m,p),minval=-1,maxval=1)
		a=jax.random.uniform(subkeys[5],shape=(m,),minval=-1,maxval=1)

		self.PARAMS={'T':T,'c':c,'V':V,'b':b,'W':W,'a':a}


	def evaluate_(self,X,PARAMS):
		T,c,V,b,W,a=(PARAMS[key] for key in PARAM_names)
		n=self.params['n']
		Z=activation0(jnp.dot(T,X.T)+jnp.repeat(jnp.expand_dims(c,1),n,axis=1))
		square_matrices_list=activation1(jnp.dot(V,Z)+jnp.repeat(jnp.expand_dims(b,2),n,axis=2))
		determinants_list=jax.vmap(jnp.linalg.det)(square_matrices_list)
		layer2=odd_activation(jnp.dot(W,determinants_list))
		return jnp.dot(a,layer2)

	def evaluate(self,X_list):	
		return self.evaluate_(X_list,self.PARAMS)


	def regularize(self,r):
		for key,M in self.PARAMS.items():
			self.PARAMS[key]=jnp.tanh(self.PARAMS[key]/r)*r 

	def sum_loss_(self,data,PARAMS):
		X_list=data[0]
		y_list=data[1]
		f_list=jax.vmap(self.evaluate_,[0,None])(X_list,PARAMS) #X configurations in parallel
		fy_list=jnp.array([f_list,y_list])
		return jnp.sum(jax.vmap(loss,1)(fy_list))


	def update_gradients(self,X_list,y_list):	
		loss,self.dPARAMS=jax.value_and_grad(self.sum_loss_,1)([X_list,y_list],self.PARAMS)
		return loss


	def update(self,learning_rates_tuple):
		learning_rates={PARAM_names[i]:learning_rates_tuple[i] for i in range(len(PARAM_names))}
		for key,M in self.PARAMS.items():
			self.PARAMS[key]-=learning_rates[key]*self.dPARAMS[key]


class TwoLayer:
	def __init__(self,params,randomness_key):
		
		self.params=params
		d,n,m=params['d'],params['n'],params['m']
		key,*subkeys=jax.random.split(randomness_key,4)

		self.W=jax.random.uniform(subkeys[0],shape=(m,n*d),minval=-1,maxval=1)
		self.b=jax.random.uniform(subkeys[1],shape=(m,),minval=-1,maxval=1)
		self.a=jax.random.uniform(subkeys[2],shape=(m,),minval=-1,maxval=1)

	def evaluate(self,X):
		X_vec=jnp.ravel(X)
		layer1=activation_f(jnp.dot(self.W,X_vec)+self.b)
		return jnp.dot(self.a,layer1)
		
	
	
class GenericAntiSymmetric:
	def __init__(self,params,key):
		self.twolayer=TwoLayer(params,key)		
		self.params=params
		self.normalize()

	def normalize(self):# ensure true function has variance 1
		samples=250
		key=jax.random.PRNGKey(123)
		X_list=jax.random.uniform(key,shape=(samples,self.params['n'],self.params['d']),minval=-1,maxval=1)
		y_list=jax.vmap(self.evaluate)(X_list)
		self.twolayer.a/=jnp.sqrt(jnp.var(y_list))

	def evaluate(self,X):
		y=0
		for P in itertools.permutations(jnp.identity(len(X))):
			sign=jnp.linalg.det(P)
			PX=jnp.matmul(jnp.array(P),X)	
			y+=sign*self.twolayer.evaluate(PX)
		return y




def train_on(truth,ansatz,X_list):
	y_list=jax.vmap(truth.evaluate)(X_list)
	loss=ansatz.update_gradients(X_list,y_list)
	return loss

def train(truth,ansatz,params,randkey):
	X_list=jax.random.uniform(randkey,shape=(params['samples'],params['n'],params['d']),minval=-1,maxval=1)
	return train_on(truth,ansatz,X_list)
	
def test_on(ansatz,X_list,y_list):
	f_list=jax.vmap(ansatz.evaluate)(X_list)
	return f_list,jnp.sum((y_list-f_list)**2)


def try_stepsizes_and_learn(truth,ansatz,rates1,rates2,TRAIN_batch_size,test_batch_size,randkey):
	randkey,subkey=jax.random.split(randkey)

	ansatz.regularize(5)
	n=ansatz.params['n']
	d=ansatz.params['d']
	loss=train(truth,ansatz,{'samples':TRAIN_batch_size,'n':n,'d':d},randkey)

	ansatz1=copy.deepcopy(ansatz)
	ansatz2=copy.deepcopy(ansatz)	
	ansatz1.update(rates1)
	ansatz2.update(rates2)

	X_test=jax.random.uniform(subkey,shape=(test_batch_size,n,d),minval=-1,maxval=1)
	y_test=jax.vmap(truth.evaluate)(X_test)
	_,test_error_1=test_on(ansatz1,X_test,y_test)
	_,test_error_2=test_on(ansatz2,X_test,y_test)
	(ansatz,rates)=(ansatz1,rates1) if test_error_1<test_error_2 else (ansatz2,rates2)
	del ansatz1
	del ansatz2
	return ansatz,rates,{'avg_loss':loss/TRAIN_batch_size,'avg_test_error_1':test_error_1/test_batch_size,'avg_test_error_2':test_error_2/test_batch_size,'true_variance':jnp.var(y_test)}
