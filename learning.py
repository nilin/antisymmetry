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
import optax
	


#loss: y[0]=ansatz(X), y[1]=truth(X) 
#loss=lambda y:abs(y[0]-y[1])
#loss=lambda y:jnp.sqrt((y[0]-y[1])**2+.01)
loss=lambda y:(y[0]-y[1])**2



#############################################################

odd_angle=lambda x:(jnp.abs(x-1)-abs(x+1)+2*x)/4
ReLU=lambda x:(jnp.abs(x)+x)/2

odd_angle_leaky=lambda x:odd_angle(x)+.1*x
ReLU_leaky=lambda x:ReLU(x)+.1*x

############## true (anti)symmetrized functions ################################# 

activations_truth=[jnp.abs,jnp.tanh] #[s,a]


############## antisymmetric Ansatz ##########################

activation0a=ReLU_leaky
activation1a=jnp.tanh
odd_activation=jnp.tanh


############## symmetric Ansatz ##########################

activation0s=ReLU_leaky
activation1s=ReLU_leaky



class Ansatz:

	def __init__(self):
		self.velocity={key:0*M for key,M in self.PARAMS.items()}


	def evaluate(self,X_list):	
		return self.evaluate_(X_list,self.PARAMS)



	def sum_loss(self,PARAMS,X_list,y_list):
		f_list=jax.vmap(self.evaluate_,[0,None])(X_list,PARAMS) #X configurations in parallel
		fy_list=jnp.array([f_list,y_list])
		return jnp.sum(jax.vmap(loss,1)(fy_list))


	def regularize(self,r):
		for key,M in self.PARAMS.items():
			self.PARAMS[key]=jnp.tanh(self.PARAMS[key]/r)*r 
#	def update_gradients(self,X_list,y_list):	
#		loss,self.dPARAMS=jax.value_and_grad(self.sum_loss,0)(self.PARAMS,X_list,y_list)
#		return loss
#
#
#	def update(self,learning_rate):
#		r=.9
#		for key,M in self.PARAMS.items():
#			self.velocity[key]=-(1-r)*learning_rate*self.dPARAMS[key]+r*self.velocity[key]
#			self.PARAMS[key]+=self.velocity[key]



class FermiNet():
		
	def __init__(self,params,randomness_key):
		self.params=params
		d_list=params['d_list']
		key,*subkeys=jax.random.split(randomness_key,3+3*len(d_list))
		self.W_list,self.V_list,self.b_list=[],[],[]
		for l in range(1,len(d_list)):
			d_=d_list[l],d=d_list[l-1]
			W=jax.random.uniform(subkeys[3*l-3],shape=(d_,d),minval=-1,maxval=1)
			V=jax.random.uniform(subkeys[3*l-2],shape=(d_,d),minval=-1,maxval=1)
			b=jax.random.uniform(subkeys[3*l-1],shape=(d_,1),minval=-1,maxval=1)
			self.W_list.append(W)
			self.V_list.append(V)
			self.b_list.append(b)

		

	def evaluate_(self,X,PARAMS):
		W_list=PARAMS['W_list']
		V_list=PARAMS['V_list']
		b_list=PARAMS['b_list']
		X=X.T
		for l in range(len(self.d_list)):
			W=W_list[l]
			V=V_list[l]
			b=b_list[l]
			S=jax.sum(X,axis=1)
			Y=jnp.tanh(jnp.dot(W,X)+jnp.repeat(jnp.dot(V,S)+b,n,axis=1))
			X=Y
		return jnp.det(Y)


class Antisatz(Ansatz):

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
		super().__init__()


	def evaluate_(self,X,PARAMS):
		T,c,V,b,W,a=(PARAMS[key] for key in ['T','c','V','b','W','a'])
		n=self.params['n']
		Z=activation0a(jnp.dot(T,X.T)+jnp.repeat(jnp.expand_dims(c,1),n,axis=1))
		square_matrices_list=activation1a(jnp.dot(V,Z)+jnp.repeat(jnp.expand_dims(b,2),n,axis=2))
		determinants_list=jax.vmap(jnp.linalg.det)(square_matrices_list)
		layer2=odd_activation(jnp.dot(W,determinants_list))
		return jnp.dot(a,layer2)



class SymAnsatz(Ansatz):

	def __init__(self,params,randomness_key):
	
		self.params=params
		d,n,p,m=params['d'],params['n'],params['p'],params['m']
		key,*subkeys=jax.random.split(randomness_key,7)

		V=jax.random.uniform(subkeys[2],shape=(p,d),minval=-1,maxval=1)
		c=jax.random.uniform(subkeys[1],shape=(p,),minval=-1,maxval=1)
		W=jax.random.uniform(subkeys[4],shape=(m,p),minval=-1,maxval=1)
		b=jax.random.uniform(subkeys[4],shape=(m,),minval=-1,maxval=1)
		a=jax.random.uniform(subkeys[5],shape=(m,),minval=-1,maxval=1)

		self.PARAMS={'V':V,'c':c,'W':W,'b':b,'a':a}
		super().__init__()


	def evaluate_(self,X,PARAMS):
		V,c,W,b,a=(PARAMS[key] for key in ['V','c','W','b','a'])
		n=self.params['n']
		S=jnp.sum(activation0s(jnp.dot(V,X.T)+jnp.repeat(jnp.expand_dims(c,1),n,axis=1)),axis=1)
		layer2=activation1s(jnp.dot(W,S)+b)
		return jnp.dot(a,layer2)



class TwoLayer:
	def __init__(self,params,randomness_key):
		
		self.params=params
		d,n,m=params['d'],params['n'],params['m']
		key,*subkeys=jax.random.split(randomness_key,4)

		self.W=jax.random.uniform(subkeys[0],shape=(m,n*d),minval=-1,maxval=1)
		self.b=jax.random.uniform(subkeys[1],shape=(m,),minval=-1,maxval=1)
		self.a=jax.random.uniform(subkeys[2],shape=(m,),minval=-1,maxval=1)

		self.normalize()

	def eval_raw(self,X):
		X_vec=jnp.ravel(X)
		layer1=activations_truth[self.activationchoice](jnp.dot(self.W,X_vec)+self.b)
		return jnp.dot(self.a,layer1)
		
	def normalize(self):# ensure true function has variance 1
		samples=250
		key=jax.random.PRNGKey(123)
		X_list=jax.random.uniform(key,shape=(samples,self.params['n'],self.params['d']),minval=-1,maxval=1)
		y_list=jax.vmap(self.evaluate)(X_list)
		self.a/=jnp.sqrt(jnp.var(y_list))
	
	
class GenericAntiSymmetric(TwoLayer):
	def __init__(self,params,randomness_key):
		self.activationchoice=1
		super().__init__(params,randomness_key)

	def evaluate(self,X):
		y=0
		for P in itertools.permutations(jnp.identity(len(X))):
			sign=jnp.linalg.det(P)
			PX=jnp.matmul(jnp.array(P),X)	
			y+=sign*self.eval_raw(PX)
		return y

class GenericSymmetric(TwoLayer):
	def __init__(self,params,randomness_key):
		self.activationchoice=0
		super().__init__(params,randomness_key)

	def evaluate(self,X):
		y=0
		for P in itertools.permutations(jnp.identity(len(X))):
			PX=jnp.matmul(jnp.array(P),X)	
			y+=self.eval_raw(PX)
		return y


def test(truth,ansatz,batchsize,randkey):	
	n=ansatz.params['n']
	d=ansatz.params['d']
	randkey,subkey=jax.random.split(randkey)
	X_list=jax.random.uniform(subkey,shape=(batchsize,n,d),minval=-1,maxval=1)
	Y_list=jax.vmap(truth.evaluate)(X_list)
	f_list=jax.vmap(ansatz.evaluate)(X_list)
	return jnp.sum((f_list-Y_list)**2)/batchsize


def learn(truth,ansatz,learning_rate,batchsize,batchnumber,randkey):
	opt=optax.adamw(optax.exponential_decay(init_value=learning_rate,decay_rate=.5,transition_steps=100))
	state=opt.init(ansatz.PARAMS)
	n=ansatz.params['n']
	d=ansatz.params['d']
	
	losses=[]
	for i in range(batchnumber):
		randkey,subkey=jax.random.split(randkey)
		X_list=jax.random.uniform(subkey,shape=(batchsize,n,d),minval=-1,maxval=1)
		Y_list=jax.vmap(truth.evaluate)(X_list)
		loss,grads=jax.value_and_grad(ansatz.sum_loss,0)(ansatz.PARAMS,X_list,Y_list)
		losses.append(loss/batchsize)
		updates,_=opt.update(grads,state,ansatz.PARAMS)
		ansatz.PARAMS=optax.apply_updates(ansatz.PARAMS,updates)
		ansatz.regularize(10)

		randkey,subkey=jax.random.split(randkey)
		print(str(test(truth,ansatz,100,subkey))+'_____'+str(losses[-1])+'_____'+str(i)+'______',end='\r')

	return losses
