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
FN_activation=jnp.tanh


############## symmetric Ansatz ##########################

activation0s=ReLU_leaky
activation1s=ReLU_leaky

############## localization ########################

box=lambda X:jnp.product(jnp.product(jnp.heaviside(1-jnp.square(X),0),axis=-1),axis=-1)

envelope=lambda X:jnp.exp(-jnp.sum(jnp.square(X)))
envelope_FN=envelope

#envelope=box
#envelope_FN=box


class Ansatz:

	def __init__(self):
		self.velocity={key:[0*m for m in M] for key,M in self.PARAMS.items()}
		self.scaling=1


	def evaluate(self,X):	
		return self.evaluate_(X,self.PARAMS)


	def regularize(self,r):
		for key,M in self.PARAMS.items():

			if isinstance(self.PARAMS[key],list):
				for i in range(len(self.PARAMS[key])):
					self.PARAMS[key][i]=jnp.tanh(self.PARAMS[key][i]/r)*r 
			else:
				self.PARAMS[key]=jnp.tanh(self.PARAMS[key]/r)*r 
				

	def avg_loss(self,PARAMS,X_list,y_list):
		f_list=jax.vmap(self.evaluate_,[0,None])(X_list,PARAMS) #X configurations in parallel
		fy_list=jnp.array([f_list,y_list])
		return jnp.average(jax.vmap(loss,1)(fy_list))

	def normalize(self,X_distribution):# ensure true function has variance 1
		samples=250
		key=jax.random.PRNGKey(123)
		
		X_list=X_distribution(key,samples)
		y_list=jax.vmap(self.evaluate)(X_list)
		self.scaling/=jnp.sqrt(jnp.var(y_list))


class Antisatz(Ansatz):

	def __init__(self,params,randomness_key):
	
		self.params=params
		d,n,p,m=params['d'],params['n'],params['p'],params['m']
		key,*subkeys=jax.random.split(randomness_key,7)

		V=jax.random.normal(subkeys[2],shape=(p,n,d))
		b=jax.random.normal(subkeys[3],shape=(p,n))
		W=jax.random.normal(subkeys[4],shape=(m,p))
		a=jax.random.normal(subkeys[5],shape=(m,))

		self.PARAMS={'V':V,'b':b,'W':W,'a':a}
		super().__init__()


	def evaluate_(self,X,PARAMS):
		V,b,W,a=(PARAMS[key] for key in ['V','b','W','a'])
		n=self.params['n']
		square_matrices_list=activation1a(jnp.dot(V,X.T)+jnp.repeat(jnp.expand_dims(b,2),n,axis=2))
		determinants_list=jax.vmap(jnp.linalg.det)(square_matrices_list)
		layer2=odd_activation(jnp.dot(W,determinants_list))
		return self.scaling*envelope(X)*jnp.dot(a,layer2)


#class FermiNet(Ansatz):
#
#	def __init__(self,params,randomness_key):
#		self.params=params
#		d,internal_layer_width,L,n,ndets=params['d'],params['internal_layer_width'],params['layers'],params['n'],params['ndets']
#		key,*subkeys=jax.random.split(randomness_key,3*L+10)
#
#		layer_width_list=[d]+(L-1)*[internal_layer_width]
#		self.layer_width_list=layer_width_list
#		W_list=[];V_list=[];b_list=[]
#		for l in range(1,L):
#			W=jax.random.normal(subkeys[3*l-3],shape=(layer_width_list[l],layer_width_list[l-1]))
#			V=jax.random.normal(subkeys[3*l-2],shape=(layer_width_list[l],layer_width_list[l-1]))
#			b=jax.random.normal(subkeys[3*l-1],shape=(layer_width_list[l],1))
#			W_list.append(W); V_list.append(V); b_list.append(b)
#
#		W_fi=jax.random.normal(subkeys[-2],shape=(ndets,n,sum(layer_width_list)))
#		b_fi=jax.random.normal(subkeys[-1],shape=(ndets,n,1))
#
#		self.PARAMS={'W_list':W_list,'V_list':V_list,'b_list':b_list,'W_fi':W_fi,'b_fi':b_fi}
#		super().__init__()
#
#
#	def evaluate_(self,X,PARAMS):
#		n=self.params['n']
#		multiplier=self.scaling*envelope_FN(X)
#
#		X=X.T
#		skips=[X]
#		for l in range(self.params['layers']-1):
#			W,V,b=(PARAMS[key][l] for key in ['W_list','V_list','b_list'])
#			S=jnp.expand_dims(jnp.average(X,axis=-1),-1)
#			Y=FN_activation(jnp.dot(W,X)+jnp.repeat(jnp.dot(V,S)+b,n,axis=-1))
#			if(self.layer_width_list[l]==self.layer_width_list[l+1]):
#				Y+=X
#			X=Y
#			skips.append(Y)
#
#		history=jnp.concatenate(skips,axis=-2)
#
#		Phi=FN_activation(jnp.tensordot(self.PARAMS['W_fi'],history,axes=1)+jnp.repeat(self.PARAMS['b_fi'],n,axis=-1))
#
#		return multiplier*jnp.sum(jax.vmap(jnp.linalg.det)(Phi))



class FermiNet(Ansatz):

	def __init__(self,params,randomness_key):
		self.params=params
		d,internal_layer_width,L,n,ndets=params['d'],params['internal_layer_width'],params['layers'],params['n'],params['ndets']
		key,*subkeys=jax.random.split(randomness_key,3*L+10)

		layer_width_list=[d]+(L-1)*[internal_layer_width]
		self.layer_width_list=layer_width_list
		W_list=[];V_list=[];b_list=[]
		for l in range(1,L):
			W=jax.random.normal(subkeys[3*l-3],shape=(layer_width_list[l],layer_width_list[l-1]))
			V=jax.random.normal(subkeys[3*l-2],shape=(layer_width_list[l],layer_width_list[l-1]))
			b=jax.random.normal(subkeys[3*l-1],shape=(layer_width_list[l],1))
			W_list.append(W); V_list.append(V); b_list.append(b)

		W_fi=jax.random.normal(subkeys[-2],shape=(ndets,n,layer_width_list[-1]))
		b_fi=jax.random.normal(subkeys[-1],shape=(ndets,n,1))

		self.PARAMS={'W_list':W_list,'V_list':V_list,'b_list':b_list,'W_fi':W_fi,'b_fi':b_fi}
		super().__init__()


	def evaluate_(self,X,PARAMS):
		n=self.params['n']
		multiplier=self.scaling*envelope_FN(X)

		X=X.T
		for l in range(self.params['layers']-1):
			W,V,b=(PARAMS[key][l] for key in ['W_list','V_list','b_list'])
			S=jnp.expand_dims(jnp.average(X,axis=-1),-1)
			Y=FN_activation(jnp.dot(W,X)+jnp.repeat(jnp.dot(V,S)+b,n,axis=-1))
			if(self.layer_width_list[l]==self.layer_width_list[l+1]):
				Y+=X
			X=Y

		Phi=FN_activation(jnp.tensordot(self.PARAMS['W_fi'],Y,axes=1)+jnp.repeat(self.PARAMS['b_fi'],n,axis=-1))

		return multiplier*jnp.sum(jax.vmap(jnp.linalg.det)(Phi))




class SymAnsatz(Ansatz):

	def __init__(self,params,randomness_key):
	
		self.params=params
		d,n,p,m=params['d'],params['n'],params['p'],params['m']
		key,*subkeys=jax.random.split(randomness_key,7)

		V=jax.random.normal(subkeys[2],shape=(p,d))
		c=jax.random.normal(subkeys[1],shape=(p,))
		W=jax.random.normal(subkeys[4],shape=(m,p))
		b=jax.random.normal(subkeys[4],shape=(m,))
		a=jax.random.normal(subkeys[5],shape=(m,))

		self.PARAMS={'V':V,'c':c,'W':W,'b':b,'a':a}
		super().__init__()


	def evaluate_(self,X,PARAMS):
		V,c,W,b,a=(PARAMS[key] for key in ['V','c','W','b','a'])
		n=self.params['n']
		S=jnp.sum(activation0s(jnp.dot(V,X.T)+jnp.repeat(jnp.expand_dims(c,1),n,axis=1)),axis=1)
		layer2=activation1s(jnp.dot(W,S)+b)
		return envelope(X)*jnp.dot(a,layer2)



class TwoLayer:
	def __init__(self,params,randomness_key):
		
		self.params=params
		d,n,m=params['d'],params['n'],params['m']
		key,*subkeys=jax.random.split(randomness_key,4)

		self.W=jax.random.normal(subkeys[0],shape=(m,n*d))
		self.b=jax.random.normal(subkeys[1],shape=(m,))
		self.a=jax.random.normal(subkeys[2],shape=(m,))
		self.scale=1

	def eval_raw(self,X):
		X_vec=jnp.ravel(X)
		layer1=activations_truth[self.activationchoice](jnp.dot(self.W,X_vec)+self.b)
		return jnp.dot(self.a,layer1)

	def evaluate(self,X):
		return self.scale*envelope(X)*jnp.tanh(self.eval_non_regularized(X))
		
	def normalize(self,X_distribution):# ensure true function has variance 1
		samples=250
		key=jax.random.PRNGKey(123)
		
		X_list=X_distribution(key,samples)
		y_list=jax.vmap(self.eval_non_regularized)(X_list)
		self.a/=jnp.sqrt(jnp.var(y_list))

		y_list=jax.vmap(self.evaluate)(X_list)
		self.scale/=jnp.sqrt(jnp.var(y_list))

	
	
class GenericAntiSymmetric(TwoLayer):
	def __init__(self,params,randomness_key):
		self.activationchoice=1
		super().__init__(params,randomness_key)

	def eval_non_regularized(self,X):
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

	def eval_non_regularized(self,X):
		y=0
		for P in itertools.permutations(jnp.identity(len(X))):
			PX=jnp.matmul(jnp.array(P),X)	
			y+=self.eval_raw(PX)
		return y

	

def apply_updates(PARAMS,grads,rate):
	NEWPARAMS={}
	for key,param in PARAMS.items():
		NEWPARAMS[key]=param-rate*grads[key]
	return NEWPARAMS


def learn(truth,ansatz,batchsize,batchnumber,randkey,X_distribution,optimizer=optax.rmsprop(.01)):
	#opt=optax.adamw(optax.exponential_decay(init_value=learning_rate,decay_rate=.9,transition_steps=10))
	#opt=optax.rmsprop(learning_rate)
	state=optimizer.init(ansatz.PARAMS)
	n=ansatz.params['n']
	d=ansatz.params['d']
	
	losses=[]
	for i in range(batchnumber):
		randkey,subkey=jax.random.split(randkey)
		X_list=X_distribution(subkey,batchsize)
		Y_list=jax.vmap(truth.evaluate)(X_list)

		loss,grads=jax.value_and_grad(ansatz.avg_loss,0)(ansatz.PARAMS,X_list,Y_list)
		losses.append(loss)
		loss_estimate=loss if i==0 else .1*loss+.9*loss_estimate

		updates,_=optimizer.update(grads,state,ansatz.PARAMS)
		ansatz.PARAMS=optax.apply_updates(ansatz.PARAMS,updates)
		#ansatz.PARAMS=apply_updates(ansatz.PARAMS,grads,.005)
		ansatz.regularize(10)

		randkey,subkey=jax.random.split(randkey)
		barlength=100;
		roundloss=round(loss_estimate*1000)/1000
		print((7-len(str(i)))*' '+str(i)+' batches done. Loss: ['+(round(barlength*min(loss_estimate,1)))*'\u2588'+(barlength-round(barlength*loss_estimate))*'_'+'] '+str(roundloss),end='\r')

	return losses

