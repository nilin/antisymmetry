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
		self.a=jax.random.normal(subkeys[1],shape=(m,))*jnp.sqrt(2/m)

	def evaluate(self,X):
		X_vec=jnp.ravel(X)
		layer1=activation(jnp.dot(self.W,X_vec))
		return jnp.dot(self.a,layer1)



class Simple:
	def __init__(self,params,randomness_key,normalize=False):
		self.params=params
		d,n,instances=params['d'],params['n'],params['instances']
		key,*subkeys=jax.random.split(randomness_key,4)

		self.W=jax.random.normal(subkeys[0],shape=(instances,n*d))*jnp.sqrt(2/(d*n))
		self.a=jax.random.normal(subkeys[1],shape=(instances,))*jnp.sqrt(2)
		if(normalize):
			self.W=normalize_rows(self.W)
			self.a=jnp.ones(shape=(instances,))

	def evaluate(self,X):
		X_vec=jnp.ravel(X)
		return jnp.multiply(self.a,activation(jnp.dot(self.W,X_vec)))


def antisymmetrize(f):
	def antisymmetric(X):
		y=jnp.zeros(len(f(X)))
		for P in itertools.permutations(jnp.identity(X.shape[-2])):
			sign=jnp.linalg.det(P)
			PX=jnp.matmul(jnp.array(P),X)	
			y+=sign*f(PX)
		return y
	return antisymmetric
	


def distribution(f,X_distribution,samples=100):
	key=jax.random.PRNGKey(np.random.randint(100))
	
	X_list=X_distribution(key,samples)
	y_list=jax.vmap(f)(X_list)

	return y_list




def normalize(X):
	return lambda x:x/jnp.sqrt(jnp.sum(jnp.square(x)))(X)
def normalize_rows(X):
	return jax.vmap(lambda x:x/jnp.sqrt(jnp.sum(jnp.square(x))))(X)

	


def Gaussian(n,d):
	return (lambda key,samples:jax.random.normal(key,shape=(samples,n,d)))

def spherical(n,d,radius=1): 
	g=Gaussian(n,d)
	return (lambda key,samples:normalize_rows(g(key,samples))*radius)



def plots():

	m_list=[1,10,100]
	antisymvars={m:[] for m in m_list}

	for n in range(2,15):
		plt.figure()


		for m in m_list:
		
			params={'d':3,'n':n,'m':m}
			k=10
			X_distribution=Gaussian(params['n'],params['d'])
			#X_distribution=spherical(params['n'],params['d'],jnp.sqrt(params['n']*params['d']))
			key=jax.random.PRNGKey(n)
			key,*subkeys=jax.random.split(key,k+2)
			antisymvar=[]
			for i in range(k):
				tl=TwoLayer(params,subkeys[i])
				f=tl.evaluate
				g=antisymmetrize(f)

				nsd=distribution(f,X_distribution)
				asd=distribution(g,X_distribution)


				antisymvar.append(jnp.var(asd))
				print('variance of antisymmetrized: '+str(antisymvar[-1]))
				print('')
			
			print('average variance of antisymmetrized: '+str(jnp.average(jnp.array(antisymvar))))
			print(str(n)+100*'_')

			antisymvars[m].append(jnp.average(jnp.array(antisymvar)))
			plt.plot(range(2,n+1),jnp.log(jnp.array(antisymvars[m])),color='r')

		savedata(antisymvars)	

		plt.plot(range(2,n+1),jnp.log(jnp.array([math.factorial(i) for i in range(2,n+1)])),color='b')
		plt.savefig('plots/vars'+str(n)+'.pdf')


def savedata(thedata):
        filename='data/vars'+str(len(thedata[1]))
        with open(filename,'wb') as file:
                pickle.dump(thedata,file)

#plots()

