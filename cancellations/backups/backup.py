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
	



key=jax.random.PRNGKey(0)
key,*subkeys=jax.random.split(key,1000)


def estimate_var(f,X_distribution,n_samples,key):

	X=X_distribution(key,n_samples)

	Y=jax.vmap(f)(X)
	variance=jnp.var(Y)

	validate(Y)

	return variance,Y

def validate(Y):
	Xs=Y.shape[0]
	fs=Y.shape[1]
	T=jnp.take(Y,np.array([0,Xs//2]),axis=0)
	B=jnp.take(Y,np.array([Xs//2,Xs]),axis=0)
	L=jnp.take(Y,np.array([0,fs//2]),axis=1)
	R=jnp.take(Y,np.array([fs//2,fs]),axis=1)

	vars_by_f=jnp.var(Y,axis=0)
	vars_by_x=jnp.var(Y,axis=1)
	print(jnp.var(vars_by_f))
	print(jnp.var(vars_by_x))
	print('')

	print(jnp.var(T))
	print(jnp.var(B))
	print(jnp.var(L))
	print(jnp.var(R))
	print('')

	print('var of first/last Xs log-ratio '+str(jnp.log(jnp.var(T)/jnp.var(B))))
	print('var of first/last fs log-ratio '+str(jnp.log(jnp.var(L)/jnp.var(R))))
	print('\n')
	





def duality(key,dist_W,dist_X,instances,samples):
	
	key,*subkeys=jax.random.split(key,4)
	W=dist_W(subkeys[0],instances)
	X=dist_X(subkeys[1],samples)

	F=lambda X:canc.ReLU(jnp.matmul(jax.lax.collapse(W,1,3),jax.lax.collapse(X,1,3).T))
	G=canc.antisymmetrize(F)

	return W,X,G(X)


def lipschitz(f,Xdist,samples,eps,key):

	key,*subkeys=jax.random.split(key,4)
	s=X.shape
	dXdist=canc.sphere(s[-2],s[-1],radius=eps)

	X0=Xdist(subkeys[0],samples)
	dX=dXdist(subkeys[1],samples)
	X1=X0+dX

	dY=f(X1)-f(X0)
	return jnp.max(jnp.abs(dY)/eps)
	
	

	









def plot_duality(key,instances,samples,by='X',bw_=.1,figname='dual.pdf'):

	params={'n':6,'d':3}
	Gaussian=canc.Gaussian(params['n'],params['d'])
	sphere=canc.spherical(params['n'],params['d'])

	W,X,Y=duality(key,Gaussian,Gaussian,instances,samples)
	#W,X,Y=duality(key,sphere,sphere,instances,samples)
	dW_=full.mindist(W)
	dX_=full.mindist(X)	
	dW=jnp.repeat(jnp.expand_dims(dW_,axis=1),samples,axis=1)
	dX=jnp.repeat(jnp.expand_dims(dX_,axis=0),instances,axis=0)

	dWdX=jnp.ravel(jnp.multiply(dW,dX))
	absY=jnp.ravel(jnp.abs(Y))

	print(jnp.corrcoef(dWdX,absY))

	plt.figure()
	plt.xlim(left=0)
	plt.ylim(0,jnp.max(absY))
	if by=='X':
		plt.scatter(jnp.ravel(dX),absY,s=1)
	elif by=='W':
		plt.scatter(jnp.ravel(dW),absY,s=1)
	else:
		sns.kdeplot(dWdX,absY,color='r',bw=bw_)
		plt.scatter(dWdX,absY,s=2)

	plt.savefig('plots/'+figname)
	plt.show()


def d_vs_var(key,instances,samples,params={'n':6,'d':3},draw=True):

	n,d=params['n'],params['d']	
	Gaussian=canc.Gaussian(n,d)
	sphere=canc.spherical(n,d)

	W,X,Y=duality(key,Gaussian,Gaussian,instances,samples)
	#W,X,Y=duality(key,sphere,sphere,instances,samples)
	dW=full.mindist(W)

	variances=jnp.var(Y,axis=1)
	std_devs=jnp.sqrt(variances)

	#print(jnp.corrcoef(dW,std_devs))
	a=jnp.dot(dW,std_devs)/jnp.dot(dW,dW)
	
	if(draw):
		plt.figure()
		plt.xlim(0,jnp.max(dW))
		plt.ylim(0,jnp.max(std_devs))
		plt.scatter(dW,std_devs,s=2)
		plt.savefig('plots/d_vs_dev_n='+str(n)+'_d='+str(d)+'.pdf')

	return a

def plotphi():
	slopes=[]
	n_max=8

	for d in range(2,7):
		slopes_d=[]
		for n in range(2,n_max):
			key=subkeys[10*n+d]
			a=d_vs_var(key,1000,1000,params={'n':n,'d':d},draw=False)	
			slopes_d.append(a)
		slopes.append(slopes_d)

	print(jnp.array(slopes))

	plt.figure()
	plt.yscale('log')
	for d in range(5):
		plt.plot(range(2,n_max),slopes[d],color='b')
		plt.scatter(range(2,n_max),slopes[d],color='b')
	plt.savefig('plots/slopes.pdf')
			

def plot_dsquare():
	dsquares=[]
	n_max=25

	for d in [2,3,4]:
		print('d '+str(d))
		dsquares_d=[]
		for n in range(2,n_max):
			print('n '+str(n))
			key=subkeys[10*n+d]
			Gaussian=canc.Gaussian(n,d)
			W=Gaussian(key,10000)/jnp.sqrt(n*d)
			dist=full.mindist(W)
			dsquare=jnp.average(jnp.square(dist))
			dsquares_d.append(dsquare)
		dsquares.append(dsquares_d)

	print(jnp.array(dsquares))

	plt.figure()
	plt.yscale('log')
	for d in range(len(dsquares)):
		plt.plot(range(2,n_max),dsquares[d],color='b')
		#plt.scatter(range(2,n_max),dsquares[d],color='b')
	plt.savefig('plots/dsquares.pdf')
	

def plots():

	instances=1000
	samples=1000
	ds={2,3,4,5,6}
	variances={d:[] for d in ds}

	for n in range(2,20):
		for d in ds:
			Gaussian=canc.Gaussian(n,d)
			W_distribution=lambda key,i:Gaussian(key,i)*jnp.sqrt(1/(d*n))
			X_distribution=Gaussian

			W,X,Y=duality(subkeys[n],W_distribution,X_distribution,instances,samples)
			#validate(Y)

			variances[d].append(jnp.var(Y))
			print('at n='+str(n)+', d='+str(d)+', var='+str(jnp.var(Y)))
		print('')
		savedata(variances,'variances')	

		plt.figure()
		plt.yscale('log')
		plt.plot(range(2,n+1),jnp.array([math.factorial(i) for i in range(2,n+1)]),color='b')
		for d in ds:
			plt.plot(range(2,n+1),jnp.array(variances[d]),color='r')
			plt.scatter(range(2,n+1),jnp.array(variances[d]),color='r')
		plt.savefig('plots/vars'+str(n)+'.pdf')




def plots_():

	variances=[]

	for n in range(2,20):

		paramstring=''

		params={'d':3,'n':n,'instances':1000}
		X_distribution=canc.Gaussian(params['n'],params['d'])

		simple=canc.Simple(params,subkeys[2*n])

		f=simple.evaluate
		g=canc.antisymmetrize(f)

		var,_=estimate_var(g,X_distribution,1000,subkeys[2*n+1])

		variances.append(var)
		savedata(variances,'variances'+paramstring)	

		plt.figure()
		plt.plot(range(2,n+1),jnp.log(jnp.array(variances)),color='r')
		plt.plot(range(2,n+1),jnp.log(jnp.array([math.factorial(i) for i in range(2,n+1)])),color='b')
		plt.savefig('plots/vars'+paramstring+str(n)+'.pdf')


def savedata(thedata,filename):
        filename='data/'+filename
        with open(filename,'wb') as file:
                pickle.dump(thedata,file)

#plot_dsquare()
#plotphi()
#plot_duality(key,250,100,by='W',bw_=.1,figname='by_w.pdf')	
#d_vs_var(key,10000,100)	
#plots()


