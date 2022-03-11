import numpy as np
from mpl_toolkits.mplot3d import Axes3D
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
import util
import bookkeep
import cancellation as canc
import DPP
import opt
	



key=jax.random.PRNGKey(0)
key,*subkeys=jax.random.split(key,1000)





def duality(key,dist_W,dist_X,instances,samples):
	
	key,*subkeys=jax.random.split(key,4)
	W=dist_W(subkeys[0],instances)
	X=dist_X(subkeys[1],samples)

	F=lambda X:canc.ReLU(jnp.matmul(jax.lax.collapse(W,1,3),jax.lax.collapse(X,1,3).T))
	G=canc.antisymmetrize(F)

	return W,X,G(X)


def lipschitz(f,Xdist,samples,eps,key):

	key,*subkeys=jax.random.split(key,4)
	X0=Xdist(subkeys[0],samples)
	s=X0.shape
	dXdist=canc.spherical(s[-2],s[-1],radius=eps)

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
	dW_=util.mindist(W)
	dX_=util.mindist(X)	
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


def E_inv_vs_var_(W,X,draw=True,transform=lambda x:x,msg=''):	
	dW=1/util.Coulomb(transform(W))
	Y=canc.apply_alpha(W,X)
	variances=jnp.var(Y,axis=1)
	std_devs=jnp.sqrt(variances)
	if(draw):
		plt.figure()
		plt.xlim(0,jnp.max(dW))
		plt.ylim(0,jnp.max(std_devs))
		plt.scatter(dW,std_devs,s=2)
		plt.savefig('plots/d_vs_dev'+msg+'.pdf')
		plt.show()
	a=jnp.dot(dW,std_devs)/jnp.dot(dW,dW)
	return a

def d_vs_var_(W,X,draw=True,transform=lambda x:x,msg=''):	
	dW=util.mindist(transform(W))
	Y=canc.apply_alpha(W,X)
	variances=jnp.var(Y,axis=1)
	std_devs=jnp.sqrt(variances)
	if(draw):
		plt.figure()
		plt.xlim(0,jnp.max(dW))
		plt.ylim(0,jnp.max(std_devs))
		plt.scatter(dW,std_devs,s=2)
		plt.savefig('plots/d_vs_dev'+msg+'.pdf')
		plt.show()
	a=jnp.dot(dW,std_devs)/jnp.dot(dW,dW)
	return a


def d_vs_var(key,instances,samples,params={'n':6,'d':3},draw=True):
	n,d=params['n'],params['d']	
	Gaussian=canc.Gaussian(n,d)
	sphere=canc.spherical(n,d)

	W,X,Y=duality(key,Gaussian,Gaussian,instances,samples)
	#W,X,Y=duality(key,sphere,sphere,instances,samples)
	dW=util.mindist(W)

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


def d_vs_var_3d(W,X,transform1,transform2,msg=''):
	
	dW1=util.mindist(transform1(W))
	dW2=util.mindist(transform2(W))
	Y=canc.apply_alpha(W,X)
	variances=jnp.var(Y,axis=1)
	std_devs=jnp.sqrt(variances)

	fig=plt.figure()
	ax=fig.add_subplot(111,projection='3d')
	ax.scatter(dW1,dW2,std_devs,s=2)
	#fig.savefig('plots/d_vs_dev_3d'+msg+'.pdf')
	plt.show()





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
			dist=util.mindist(W)
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
		bookkeep.savedata(variances,'variances')	

		plt.figure()
		plt.yscale('log')
		plt.plot(range(2,n+1),jnp.array([math.factorial(i) for i in range(2,n+1)]),color='b')
		for d in ds:
			plt.plot(range(2,n+1),jnp.array(variances[d]),color='r')
			plt.scatter(range(2,n+1),jnp.array(variances[d]),color='r')
		plt.savefig('plots/vars'+str(n)+'.pdf')




def test_lipschitz():

	variances=[]

	for n in range(2,20):

		paramstring=''

		params={'d':3,'n':n,'instances':1000}
		X_distribution=canc.Gaussian(params['n'],params['d'])

		simple=canc.Simple(params,subkeys[2*n])

		f=simple.evaluate
		g=canc.antisymmetrize(f)

		print(lipschitz(f,X_distribution,1000,.01,subkeys[2*n+1]))


		


def test(gen_W,key):

	key,*subkeys=jax.random.split(key,1000)

	instances=1000
	samples=1000
	d=3
	variances={}
	n_range=range(1,10)

	W_={n:gen_W(subkeys[2*n],shape=(instances,n,d)) for n in n_range}
	X_={n:jax.random.normal(subkeys[2*n+1],shape=(samples,n,d)) for n in n_range}

	vars_nonsymmetrized=[jnp.var(canc.apply_tau(W_[n],X_[n])) for n in n_range]

	plt.figure()
	plt.ylim(bottom=0)
	plt.plot(n_range,vars_nonsymmetrized,color='r')


	for n in range(2,10):

		W=W_[n]
		X=X_[n]
		Y=canc.apply_alpha(W,X)
		variances[n]=jnp.var(Y)
		range_vars=jnp.array([[k,v] for k,v in variances.items()]).T
		_range,_vars=range_vars[0],range_vars[1]
		print(_range)
		print(_vars)

		plt.figure()
		plt.yscale('log')
		plt.plot(_range,[math.factorial(int(k)) for k in _range],color='b')
		plt.plot(_range,_vars,color='r')
		plt.scatter(_range,_vars,color='r')
		plt.savefig('plots/test_det_'+str(n)+'.pdf')





def sample_W_DPP(key,rho,shape,steps=1000):

	(instances,n,d)=shape
	walkers=instances

	sampler=DPP.Sampler(rho,n,d,walkers,key,1)

	W=sampler.sample(1,steps)
	return W[0]


#def W_DPP_disttribution(n,d):
#	rho=DPP.Gaussian_kernel_DPP_density(1)
#	sampler = lambda key,instances : sample_W_DPP(key,rho,shape=(instances,n,d))
#	return sampler
#
#def test_DPP(key):
#
#	key,*subkeys=jax.random.split(key,1000)
#
#	instances=1000
#	steps=250
#	samples=1000
#	variances={}
#	#d=3
#	d=2
#
#	n_range=range(1,10)
#
#	#rho=DPP.Gaussian_kernel_DPP_density(1)
#	rho=DPP.circular_harmonics_DPP_density()
#	W_={n:sample_W_DPP(subkeys[2*n],rho,shape=(instances,n,d),steps=steps) for n in n_range}
#	X_={n:jax.random.normal(subkeys[2*n+1],shape=(samples,n,d)) for n in n_range}
#
#	vars_nonsymmetrized=[jnp.var(canc.apply_tau(W_[n],X_[n])) for n in n_range]
#	#var_W=[jnp.average(jax.vmap(jnp.sum)(jnp.square(W_[n]))) for n in n_range]
#
#	plt.figure()
#	plt.ylim(bottom=0)
#	plt.plot(n_range,vars_nonsymmetrized,color='r')
#	#plt.plot(n_range,var_W,color='r')
#	#plt.show()
#	
#
#	for n in range(2,10):
#
#		W=W_[n]
#		X=X_[n]
#		Y=canc.apply_alpha(W,X)
#		variances[n]=jnp.var(Y)
#		range_vars=jnp.array([[k,v] for k,v in variances.items()]).T
#		_range,_vars=range_vars[0],range_vars[1]
#		print(_range)
#		print(_vars)
#
#		plt.figure()
#		plt.yscale('log')
#		plt.plot(_range,[math.factorial(int(k)) for k in _range],color='b')
#		plt.plot(_range,_vars,color='r')
#		plt.scatter(_range,_vars,color='r')
#		plt.savefig('plots/test_det_'+str(n)+'.pdf')
#
