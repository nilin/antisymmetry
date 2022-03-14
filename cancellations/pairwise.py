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
import bookkeep
import copy
import jax
import jax.numpy as jnp
import optax
import util
import spherical
import cancellation as canc
import antisymmetry.mcmc as mcmc
import DPP
import opt
	







key=jax.random.PRNGKey(0)
key,*keys=jax.random.split(key,1000)



#opt.gen_W(keys[0],(instances,n,d))



energies=lambda W:util.Coulomb(W)+jnp.sum(jnp.square(W),axis=(-2,-1))
energy=lambda W:jnp.sum(energies(W))



def dT_test(W,X,apply_tau):

	ijs=util.argmindist(W)
	Ws=util.transpositions(W,ijs)

	Y=apply_tau(W,X)
	Ys=apply_tau(Ws,X)
	dY=Y-Ys

	return jnp.sqrt(jnp.average(jnp.square(dY),axis=-1))

	
def dT_vs_AT(W,X,apply_tau,apply_alpha):
	dT_by_w=dT_test(W,X,apply_tau)
	AT_by_w=jnp.sqrt(jnp.average(jnp.square(apply_alpha(W,X)),axis=-1))
	plt.scatter(dT_by_w,AT_by_w,s=.2)


def test(apply_tau=canc.apply_tau,nmax=0,WXname='trivial d=3'):
	L2={}
	W_,X_,instances,samples,n_range,d=bookkeep.getdata(WXname)

	if nmax==0:
		nmax=n_range[-1]

	for n in range(2,nmax):

		X=X_[n]
		W=W_[n]

		norms=dT_test(W,X,apply_tau)
		L2[n]=jnp.sqrt(jnp.average(jnp.square(norms)))

		fn='pairwise '+WXname+' n='+str(n)
		bookkeep.savedata(L2,fn)
		bookkeep.saveplot([fn],fn,colors=['r'])

		print(n)


def plot_dT_AT():
	WXname='trivial d=3'
	W_,X_,instances,samples,n_range,d=bookkeep.getdata(WXname)
	activation=util.ReLU
	plt.figure()
	for n in range(5,8):
		x=jnp.sort(util.sample_mu(n*d,5000,keys[n]))
		p=util.bestpolyfunctionfit(activation,n-2,x)
		remainder=lambda x:activation(x)-p(x)
		#plt.plot(x,remainder(x))
		apply_remainder=lambda W,X:canc.apply_tau_(W,X,remainder)
		apply_alpha=lambda W,X:canc.apply_alpha(W,X,activation)
		dT_vs_AT(W_[n],X_[n],apply_remainder,apply_alpha)
	plt.savefig('dT_AT')
	plt.show()


"""
def plot_remainder(activation,name=''):
	d=3
	X=[]
	P=[]
	R=[]
	for n in range(2,10):
		x=jnp.sort(util.sample_mu(n*d,5000,keys[n]))
		X.append(x)
		p=util.bestpolyfunctionfit(activation,n-2,x)
		P.append(p(x))
		R.append(activation(x)-P[-1])

	plt.figure()
	for i in range(len(X)):
		plt.plot(X[i],P[i])
	plt.savefig('plots/polyapprox'+name+'.pdf')

	plt.figure()
	for i in range(len(X)):
		plt.plot(X[i],R[i])
	plt.savefig('plots/remainder'+name+'.pdf')
"""
def plot_remainder(activation,name=''):
	d=3
	x=jnp.sort(jax.random.normal(key,shape=(10000,)))
	P=[]
	R=[]
	for n in range(2,10):
		p=util.bestpolyfunctionfit(activation,n-2,x)
		P.append(p(x))
		R.append(activation(x)-P[-1])

	plt.figure()
	for i in range(len(P)):
		plt.plot(x,P[i])
	plt.savefig('plots/polyapprox'+name+'.pdf')

	plt.figure()
	for i in range(len(R)):
		plt.plot(x,R[i])
	plt.savefig('plots/remainder'+name+'.pdf')


def plot_dT_proxy():
	WXname='trivial d=3'
	W_,X_,instances,samples,n_range,d=bookkeep.getdata(WXname)
	activation=util.ReLU
	plt.figure()
	for n in range(5,8):
		W=W_[n]
		X=X_[n]
		f=util.ReLU

		lengths_squared=jnp.sum(jnp.square(W),axis=(-2,-1))
		eps_squared=2*jnp.square(util.mindist(W))

		proxy=jnp.sqrt(util.variations(keys[n],f,lengths_squared,eps_squared))

		apply_tau=lambda W,X:canc.apply_tau(W,X,f)
		dT_by_w=dT_test(W,X,apply_tau)

		plt.scatter(dT_by_w,proxy,s=.3)
	plt.savefig('dT_proxy')	
	plt.show()


def plot_proxy_AT(activation):
	WXname='trivial d=3'
	W_,X_,instances,samples,n_range,d=bookkeep.getdata(WXname)
	plt.figure()
	for n in range(3,8):
		x=jnp.sort(util.sample_mu(n*d,5000,keys[n]))
		p=util.bestpolyfunctionfit(activation,n-2,x)
		f=lambda x:activation(x)-p(x)
		#f=activation

		W=W_[n]
		X=X_[n]
		lengths_squared=jnp.sum(jnp.square(W),axis=(-2,-1))
		eps_squared=2*jnp.square(util.mindist(W))
		proxy=jnp.sqrt(util.variations(keys[n],f,lengths_squared,eps_squared))

		AT=jnp.sqrt(jnp.average(jnp.square(canc.apply_alpha(W,X,activation)),axis=-1))
		plt.scatter(proxy,AT,s=.3)
	plt.savefig('plot_proxy_AT')
	plt.show()


def plot_naive_proxy_AT(activation):
	WXname='trivial d=3'
	W_,X_,instances,samples,n_range,d=bookkeep.getdata(WXname)
	plt.figure()
	for n in range(3,8):
		x=jnp.sort(util.sample_mu(n*d,5000,keys[n]))
		p=util.bestpolyfunctionfit(activation,n-2,x)
		f=lambda x:activation(x)-p(x)

		W=W_[n]
		X=X_[n]
		naive_proxy=util.mindist(W)

		AT=jnp.sqrt(jnp.average(jnp.square(canc.apply_alpha(W,X,activation)),axis=-1))
		plt.scatter(naive_proxy,AT,s=.3)
	plt.savefig('plots/naive_proxy_AT')
	plt.show()

def plot_proxy_avgproxy():
	WXname='trivial d=3'
	W_,X_,instances,samples,n_range,d=bookkeep.getdata(WXname)
	activation=util.ReLU
	plt.figure()
	a=[]
	b=[]
	for n in range(2,8):
		W=W_[n]
		X=X_[n]
		f=util.ReLU

		lengths_squared=jnp.sum(jnp.square(W),axis=(-2,-1))
		eps_squared=2*jnp.square(util.mindist(W))
		avgofproxies=jnp.sqrt(jnp.average(util.variations(keys[n],f,lengths_squared,eps_squared)))
		a.append(avgofproxies)

		r_sq=jnp.atleast_1d(jnp.average(lengths_squared))
		eps_sq=jnp.atleast_1d(jnp.array(jnp.average(eps_squared)))
		avgproxy=jnp.squeeze(jnp.sqrt(util.variations(keys[n],f,r_sq,eps_sq)))
		b.append(avgproxy)

	plt.scatter(a,b)	
	I=[0,max(a)]
	plt.plot(I,I)
	plt.savefig('plots/proxy_avgproxy')
	plt.show()


def plot_avgproxy_avgAT(activation):
	WXname='trivial d=3'
	W_,X_,instances,samples,n_range,d=bookkeep.getdata(WXname)
	plt.figure()
	avgATs=[]
	proxies0=[]
	proxies1=[]
	proxies2=[]
	proxies3=[]
	_range=range(2,8)
	for n in _range:
		x=jnp.sort(util.sample_mu(n*d,5000,keys[n]))
		p=util.bestpolyfunctionfit(activation,n-2,x)
		f=lambda x:activation(x)-p(x)
		proxies0.append(util.L2norm(f(x)))

		W=W_[n]
		X=X_[n]

		lengths_squared=jnp.sum(jnp.square(W),axis=(-2,-1))
		eps_squared=2*jnp.square(util.mindist(W))
		avgofproxies=jnp.sqrt(jnp.average(util.variations(keys[n],f,lengths_squared,eps_squared)/2))
		proxies1.append(avgofproxies)

		badproxies=jnp.sqrt(jnp.average(util.variations(keys[n],activation,lengths_squared,eps_squared)/2))
		proxies2.append(badproxies)

		r_sq=jnp.atleast_1d(jnp.average(lengths_squared))
		eps_sq=jnp.atleast_1d(jnp.array(jnp.average(eps_squared)))
		#proxies2.append(jnp.squeeze(jnp.sqrt(variations(keys[n],f,r_sq,eps_sq))))
		proxies3.append(jnp.sqrt(util.variations(keys[n],f,r_sq,eps_sq)/2))

		avgATs.append(jnp.sqrt(jnp.average(jnp.square(canc.apply_alpha(W,X,activation)))))
		

	plt.plot(_range,proxies0,'r-.')	
	plt.plot(_range,proxies1,'r')	
	plt.plot(_range,proxies2,'k--')	
	#plt.plot(_range,proxies3,'r--')	
	plt.scatter(_range,avgATs)	
	plt.yscale('log')
	plt.savefig('plots/avgproxy_avgAT')
	plt.show()



plot_remainder(util.heaviside,'HS')

		
#x=jnp.sort(util.sample_mu(n*d,5000,keys[n]))
#p=util.bestpolyfunctionfit(activation,n-2,x)
#remainder=lambda x:activation(x)-p(x)
##plt.plot(x,remainder(x))
#apply_remainder=lambda W,X:canc.apply_tau_(W,X,remainder)
#apply_tau=apply_remainder
#dT_by_w=dT_test(W,X,apply_tau)


#plot_dT_AT()
#plot_dT_proxy()
#plot_proxy_avgproxy()
#plot_proxy_AT(util.heaviside)
#plot_naive_proxy_AT(util.heaviside)
#plot_avgproxy_avgAT(util.heaviside)




"""	
def variations(key,f,marginal_var,diff_var,samples=1000):
	key1,key2=jax.random.split(key)
	r=jnp.sqrt(marginal_var-diff_var/4)
	eps=jnp.sqrt(diff_var)
	scales1=jnp.repeat(jnp.expand_dims(r,axis=1),samples,axis=1)
	scales2=jnp.repeat(jnp.expand_dims(eps,axis=1),samples,axis=1)
	instances=r.size
	Z=jnp.multiply(jax.random.normal(key1,shape=(instances,samples)),scales1)
	Z_=jnp.multiply(jax.random.normal(key2,shape=(instances,samples)),scales2)

	Y1=Z-Z_/2
	Y2=Z+Z_/2

	return jnp.average(jnp.square(f(Y2)-f(Y1)),axis=-1)

def variations_(key,f,marginal_var_squared,eps_squared,samples=1000):
	key1,key2=jax.random.split(key)
	r=jnp.sqrt(marginal_var_squared)
	eps=jnp.sqrt(eps_squared)
	scales1=jnp.repeat(jnp.expand_dims(r,axis=1),samples,axis=1)
	scales2=jnp.sqrt(2)*jnp.repeat(jnp.expand_dims(eps,axis=1),samples,axis=1)
	instances=r.size
	Z=jnp.multiply(jax.random.normal(key1,shape=(instances,samples)),scales1)
	Z_=jnp.multiply(jax.random.normal(key2,shape=(instances,samples)),scales2)
	return jnp.average(jnp.square(f(Z)-f(Z+Z_)),axis=-1)
"""
