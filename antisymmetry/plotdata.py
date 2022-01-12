import antisymmetry.learning as learning
import math
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
import copy
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
import jax.numpy as jnp
import jax
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import itertools
import sys
			
		
class Plots:
	def __init__(self,filename):
		with open(filename,"rb") as file:
			alldata=pickle.load(file)
			
			self.truth=alldata["true_f"]
			self.ansatz=alldata["Ansatz"]
			self.params=alldata["params"]

			self.n=self.params['n']
			self.d=self.params['d']




##################################################################################################################################################################

	def segment(self,axes,randkey):

		length=10
		x2=jnp.concatenate([jnp.array([length/2]),jnp.zeros(self.d-1)],axis=0)
		x1=-x2
		X_rest=jax.random.uniform(randkey,shape=(self.n-1,self.d),minval=-1,maxval=1)
		X1=jnp.concatenate([jnp.expand_dims(x1,axis=0),X_rest],axis=0)
		X2=jnp.concatenate([jnp.expand_dims(x2,axis=0),X_rest],axis=0)
			
		I=jnp.arange(0,1,.01)
		Xlist=jnp.array([X2*t+X1*(1-t) for t in I])
		ylist=jax.vmap(self.truth.evaluate)(Xlist)
		flist=jax.vmap(self.ansatz.evaluate)(Xlist)
		ymax=jnp.sqrt(jnp.max(jnp.square(ylist)))

		axes[0].plot(I,ylist,label="true f",color='b')
		axes[0].plot(I,flist,label="Ansatz",color='r')
		axes[0].set_ylim(-ymax,ymax)	
		axes[0].set_xticklabels([])

	
	def levelsets(self,ax,f,X,vecs,**kwargs):

		f_square_function=lambda s,t : f( X + s*vecs[0] + t*vecs[1] )
		f_square_vecinput=lambda t : f_square_function(t[0],t[1])

		S,T=jnp.meshgrid(jnp.arange(0,1,.02),jnp.arange(0,1,.02))
		square=jnp.array([S,T])

		f_square=jax.vmap(jax.vmap(f_square_vecinput,1),2)(square)

		
		ax.imshow(f_square,cmap=plt.get_cmap('PiYG'),interpolation='bilinear')
		ax.contour(f_square,colors='k',**kwargs)
		ax.set_box_aspect(1)

	def nodalsurface(self,axes,randkey):

		axes[0].set_title('Ansatz')
		axes[1].set_title('true f')

		r=5
		x=jnp.array([-r,-r])
		if self.d>2:
			x=jnp.concatenate([x,jnp.zeros(self.d-2)],axis=0)
		X_rest=jax.random.uniform(randkey,shape=(self.n-1,self.d),minval=-1,maxval=1)
		X=jnp.concatenate([jnp.expand_dims(x,axis=0),X_rest],axis=0)
		
		vecs=[np.zeros([self.n,self.d]),np.zeros([self.n,self.d])]
		vecs[0][0][0]=2*r
		vecs[1][0][1]=2*r

		self.levelsets(axes[0],self.ansatz.evaluate,X,vecs)
		self.levelsets(axes[1],self.truth.evaluate,X,vecs)

	def comparelevelsets(self,axes,randkey):

		axes[0].set_title('Ansatz')
		axes[1].set_title('true f')

		randkey,*subkeys=jax.random.split(randkey,5)
		vecs=[jax.random.normal(subkeys[i],shape=(self.n,self.d)) for i in [0,1]]
		X=-(vecs[0]+vecs[1])/2
		self.levelsets(axes[0],self.ansatz.evaluate,X,vecs)
		self.levelsets(axes[1],self.truth.evaluate,X,vecs)



	def showsymmetry(self,axes,randkey):

		axes[0].set_title('Ansatz')
		axes[1].set_title('true f')
		
		randkey,*subkeys=jax.random.split(randkey,4)
		X_=jax.random.normal(subkeys[0],shape=(self.n-2,self.d))/2
		x1=jax.random.normal(subkeys[1],shape=(self.d,))/2
		x2=jax.random.normal(subkeys[2],shape=(self.d,))/2

		X1=jnp.row_stack([x2,x1,X_])
		X2=jnp.row_stack([x1,x2,X_])
		O=jnp.zeros([self.n,self.d])

		self.levelsets(axes[0],self.ansatz.evaluate,O,[X1,X2])
		self.levelsets(axes[1],self.truth.evaluate,O,[X1,X2])


##################################################################################################################################################################


	def plotgrid(self,titlebar,plotmethod,randkey,a=1,b=2,A=3,B=3,savename="plot"):	

		fig=plt.figure(titlebar+', d='+str(self.d)+', n='+str(self.n))
		randkey,*subkeys=jax.random.split(randkey,A*B+2)

		gs=GridSpec(A,B,hspace=.3)

		for i in range(A*B):
			axes_=gridspec.GridSpecFromSubplotSpec(a,b,subplot_spec=gs[i])
			axes=[]
			for j in range(a*b):
				ax=plt.Subplot(fig,axes_[j])
				fig.add_subplot(ax)
				axes.append(ax)			
			plotmethod(axes,subkeys[i])
		
		self.saveplot(savename)




	def saveplot(self,string):
		paramtext=""
		for x,y in self.params.items():
			paramtext+="_"+x+"="+str(y)
		plt.savefig("theplots/"+string+paramtext+".pdf")

	

	def allplots(self):

#		space=' '*25
#		line='-'*20
#		vlines=('\n'+space+'     |'+space+'         |')*15+'\n'
#		square4pt=space+'  f(X)-----------'+line+'f(X+v2)'+vlines+space+'  f(X+v1)-----------'+line+'f(X+v1+v2)\n'
#		print('\nDescription of 4-point level set plot: Corners correspond to\n\n'+square4pt)
#
#		square=space+'f(0,0,x3..)--'+line+'f(x1,x2,x3..)'+vlines+space+'f(x2,x1,x3..)'+line+'--f(x1+x2,x1+x2,x3..)\n'
#		print('\nDescription of '+self.antistring+'symmetry plot: Corners correspond to\n\n'+square)


		randkey=jax.random.PRNGKey(1)

		randkey,subkey=jax.random.split(randkey)
		self.plotgrid('nodal surface',self.nodalsurface,subkey,A=1,B=1,savename='nodalsurface')

#		randkey,subkey=jax.random.split(randkey)
#		self.plotgrid('4-point level sets',self.comparelevelsets,subkey,A=1,B=1,savename='levelsets')
#
#		randkey,subkey=jax.random.split(randkey)
#		self.plotgrid('symmetry plots',self.showsymmetry,subkey,A=1,B=1,savename=self.antistring+'symmetry')
#
		randkey,subkey=jax.random.split(randkey)
		self.plotgrid('segments',self.segment,subkey,b=1,A=2,B=3,savename='segments')

		plt.show()
	


if __name__=='__main__':

	args=sys.argv[1:]
	if len(args)==0:
		filename='most_recent'
	else:
		filename=args[0]

	plots=Plots("data/"+filename)
	plots.allplots()
	


