import learning
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
			
		
class Plots:
	def __init__(self,filename):
		with open(filename,"rb") as file:
			alldata=pickle.load(file)
			
			self.truth=alldata["true_f"]
			self.ansatz=alldata["Ansatz"]
			self.losslist=jnp.array(alldata["losslist"]).T
			self.params=alldata["params"]

			self.n=self.params['n']
			self.d=self.params['d']



##################################################################################################################################################################

	def segment(self,axes,randkey):
		randkey1,randkey2=jax.random.split(randkey)
		
		X1=jax.random.uniform(randkey1,shape=(self.n,self.d),minval=-1,maxval=1)
		X2=jax.random.uniform(randkey2,shape=(self.n,self.d),minval=-1,maxval=1)
			
		I=jnp.arange(0,1,.01)
		Xlist=jnp.array([X2*t+X1*(1-t) for t in I])
		ylist=jax.vmap(self.truth.evaluate)(Xlist)
		flist=jax.vmap(self.ansatz.evaluate)(Xlist)
		
		axes[0].plot(I,ylist,label="true f",color='b')
		axes[0].plot(I,flist,label="Ansatz",color='r')

		axes[0].set_ylim(-2,2)	
		axes[0].set_xticklabels([])

	
	def levelsets(self,ax,f,X,vecs,**kwargs):

		f_square=lambda s,t : f( X + s*vecs[0] + t*vecs[1] )
		f_square_vecinput=lambda t : f_square(t[0],t[1])

		S,T=jnp.meshgrid(jnp.arange(0,1,.02),jnp.arange(0,1,.02))
		square=jnp.array([S,T])

		f_square=jax.vmap(jax.vmap(f_square_vecinput,1),2)(square)
		
		ax.imshow(f_square,cmap=plt.get_cmap('PiYG'),interpolation='bilinear')
		ax.contour(f_square,colors='k',**kwargs)
		ax.set_box_aspect(1)


	def comparelevelsets(self,axes,randkey):

		axes[0].set_title('Ansatz')
		axes[1].set_title('true f')

		randkey,*subkeys=jax.random.split(randkey,5)
		vecs=[jax.random.uniform(subkeys[i],shape=(self.n,self.d),minval=-1,maxval=1) for i in [0,1]]
		X=-(vecs[0]+vecs[1])/2
		self.levelsets(axes[0],self.ansatz.evaluate,X,vecs)
		self.levelsets(axes[1],self.truth.evaluate,X,vecs)



	def showsymmetry(self,axes,randkey):

		axes[0].set_title('Ansatz')
		axes[1].set_title('true f')
		
		randkey,*subkeys=jax.random.split(randkey,4)
		X_=jax.random.uniform(subkeys[0],shape=(self.n-2,self.d),minval=-1,maxval=1)/2
		x1=jax.random.uniform(subkeys[1],shape=(self.d,),minval=-1,maxval=1)/2
		x2=jax.random.uniform(subkeys[2],shape=(self.d,),minval=-1,maxval=1)/2

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

	

def allplots(plots):

	space=' '*25
	line='-'*20
	vlines=('\n'+space+'     |'+space+'         |')*15+'\n'
	square4pt=space+'  f(X)-----------'+line+'f(X+v2)'+vlines+space+'  f(X+v1)-----------'+line+'f(X+v1+v2)\n'
	print('\nDescription of 4-point level set plot: Corners correspond to\n\n'+square4pt)

	square=space+'f(0,0,x3..)=0'+line+'f(x1,x2,x3..)'+vlines+space+'f(x2,x1,x3..)'+line+'--f(x1+x2,x1+x2,x3..)=0\n'
	print('\nDescription of antisymmetry plot: Corners correspond to\n\n'+square)



	randkey=jax.random.PRNGKey(1)

	randkey,subkey=jax.random.split(randkey)
	plots.plotgrid('4-point level sets with no antisymmetry property (see plot description in terminal)',plots.comparelevelsets,subkey,A=1,B=1,savename='levelsets')

	randkey,subkey=jax.random.split(randkey)
	plots.plotgrid('Antisymmetry plots (see plot description in terminal)',plots.showsymmetry,subkey,A=1,B=1,savename='antisymmetry')

	randkey,subkey=jax.random.split(randkey)
	plots.plotgrid('segments',plots.segment,subkey,b=1,A=3,B=5,savename='segments')

	plt.show()
	


if __name__=='__main__':
	filename=input("type name of file to plot or press enter for most recent. ")
	if(filename==""):
		filename="most_recent"
	plots=Plots("data/"+filename)
	allplots(plots)	
	


