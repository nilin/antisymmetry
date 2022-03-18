# nilin

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
import bookkeep as bk
import util
import jax
import sys
import jax.numpy as jnp
import optax
import bisect
import time
	

def nextperm(p):
	n=len(p)
	i=n-1
	while p[i-1]>p[i]:
		i=i-1
		if i==0:
			return list(range(n))
		
	first_left_downstep=i-1
	last_upstep=i
	
	while i+1<n and p[i+1]>p[first_left_downstep]:
		i=i+1
	last_above_downstep=i

	p[first_left_downstep],p[last_above_downstep]=p[last_above_downstep],p[first_left_downstep]
	return p[:last_upstep]+list(reversed(p[last_upstep:]))
		

def perm_to_selections(p):
	n=len(p)	
	seen=[]
	selections=[]

	for i in range(n):
		s=p[i]-np.searchsorted(seen,p[i])
		selections.append(s)
		bisect.insort(seen,p[i]) #O(n) :|
	
	return selections


def selections_to_perm(S):
	n=len(S)
	options=list(range(n))
	p=[]
	for i in range(n):
		s=S[i]	
		p.append(options[s])	
		options=options[:s]+options[s+1:]
	return p


def perm_to_k(p):
	selections=perm_to_selections(p)
	n=len(p)	
	base=1
	k=0
	for i in range(1,n+1):
		j=n-i
		k=k+base*selections[j]
		base=base*i
	return k

def k_to_perm(k,n):
	s=[]
	base=1
	r=k
	for base in range(1,n+1):
		s.append(r%base)
		r=r//base
	s.reverse()
	return selections_to_perm(s)


def sign(p):
	n=len(p)
	p_j=jnp.repeat(jnp.expand_dims(jnp.array(p),axis=0),n,axis=0)
	p_i=p_j.T
	inversions=jnp.sum(jnp.triu(jnp.heaviside(-(p_j-p_i),0)))
	return int(1-2*(inversions%2))

def perm_as_matrix_1(p):
	n=len(p)
	P=[list(jnp.zeros(n)) for i in range(n)]
	for i in range(n):
		P[i][p[i]]=1
	return jnp.array(P)

def perm_as_matrix_2(p):
	n=len(p)
	i_=jnp.repeat(jnp.expand_dims(jnp.arange(n),axis=1),n,axis=1)
	diffs=jax.vmap(jnp.add,in_axes=(0,None))(i_,-jnp.array(p))
	discretedelta=lambda x:util.ReLU(-jnp.square(x)+1)
	return discretedelta(diffs)

perm_as_matrix=perm_as_matrix_2

def printperm(p):
	print('k: '+str(perm_to_k(p)))
	print('p: '+str([i+1 for i in p]))
	print('sign: '+str(sign(p)))
	print(perm_as_matrix_1(p))
	print(perm_as_matrix_2(p))
	print()
	
def id(n):
	return list(range(n))



"""
tests----------------------------------------------------------------------------------------------------s
"""

	

def performancetest(n):

	N=100
	clock=bk.Stopwatch()

	p=id(n)
	for i in range(N):
		nextperm(p)

	print('next_perm '+str(N/clock.tick())+'/second')

	for i in range(N):
		k_to_perm(i,n)

	print('k_to_perm '+str(N/clock.tick())+'/second')

	p=id(n)
	for i in range(N):
		perm_as_matrix_1(p)
		p=nextperm(p)

	print('perm_as_matrix_1: '+str(N/clock.tick())+'/second')

	p=id(n)
	for i in range(N):
		perm_as_matrix_2(p)
		p=nextperm(p)

	print('perm_as_matrix_2: '+str(N/clock.tick())+'/second')
	

	p=id(n)
	for i in range(N):
		sign(p)
		p=nextperm(p)

	print('sign: '+str(N/clock.tick())+'/second')


	


def test(n):
	p=list(range(n))

	print('\nsequentially generated'+100*'-')
	for k in range(2*math.factorial(n)):
		printperm(p)
		verify(k,p,n)
		p=nextperm(p)

	print('generated from k'+100*'-')
	for k in range(2*math.factorial(n)):
		p=k_to_perm(k,n)
		printperm(p)
		verify(k,p,n)

def verify(k,p,n):		
	assert(perm_to_k(p)==k%math.factorial(n))
	assert(k_to_perm(k,n)==p)		
	assert(selections_to_perm(perm_to_selections(p))==p)

"""
tests----------------------------------------------------------------------------------------------------s
"""

if len(sys.argv)>1 and sys.argv[1]=='test':
	test(4)
	performancetest(int(sys.argv[2]))
