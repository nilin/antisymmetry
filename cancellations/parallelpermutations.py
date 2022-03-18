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
import util
import jax
import jax.numpy as jnp
import optax
import bisect
	

def nextperm(p):
	n=len(p)
	i=n-1
	while p[i-1]>p[i]:
		i=i-1
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
	return int((-1)**inversions)

	

def printperm(p):
	print('k: '+str(perm_to_k(p)))
	print('p: '+str([i+1 for i in p]))
	print('sign: '+str(sign(p)))
	

def test(n):
	p=list(range(n))
	for k in range(math.factorial(n)):
		print('\nsequentially generated')
		printperm(p)
		print('generated from k')
		printperm(k_to_perm(k,n))
		
		
		assert(selections_to_perm(perm_to_selections(p))==p)
		assert(perm_to_k(p)==k)
		assert(k_to_perm(k,n)==p)		

		p=nextperm(p)
	


test(5)	
