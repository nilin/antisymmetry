import multiprocessing as mp
import jax.numpy as jnp


def f(x):
	return jnp.tanh(x)


print('test')

if __name__=='__main__':
	with mp.Pool(4) as pool:
		a=pool.map(f,[0,1,2,3])
		print(a)
