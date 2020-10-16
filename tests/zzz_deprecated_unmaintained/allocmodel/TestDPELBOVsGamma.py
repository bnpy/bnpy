import argparse
import numpy as np
from bnpy.allocmodel.mix.DPMixtureModel import c_Beta, convertToN0

from matplotlib import pylab

def calc_Lalloc_from_count_vec(N_K=None, gamma1=1.0, gamma0=1.0):
	''' Compute allocation-model ELBO term for DP mixture

	Args
	----
	N_K : 1D array, size K
	gamma1 : positive scalar
	gamma0 : positive scalar

	Returns
	-------
	Lalloc : scalar
	'''
	N_K = np.asarray(N_K, dtype=np.float64)
	K = N_K.size
	eta1 = N_K + gamma1
	eta0 = convertToN0(N_K) + gamma0
	return K * c_Beta(gamma1, gamma0) - c_Beta(eta1, eta0)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--N_K', type=str, default='100,50,10')
	parser.add_argument('--Ntotal', type=int, default=0)
	args = parser.parse_args()
	Ntotal = args.Ntotal
	
	if args.N_K.count(','):
		N_K = np.asarray([float(x) for x in args.N_K.split(',')])
		if Ntotal > 0:
			N_K = Ntotal / N_K.sum() * N_K
		N_K_list = [N_K]
		frac_vals = [0]
		K = N_K.size
	else:
		K = 3
		frac_vals = np.linspace(0.5, 0.999, 10)
		N_K_list = list()
		for ff, frac in enumerate(frac_vals):
			N_K_list.append(Ntotal * np.asarray([frac, (1-frac)]))

	for ff, N_K in enumerate(N_K_list):
		print(' '.join(['%5.1f' % (x) for x in N_K]))
		gamma0_vals = np.arange(0.05, 20, 0.05)
		elbo_vals = np.zeros_like(gamma0_vals)
		for ii, gamma0 in enumerate(gamma0_vals):
			elbo_vals[ii] = calc_Lalloc_from_count_vec(
				N_K=np.concatenate([N_K, np.zeros(K - N_K.size)]),
				gamma0=gamma0)
		pylab.plot(gamma0_vals, elbo_vals, '.-', label='frac=%.3f' % (frac_vals[ff]))
		print('best gamma0 = ', (gamma0_vals[np.argmax(elbo_vals)]))
	pylab.legend()
	pylab.show()
