import numpy as np
from scipy.special import digamma, gammaln

from bnpy.allocmodel.topics import OptimizerRhoOmegaBetter as OROB

def gap(N1, N2, A):
	return \
		gammaln(N1 + A) - gammaln(N1 + A + 1) \
		- gammaln(N2 + A)  + gammaln(N2 + A + 1)

def L_hdp(beta, omega, Tvec, alpha):
	''' Compute top-tier of hdp bound.
	'''
	K = omega.size
	assert K == beta.size
	assert K == Tvec.size
	rho = OROB.beta2rho(beta, K)
	eta1 = rho * omega
	eta0 = (1-rho) * omega
	digamma_omega = digamma(omega)
	digamma_eta1 = digamma(eta1)
	digamma_eta0 = digamma(eta0)

	ElogU = digamma_eta1 - digamma_omega
	Elog1mU = digamma_eta0 - digamma_omega
	Lscore = \
		np.sum(gammaln(eta1) + gammaln(eta0) - gammaln(omega)) \
		+ np.inner(nDoc + 1 - eta1, ElogU) \
		+ np.inner(nDoc * OROB.kvec(K) + gamma - eta0, Elog1mU) \
		+ alpha * np.inner(beta, Tvec)
	return Lscore

def argsort_bigtosmall_stable(avec):
    avec = np.asarray(avec)
    assert avec.ndim == 1
    return np.argsort(-1* avec, kind='mergesort')

if __name__ == '__main__':
	alpha = 0.5
	gamma = 10
	nDoc = 500
	#Tvec = np.asarray([-100, -10, -20])
	Tvec = np.asarray([
		-30324.1449,  -99350.3919, -135508.8616, -135988.3834, -143709.5131
		])
	K = Tvec.size
	Uvec = np.zeros(K)

	'''
	betaSORT = np.asarray([0.3, 0.2, 0.1])
	omega = OROB.make_initomega(K, nDoc, gamma)
	print L_hdp(betaSORT, omega, Tvec, alpha)

	betaUNSORT = np.asarray([0.2, 0.3, 0.1])
	print L_hdp(betaUNSORT, omega, Tvec, alpha)
	'''
	# Find optimal rho
	rho, omega, f, Info = OROB.find_optimum(
		alpha=alpha,
		gamma=gamma,
		nDoc=nDoc,
		sumLogPiActiveVec=Tvec,
		sumLogPiRemVec=Uvec)
	beta_opt = OROB.rho2beta_active(rho)
	Lbest = L_hdp(beta_opt, omega, Tvec, alpha)
	assert np.allclose(-1 * f, Lbest)

	sortIDs = argsort_bigtosmall_stable(Tvec)
	Lbest_sorted = L_hdp(beta_opt[sortIDs], omega, Tvec[sortIDs], alpha)

	print(" % .5e Lbest" % (Lbest))
	print(" % .5e Lbest after sorting" % (Lbest_sorted))

	print(beta_opt)
	print(beta_opt[sortIDs])
