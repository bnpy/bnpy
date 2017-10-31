from builtins import *
from scipy.special import gammaln
import numpy as np

def Lscore(logPvec, logL=None, alphaEbeta=None, wc_d=1.0):
	'''

	>>> K = 3
	>>> N = 2
	>>> alpha = 0.5
	>>> logPvec = np.log(np.ones(K) / K)
	>>> logL = np.log( np.ones((N, K)) + 0.1 )
	>>> alphaEbeta = alpha * np.ones(K) / K
	>>> val = Lscore(logPvec, logL, alphaEbeta)
	>>> assert isinstance(val, float)
	'''
	N, K = logL.shape
	assert K == logPvec.size

	logR = logL + logPvec[np.newaxis,:]
	logR -= logR.max(axis=1)[:,np.newaxis]
	np.exp(logR, out=logR)
	R = logR
	sumovertopicsR = R.sum(axis=1)
	R /= sumovertopicsR[:,np.newaxis]

	logsumovertopicsR = sumovertopicsR
	if isinstance(wc_d, np.ndarray):
		assert wc_d.size == N
		DocTopicCount = np.dot(wc_d, R)
		LsumR = np.inner(wc_d, logsumovertopicsR)
	else:
		DocTopicCount = np.sum(R, axis=0)
		np.log(sumovertopicsR, out=sumovertopicsR)
		LsumR = np.sum(logsumovertopicsR)

	Lscore = -1 * np.inner(DocTopicCount, logPvec) \
		+ np.sum(gammaln(DocTopicCount + alphaEbeta)) \
		+ LsumR
	return Lscore

if __name__ == "__main__":
	K = 3
	N = 2
	alpha = 0.5
	logPvec = np.log(np.ones(K) / K)
	logL = np.log( np.ones((N, K)) + 0.1 )
	alphaEbeta = alpha * np.ones(K) / K

	print('UNIFORM PROBS')
	print(Lscore(logPvec, logL, alphaEbeta))
	print(Lscore(logPvec-1, logL, alphaEbeta))
	print(Lscore(logPvec-2, logL, alphaEbeta))
	print(Lscore(logPvec+1, logL, alphaEbeta))
	print(Lscore(logPvec+2, logL, alphaEbeta))

	for i in range(3):
		logPvec = np.log( np.random.rand(K))
		print(Lscore(logPvec, logL, alphaEbeta))
