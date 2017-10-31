import numpy as np
import time

from bnpy.data import GroupXData


def get_data(nDocTotal=200, nObsPerDoc=300,
		nLetterPerDoc=3, seed=0, dstart=0, **kwargs):
	''' Generate data as GroupXData object

	Guarantees that each letter is used at least once every 26 docs.

	'''
	nLetters = 26
	PRNG = np.random.RandomState(seed)
	# Letters decay in probability from A to Z
	LetterProbs = np.ones(nLetters)
	for i in range(1, nLetters):
		LetterProbs[i] = 0.95 * LetterProbs[i - 1]
	LetterProbs /= LetterProbs.sum()

	X = np.zeros((nDocTotal * nObsPerDoc, 64))
	TrueZ = np.zeros(nDocTotal * nObsPerDoc)
	doc_range = np.zeros(nDocTotal + 1, dtype=np.int32)
	for d in range(nDocTotal):
		start_d = d * nObsPerDoc
		doc_range[d] = start_d
		doc_range[d + 1] = start_d + nObsPerDoc

		# Select subset of letters to appear in current document
		mustIncludeLetter = (dstart + d) % 26
		chosenLetters = PRNG.choice(
			nLetters, size=nLetterPerDoc, p=LetterProbs, replace=False)
		loc = np.flatnonzero(chosenLetters == mustIncludeLetter)
		if loc.size > 0:
			chosenLetters[loc[0]] = mustIncludeLetter
		else:
			chosenLetters[-1] = mustIncludeLetter
		lProbs_d = LetterProbs[chosenLetters]/LetterProbs[chosenLetters].sum()
		nObsPerChoice = PRNG.multinomial(nObsPerDoc, lProbs_d)
		assert nObsPerChoice.sum() == nObsPerDoc
		start = start_d
		for i in range(nLetterPerDoc):
			TrueZ[start:(start+nObsPerChoice[i])] = chosenLetters[i]
			Lcovmat = letter2covmat(chr(CHRSTART + chosenLetters[i]))
			X[start:(start+nObsPerChoice[i])] = PRNG.multivariate_normal(
				np.zeros(64), Lcovmat, size=nObsPerChoice[i])
			start += nObsPerChoice[i]
	for i in range(nLetters):
		print(chr(CHRSTART + i), np.sum(TrueZ == i))
	return GroupXData(X=X, TrueZ=TrueZ, doc_range=doc_range)


CHRSTART = 65

A = (
	"00011000" + 
	"01100110" + 
	"01100110" + 
	"11111111" + 
	"11111111" + 
	"11000011" + 
	"11000011" + 
	"11000011"
	)

BLANK = (
	"00000000" +
	"00000000" +
	"00000000" +
	"00000000" +
	"00000000" +
	"00000000" +
	"00000000" +
	"00000000"
	)

B = (
	"11110000" +
	"11000111" +
	"11000111" +
	"11111000" +
	"11111000" +
	"11000111" +
	"11000111" +
	"11111000"
	)

C = (
	"11111111" +
	"11111111" +
	"11000000" +
	"11000000" +
	"11000000" +
	"11000000" +
	"11111111" +
	"11111111"
	)

D = (
	"11111000" +
	"11111111" +
	"11000111" +
	"11000011" +
	"11000011" +
	"11000111" +
	"11111111" +
	"11111000"
	)

E = (
	"11111111" +
	"11111111" +
	"11000000" +
	"11111000" +
	"11111000" +
	"11000000" +
	"11111111" +
	"11111111"
	)

F = (
	"11111111" +
	"11111111" +
	"11000000" +
	"11111000" +
	"11111000" +
	"11000000" +
	"11000000" +
	"11000000"
	)

G = (
	"11111111" +
	"11111111" +
	"11000000" +
	"11000000" +
	"11001111" +
	"11000011" +
	"11111111" +
	"11111111"
	)

H = (
	"11000011" + 
	"11000011" + 
	"11000011" + 
	"11111111" + 
	"11111111" + 
	"11000011" + 
	"11000011" + 
	"11000011"
	)

I = (
	"11111111" + 
	"11111111" + 
	"00011000" +
	"00011000" +
	"00011000" +
	"00011000" +
	"11111111" + 
	"11111111" 
	)

J = (
	"11111111" + 
	"11111111" + 
	"00011000" +
	"00011000" +
	"10011000" +
	"11011000" +
	"11111000" + 
	"11111000" 
	)

K = (
	"11000011" + 
	"11000111" + 
	"11001100" + 
	"11111000" + 
	"11111000" + 
	"11001100" + 
	"11000111" + 
	"11000011"
	)

L = (
	"11000000" + 
	"11000000" + 
	"11000000" + 
	"11000000" + 
	"11000000" + 
	"11000000" + 
	"11111111" + 
	"11111111"
	)

M = (
	"11000011" + 
	"11100111" + 
	"11100111" + 
	"11011011" + 
	"11011011" + 
	"11000011" + 
	"11000011" + 
	"11000011"
	)

N = (
	"11000011" + 
	"11100011" + 
	"11100011" + 
	"11011011" + 
	"11011011" + 
	"11000111" + 
	"11000111" + 
	"11000011"
	)


O = (
	"11111111" + 
	"11111111" + 
	"11000011" + 
	"11000011" + 
	"11000011" + 
	"11000011" + 
	"11111111" + 
	"11111111"
	)

P = (
	"11111111" + 
	"11100111" + 
	"11000011" + 
	"11100111" + 
	"11111111" + 
	"11000000" + 
	"11000000" + 
	"11000000" 
	)

Q = (
	"11111110" + 
	"11000110" + 
	"11000110" + 
	"11000110" + 
	"11000110" + 
	"11010110" + 
	"11111110" +
	"00000011" 
	)


R = (
	"11111111" + 
	"11100111" + 
	"11000011" + 
	"11100111" + 
	"11111111" + 
	"11110000" + 
	"11001110" + 
	"11000011" 
	)


S = (
	"00011111" + 
	"01111111" + 
	"11100000" + 
	"01111000" + 
	"00011110" + 
	"00000011" + 
	"11111110" + 
	"11111000"
	)

T = (
	"11111111" + 
	"11111111" + 
	"00011000" + 
	"00011000" + 
	"00011000" + 
	"00011000" + 
	"00011000" + 
	"00011000" 
	)

U = (
	"11000011" + 
	"11000011" + 
	"11000011" + 
	"11000011" + 
	"11000011" + 
	"11000011" + 
	"11111111" + 
	"11111111"
	)


V = (
	"11000011" + 
	"11000011" + 
	"11000011" + 
	"11000011" + 
	"11000011" + 
	"01000010" + 
	"00111100" + 
	"00011000"
	)

W = (
	"11000011" +
	"11000011" + 
	"11000011" + 
	"11011011" + 
	"11011011" + 
	"11100111" + 
	"11100111" + 
	"11000011"  
	)

X = (
	"11000011" + 
	"11100111" + 
	"01100110" + 
	"00011000" + 
	"00011000" + 
	"01100110" + 
	"11100111" + 
	"11000011"
	)

Y = (
	"11000011" + 
	"11100111" + 
	"01100110" + 
	"00011000" + 
	"00011000" + 
	"00011000" + 
	"00011000" + 
	"00011000" 
	)

Z = (
	"11111111" + 
	"11111111" + 
	"00000110" + 
	"00011000" + 
	"00011000" + 
	"01100000" + 
	"11111111" + 
	"11111111" 
	)

def str2imarr(s):
	assert len(s) == 64
	valList = list()
	for c in s:
		valList.append(int(c))
	return np.asarray(valList, dtype=np.int32).reshape((8,8))

def letter2covmat(Lstr, N=100, sig=0.01):
	assert len(Lstr) == 1
	seed = ord(Lstr)
	PRNG = np.random.RandomState(seed)
	Lim = str2imarr(globals()[Lstr])
	# Generate 100 samples with random noise, with on pixels having POS vals
	Xall = np.zeros((2*N, 8**2))
	for n in range(N):
		Xim = Lim + sig * PRNG.randn(8,8)
		Xim -= np.mean(Xim)
		Xall[n] = Xim.flatten()
	# Generate 100 samples with random noise, with on pixels having NEG vals
	for n in range(N):
		Xim = -1.0 * Lim + sig * PRNG.randn(8,8)
		Xim -= np.mean(Xim)
		Xall[N+n] = Xim.flatten()
	CovMat = np.dot(Xall.T, Xall) / (2*N)
	return CovMat

if __name__ == "__main__":
	from matplotlib import pylab; pylab.ion()
	Data = get_data()
	from IPython import embed; embed()
	'''
	ncols=3
	figH = pylab.subplots(nrows=1, ncols=ncols)
	for letterID in range(26):
		letter = chr(65 + letterID)

		Lcovmat = letter2covmat(letter)
		# draw lots of samples
		for n in range(ncols):
			Xflat = np.random.multivariate_normal(np.zeros(64), Lcovmat)
			Xim = np.reshape(Xflat, (8,8))
			pylab.subplot(1, ncols, n+1);
			pylab.imshow(Xim,
				cmap='coolwarm', interpolation='nearest', vmin=-1, vmax=1)
			pylab.axis('off')
		pylab.show(block=False)
		pylab.draw()
		time.sleep(0.2);
		pylab.clf();
	'''
	'''
	for letterID in range(26):
		letter = chr(65 + letterID)
		Lstr = globals()[letter]
		Lim = str2imarr(Lstr)
		pylab.imshow(Lim, interpolation='nearest', cmap='gray');
		pylab.axis('image');
		pylab.show(block=False)
		pylab.draw()
		time.sleep(0.2);
		pylab.clf();
	'''
