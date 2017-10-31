import numpy as np
import time
import bnpy
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataName', default='AdmixAsteriskK8')
	parser.add_argument('--nDocTotal', type=int, default=200)
	parser.add_argument('--K', type=int, default=200)
	parser.add_argument('--nnzPerRowLP', type=int, default=5)
	parser.add_argument('--convThrLP', type=float, default=-1.0)
	parser.add_argument('--nCoordAscentItersLP', type=int, default=50)
	parser.add_argument('--initLaps', type=int, default=2)
	args = parser.parse_args()

	if args.dataName == 'AdmixAsteriskK8':
		import AdmixAsteriskK8
		Data = AdmixAsteriskK8.get_data(nDocTotal=args.nDocTotal, nObsPerDoc=200)
		hmodel, Info = bnpy.run(Data, 'HDPTopicModel', 'Gauss', 'memoVB',
			ECovMat='diagcovdata', sF=0.1,
			nLap=args.initLaps, initname='randexamples', K=args.K, nBatch=1)
	else:
		import MixBarsK10V900
		Data = MixBarsK10V900.get_data(nDocTotal=args.nDocTotal, nWordsPerDoc=500)
		hmodel, Info = bnpy.run(Data, 'HDPTopicModel', 'Mult', 'memoVB',
			lam=0.1,
			nLap=args.initLaps, initname='randexamples', K=args.K, nBatch=1)

	tstart = time.time()
	yesaLP = hmodel.calc_local_params(Data,
		activeonlyLP=1,
		doSparseOnlyAtFinalLP=0,
		restartLP=0,
		restartNumTrialsLP=0,
		nnzPerRowLP=args.nnzPerRowLP, 
		convThrLP=args.convThrLP,
		nCoordAscentItersLP=args.nCoordAscentItersLP,
		initDocTopicCountLP='setDocProbsToEGlobalProbs')
	a_elapsed = time.time() - tstart
	tstart = time.time()
	noaLP = hmodel.calc_local_params(Data, 
		activeonlyLP=0,
		doSparseOnlyAtFinalLP=0,
		restartLP=0,
		restartNumTrialsLP=0,
		nnzPerRowLP=args.nnzPerRowLP, 
		convThrLP=args.convThrLP,
		nCoordAscentItersLP=args.nCoordAscentItersLP,
		initDocTopicCountLP='setDocProbsToEGlobalProbs')
	b_elapsed = time.time() - tstart

	try:
		assert np.allclose(
			yesaLP['DocTopicCount'], noaLP['DocTopicCount'],
			rtol=0, atol=0.05)
	except AssertionError:
		print("BADNESS! learned DocTopicCount not allclose.")
	try:
		assert np.allclose(yesaLP['spR'].toarray(), noaLP['spR'].toarray(),
			rtol=0, atol=0.05)
		#assert np.allclose(yesaLP['spR'].data, noaLP['spR'].data)
	except AssertionError:
		print("BADNESS! learned resp not allclose.")

	tstart = time.time()
	denseLP = hmodel.calc_local_params(Data, 
		activeonlyLP=0,
		doSparseOnlyAtFinalLP=0,
		restartLP=0,
		restartNumTrialsLP=0,
		nnzPerRowLP=0,
		convThrLP=args.convThrLP,
		nCoordAscentItersLP=args.nCoordAscentItersLP,
		initDocTopicCountLP='setDocProbsToEGlobalProbs')
	dense_elapsed = time.time() - tstart

	print("Without restarts. MAX_ITER: %d. convThr %.5f " % (
		args.nCoordAscentItersLP, args.convThrLP))
	print("DENSE O(K)        : ", dense_elapsed)
	print("SPARSE O(K)       : ", b_elapsed)
	print("SPARSE O(Kactive) : ", a_elapsed)

	tstart = time.time()
	denserestartLP = hmodel.calc_local_params(Data, 
		activeonlyLP=0,
		doSparseOnlyAtFinalLP=0,
		restartLP=1,
		restartNumTrialsLP=25,
		nnzPerRowLP=0,
		convThrLP=args.convThrLP,
		nCoordAscentItersLP=args.nCoordAscentItersLP,
		initDocTopicCountLP='setDocProbsToEGlobalProbs')
	dense_elapsed = time.time() - tstart

	tstart = time.time()
	sprestartLP = hmodel.calc_local_params(Data, 
		activeonlyLP=1,
		doSparseOnlyAtFinalLP=0,
		restartLP=1,
		restartNumTrialsLP=25,
		nnzPerRowLP=args.nnzPerRowLP,
		convThrLP=args.convThrLP,
		nCoordAscentItersLP=args.nCoordAscentItersLP,
		initDocTopicCountLP='setDocProbsToEGlobalProbs')
	r_elapsed = time.time() - tstart
	print('')
	print("WITH RESTARTS. MAX_ITER: %d. convThr %.5f " % (
		args.nCoordAscentItersLP, args.convThrLP))
	print("DENSE O(K)        : ", dense_elapsed)
	print("SPARSE O(Kactive) : ", r_elapsed)

	print('dense')
	for key in ['nRestartsAccepted', 'nRestartsTried']:
		print(key, denserestartLP['Info'][key])
	print('sparse')
	for key in ['nRestartsAccepted', 'nRestartsTried']:
		print(key, sprestartLP['Info'][key])
