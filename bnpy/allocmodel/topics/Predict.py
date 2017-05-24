#!/contrib/projects/anaconda-python/miniconda2/bin/python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.special import digamma
import sys
import bnpy
#from bnpy.viz import PrintTopics

from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import os

#from bnpy.data import WordsData
#import grid3x3_nD400_nW100
#import grid3x3_nD900_nW200
#import grid3x3_nD900_nW200_nV225
#import MovieReviews
import MovieReviews_sLDA01


from bnpy.util import NumericUtil

from sklearn.linear_model import LinearRegression

from sklearn import linear_model


def predict_with_linreg_on_wordcounts(TrainData,TestData):
	'''
	Do prediction using linear regression
	Features: word counts
	'''
	
	TrainData = MovieReviews_sLDA01.get_data()
	#TrainData.name = 'Train'
	TestData = MovieReviews_sLDA01.get_test_data()
	#TestData.name = 'Test'
	lap = 500
	nDocTrain = TrainData.nDoc
	nDocTest = TestData.nDoc
	vocab_size = TrainData.vocab_size
	f = open('/home/lweiner/bnpy_repo/bnpy-dev/datasets/rawData/MovieReviews_sLDA01/vocab.txt','r')
	vocab = f.readlines()
	for i in range(len(vocab)):
		vocab[i] = vocab[i].strip('\n')
		
	#unfold word count array
	Xtrain = np.zeros((nDocTrain,vocab_size))
	for i in xrange(nDocTrain):
		starti = TrainData.doc_range[i]
		endi = TrainData.doc_range[i+1]	
		wc_d = TrainData.word_count[starti:endi]
		wid_d = TrainData.word_id[starti:endi]
		Xtrain[i,wid_d] = wc_d
		Xtrain[i,:] = Xtrain[i,:] /  float(np.sum(wc_d))
	#Xtrain = ((Xtrain - Xtrain.mean(0)) / Xtrain.std(0))
	
	Ytrain = TrainData.response
	
	Xtest = np.zeros((nDocTest,vocab_size))
	for i in xrange(nDocTest):
		starti = TestData.doc_range[i]
		endi = TestData.doc_range[i+1]	
		wc_d = TestData.word_count[starti:endi]
		wid_d = TestData.word_id[starti:endi]
		Xtest[i,wid_d] = wc_d
		Xtest[i,:] = Xtest[i,:] /  float(np.sum(wc_d))
	#Xtest = ((Xtest - Xtest.mean(0)) / Xtest.std(0))
		
	Ytest = TestData.response
	
	y_pred_train = np.zeros(nDocTrain)
	y_pred_test = np.zeros(nDocTest)
		
	# Fit model to train data
	#linear_model = LinearRegression(normalize=True)
	
	#reg = linear_model.Lasso(alpha=a)
	print 'UNREGULARIZED LINEAR REGRESSION'
	reg = linear_model.LinearRegression()
	reg.fit(Xtrain,Ytrain)

	# Predict
	y_pred_train = reg.predict(Xtrain)
	y_pred_test = reg.predict(Xtest)

	# Score
	r2_Train = r2_score(Ytrain,y_pred_train)
	rmse_Train = np.sqrt(mean_squared_error(Ytrain,y_pred_train))

	r2_Test = r2_score(Ytest,y_pred_test)
	rmse_Test = np.sqrt(mean_squared_error(Ytest,y_pred_test))

	print '    r2 Train: ', r2_Train
	print '    r2 Test: ', r2_Test
	#print rmse_Train, rmse_Test
	
	print '\n\n'
	
	for a in [0,0.01,0.1,0.6,1.0]:
		
		#reg = linear_model.Lasso(alpha=a)
		lreg = linear_model.Lasso(alpha=a)
		lreg.fit(Xtrain,Ytrain)
	
		# Predict
		y_pred_train = lreg.predict(Xtrain)
		y_pred_test = lreg.predict(Xtest)
	
		# Score
		r2_Train = r2_score(Ytrain,y_pred_train)
		rmse_Train = np.sqrt(mean_squared_error(Ytrain,y_pred_train))
	
		r2_Test = r2_score(Ytest,y_pred_test)
		rmse_Test = np.sqrt(mean_squared_error(Ytest,y_pred_test))
	
		print 'LASSO with alpha=%.3f' % a
		print '    r2 Train: ', r2_Train
		print '    r2 Test: ', r2_Test

	print '\n\n'
	for a in  [0,0.01,0.1,0.6,1.0]:
		#print 'alpha: ', a
		#reg = linear_model.Lasso(alpha=a)
		rreg = linear_model.Ridge(alpha=a)
		rreg.fit(Xtrain,Ytrain)
	
		# Predict
		y_pred_train = rreg.predict(Xtrain)
		y_pred_test = rreg.predict(Xtest)
	
		# Score
		r2_Train = r2_score(Ytrain,y_pred_train)
		rmse_Train = np.sqrt(mean_squared_error(Ytrain,y_pred_train))
	
		r2_Test = r2_score(Ytest,y_pred_test)
		rmse_Test = np.sqrt(mean_squared_error(Ytest,y_pred_test))
		
		print 'RIDGE with alpha=%.3f' % a
		print '    r2 Train: ', r2_Train
		print '    r2 Test: ', r2_Test


def predict_with_lda_plus_linreg(job_path,TrainData,TestData,lap=None):
	'''
	Predict supervision response with LDA plus linear regression on resp (empirical topic distribution)
	'''
	try:
		hmodel = bnpy.ioutil.ModelReader.load_model(job_path)
	except:
		hmodel = bnpy.ioutil.ModelReader.load_model(job_path,lap=lap)
	LP = hmodel.calc_local_params(TrainData)
	respTrain = LP['resp']
	_,K = respTrain.shape
	theta_init = LP['theta']
	#Lik = LP['E_log_soft_ev']
	beta = LP['ElogphiT'].copy()
	beta -= beta.max(axis=1)[:,np.newaxis]
	NumericUtil.inplaceExp(beta)
	
	nDocTrain = TrainData.nDoc
	nDocTest = TestData.nDoc
	#eta = hmodel.allocModel.eta
	alpha = hmodel.allocModel.alpha
	
	y_pred_train = np.zeros(nDocTrain)
	Xtrain = np.zeros((nDocTrain,K))
	for i in xrange(nDocTrain):
		#print i
		starti = TrainData.doc_range[i]
		endi = TrainData.doc_range[i+1]	
		wc_d = TrainData.word_count[starti:endi]
		N_d = int(wc_d.sum())
		resp_d = respTrain[starti:endi]
		Xtrain[i] = np.sum(resp_d * wc_d[:,None],axis=0) / N_d
	
	Ytrain = TrainData.response	
			
		
	#K = resp_init.shape[1]
	#LPTest = hmodel.calc_local_params(TestData)
	#resp = LPTest['resp']
	#theta = LPTest['theta']
	respTest = np.zeros((TestData.word_count.shape[0],K))
	thetaTest = np.ones((nDocTest,K)) / K
	y_pred_test = np.zeros(nDocTest)
	Xtest = np.zeros((nDocTest,K))
	for i in xrange(nDocTest):
		starti = TestData.doc_range[i]
		endi = TestData.doc_range[i+1]	
		wc_d = TestData.word_count[starti:endi]
		wid_d = TestData.word_id[starti:endi]
		N_d = int(wc_d.sum())
		
		resp_d = respTest[starti:endi]
		theta_d = thetaTest[i]
		
		N_t = resp_d.shape[0]
		convThrLP = 0.001
		converged = False
		for iter in xrange(200):
			if converged:
				break
	
			for ll in xrange(N_t):
				wid = wid_d[ll]
				Elogpi = digamma(theta_d) - digamma(sum(theta_d))
				resp_d[ll] = beta[wid,:] * np.exp(Elogpi)
				
				rsum = sum(resp_d[ll])
				resp_d[ll] = resp_d[ll] / rsum

			prev_theta = theta_d
			theta_d = alpha + np.sum(wc_d[:,None] * resp_d,axis=0) 
			
			maxDiff = np.max(np.abs(theta_d - prev_theta))
			if maxDiff < convThrLP:
				converged = True

		Xtest[i] = np.sum(resp_d * wc_d[:,None],axis=0) / N_d
	
	Ytest = TestData.response
	
		
	# Fit model to train data
	linear_model = LinearRegression()
	linear_model.fit(Xtrain,Ytrain)
	
	# Predict
	y_pred_train = linear_model.predict(Xtrain)
	y_pred_test = linear_model.predict(Xtest)
	
	# Score
	r2_Train = r2_score(Ytrain,y_pred_train)
	rmse_Train = np.sqrt(mean_squared_error(Ytrain,y_pred_train))
	
	r2_Test = r2_score(Ytest,y_pred_test)
	rmse_Test = np.sqrt(mean_squared_error(Ytest,y_pred_test))
	
	return r2_Train, rmse_Train, r2_Test, rmse_Test
	
	
	
	
	

def predict_with_lda_plus_linreg_on_pi(job_path,TrainData,TestData,lap=None):
	'''
	Predict response with LDA plus linear regression on pi (document topic distribution)
	'''
	try:
		hmodel = bnpy.ioutil.ModelReader.load_model(job_path)
	except:
		hmodel = bnpy.ioutil.ModelReader.load_model(job_path,lap=lap)

	LP = hmodel.calc_local_params(TrainData)
	respTrain = LP['resp']
	_,K = respTrain.shape
	thetaTrain = LP['theta']
	#Lik = LP['E_log_soft_ev']
	beta = LP['ElogphiT'].copy()
	beta -= beta.max(axis=1)[:,np.newaxis]
	NumericUtil.inplaceExp(beta)
	
	nDocTrain = TrainData.nDoc
	nDocTest = TestData.nDoc
	#eta = hmodel.allocModel.eta
	alpha = hmodel.allocModel.alpha
	
	y_pred_train = np.zeros(nDocTrain)
	Xtrain = np.zeros((nDocTrain,K))
	for i in xrange(nDocTrain):
		#print i
		#start = TrainData.doc_range[i]
		#end = TrainData.doc_range[i+1]	

		#resp_d = respTrain[start:end]
		#Xtrain[i] = np.mean(resp_d,axis=0)
		Xtrain[i] = thetaTrain[i]
		
	Ytrain = TrainData.response	
		
		
	#K = resp_init.shape[1]
	#LPTest = hmodel.calc_local_params(TestData)
	#resp = LPTest['resp']
	#theta = LPTest['theta']
	respTest = np.zeros((TestData.word_count.shape[0],K))
	thetaTest = np.ones((nDocTest,K)) / K
	y_pred_test = np.zeros(nDocTest)
	Xtest = np.zeros((nDocTest,K))
	for i in xrange(nDocTest):
		start = TestData.doc_range[i]
		end = TestData.doc_range[i+1]	
		wc_d = TestData.word_count[start:end]
		wid_d = TestData.word_id[start:end]
		N_d = int(wc_d.sum())
		
		resp_d = respTest[start:end]
		theta_d = thetaTest[i]
		
		N_t = resp_d.shape[0]
		convThrLP = 0.001
		converged = False
		for iter in xrange(200):
			if converged:
				break
	
			for ll in xrange(N_t):
				wid = wid_d[ll]
				Elogpi = digamma(theta_d) - digamma(sum(theta_d))
				resp_d[ll] = beta[wid,:] * np.exp(Elogpi)
				
				rsum = sum(resp_d[ll])
				resp_d[ll] = resp_d[ll] / rsum

			prev_theta = theta_d
			theta_d = alpha + np.sum(wc_d[:,None] * resp_d,axis=0) 
			
			maxDiff = np.max(np.abs(theta_d - prev_theta))
			if maxDiff < convThrLP:
				converged = True
	
		Xtest[i] = theta_d
	
	Ytest = TestData.response
	
		
	# Fit model to train data
	linear_model = LinearRegression()
	linear_model.fit(Xtrain,Ytrain)
	
	# Predict
	y_pred_train = linear_model.predict(Xtrain)
	y_pred_test = linear_model.predict(Xtest)
	
	# Score
	r2_Train = r2_score(Ytrain,y_pred_train)
	rmse_Train = np.sqrt(mean_squared_error(Ytrain,y_pred_train))
	
	r2_Test = r2_score(Ytest,y_pred_test)
	rmse_Test = np.sqrt(mean_squared_error(Ytest,y_pred_test))
	
	return r2_Train, rmse_Train, r2_Test, rmse_Test
	

def predict_with_orig_slda(job_path,TrainData,TestData,lap=None):
	'''
	Predict response with orig sLDA (regress on empirical topics)
	Use with SupervisedTopicModel2
	'''
	
	
	
	try:
		hmodel = bnpy.ioutil.ModelReader.load_model(job_path)
	except:
		hmodel = bnpy.ioutil.ModelReader.load_model(job_path,lap=lap)
	
	try:
		hmodel.allocModel.eta = hmodel.allocModel.mean_weights
	except:
		pass
	LP = hmodel.calc_local_params(TrainData)
	respTrain = LP['resp']
	_,K = respTrain.shape
	theta_init = LP['theta']
	#Lik = LP['E_log_soft_ev']
	beta = LP['ElogphiT'].copy()
	beta -= beta.max(axis=1)[:,np.newaxis]
	NumericUtil.inplaceExp(beta)
	
	
	#TrainData.TrueParams['
	
	nDocTrain = TrainData.nDoc
	nDocTest = TestData.nDoc
	eta = hmodel.allocModel.eta
	alpha = hmodel.allocModel.alpha
	
	y_pred_train = np.zeros(nDocTrain)
	#Xtrain = np.zeros((nDocTrain,K))
	for i in xrange(nDocTrain):
		#print i
		start = TrainData.doc_range[i]
		end = TrainData.doc_range[i+1]	
		wc_d = TrainData.word_count[start:end]
		N_d = int(wc_d.sum())
		
		resp_d = respTrain[start:end]
		
		weighted_resp_d = wc_d[:,None] * resp_d 
		EZbar = np.sum(weighted_resp_d,axis=0) / N_d
		y_pred_train[i] = np.dot(eta,EZbar)
	
	Ytrain = TrainData.response	
		
		
		
	#K = resp_init.shape[1]
	#LPTest = hmodel.calc_local_params(TestData)
	#resp = LPTest['resp']
	#theta = LPTest['theta']
	respTest = np.zeros((TestData.word_count.shape[0],K))
	thetaTest = np.ones((nDocTest,K)) / K
	y_pred_test = np.zeros(nDocTest)
	#Xtest = np.zeros((nDocTest,K))
	for i in xrange(nDocTest):
		start = TestData.doc_range[i]
		end = TestData.doc_range[i+1]	
		wc_d = TestData.word_count[start:end]
		wid_d = TestData.word_id[start:end]
		N_d = int(wc_d.sum())
		
		resp_d = respTest[start:end]
		theta_d = thetaTest[i]
		
		N_t = resp_d.shape[0]
		convThrLP = 0.001
		converged = False
		for iter in xrange(200):
			if converged:
				break
	
			for ll in xrange(N_t):
				wid = wid_d[ll]
				Elogpi = digamma(theta_d) - digamma(sum(theta_d))
				resp_d[ll] = beta[wid,:] * np.exp(Elogpi)
				
				rsum = sum(resp_d[ll])
				resp_d[ll] = resp_d[ll] / rsum

			prev_theta = theta_d
			theta_d = alpha + np.sum(wc_d[:,None] * resp_d,axis=0) 
			
			maxDiff = np.max(np.abs(theta_d - prev_theta))
			if maxDiff < convThrLP:
				converged = True
	
		weighted_resp_d = wc_d[:,None] * resp_d 
		EZbar = np.sum(weighted_resp_d,axis=0) / N_d
		y_pred_test[i] = np.dot(eta,EZbar)
			
	Ytest = TestData.response
	
	# Score
	r2_Train = r2_score(Ytrain,y_pred_train)
	rmse_Train = np.sqrt(mean_squared_error(Ytrain,y_pred_train))
	
	r2_Test = r2_score(Ytest,y_pred_test)
	rmse_Test = np.sqrt(mean_squared_error(Ytest,y_pred_test))
	
	return r2_Train, rmse_Train, r2_Test, rmse_Test
	
	
	

def predict_with_alt_slda(job_path,TrainData,TestData,lap=None):
	'''
	Predict response with alt sLDA (regress on topic distribution)
	Use with SupervisedFiniteTopicModelRegressTopicDistribution
	'''
	try:
		hmodel = bnpy.ioutil.ModelReader.load_model(job_path)
	except:
		hmodel = bnpy.ioutil.ModelReader.load_model(job_path,lap=lap)

	try:
		hmodel.allocModel.eta = hmodel.allocModel.mean_weights
	except:
		pass
	LP = hmodel.calc_local_params(TrainData)
	respTrain = LP['resp']
	_,K = respTrain.shape
	theta_init = LP['theta']
	#Lik = LP['E_log_soft_ev']
	beta = LP['ElogphiT'].copy()
	beta -= beta.max(axis=1)[:,np.newaxis]
	NumericUtil.inplaceExp(beta)
	
	nDocTrain = TrainData.nDoc
	nDocTest = TestData.nDoc
	eta = hmodel.allocModel.eta
	alpha = hmodel.allocModel.alpha
	
	y_pred_train = np.zeros(nDocTrain)
	#Xtrain = np.zeros((nDocTrain,K))
	for i in xrange(nDocTrain):
		#print i
		start = TrainData.doc_range[i]
		end = TrainData.doc_range[i+1]	
		wc_d = TrainData.word_count[start:end]
		N_d = int(wc_d.sum())
		
		resp_d = respTrain[start:end]
		
		weighted_resp_d = wc_d[:,None] * resp_d 
		EZbar = np.sum(weighted_resp_d,axis=0) / N_d
		y_pred_train[i] = np.dot(eta,EZbar)
	
	Ytrain = TrainData.response	
		
		
		
	#K = resp_init.shape[1]
	#LPTest = hmodel.calc_local_params(TestData)
	#resp = LPTest['resp']
	#theta = LPTest['theta']
	respTest = np.zeros((TestData.word_count.shape[0],K))
	thetaTest = np.ones((nDocTest,K)) / K
	y_pred_test = np.zeros(nDocTest)
	#Xtest = np.zeros((nDocTest,K))
	for i in xrange(nDocTest):
		start = TestData.doc_range[i]
		end = TestData.doc_range[i+1]	
		wc_d = TestData.word_count[start:end]
		wid_d = TestData.word_id[start:end]
		N_d = int(wc_d.sum())
		
		resp_d = respTest[start:end]
		theta_d = thetaTest[i]
		
		N_t = resp_d.shape[0]
		convThrLP = 0.001
		converged = False
		for iter in xrange(200):
			if converged:
				break
				
			Elogpi = digamma(theta_d) - digamma(sum(theta_d))
			for ll in xrange(N_t):
				wid = wid_d[ll]
				
				resp_d[ll] = beta[wid,:] * np.exp(Elogpi)
				
				rsum = sum(resp_d[ll])
				resp_d[ll] = resp_d[ll] / rsum

			prev_theta = theta_d
			theta_d = alpha + np.sum(wc_d[:,None] * resp_d,axis=0) 
			
			maxDiff = np.max(np.abs(theta_d - prev_theta))
			if maxDiff < convThrLP:
				converged = True
	
		weighted_resp_d = wc_d[:,None] * resp_d 
		EZbar = np.sum(weighted_resp_d,axis=0) / N_d
		y_pred_test[i] = np.dot(eta,EZbar)
			
	Ytest = TestData.response
	
	# Score
	r2_Train = r2_score(Ytrain,y_pred_train)
	rmse_Train = np.sqrt(mean_squared_error(Ytrain,y_pred_train))
	
	r2_Test = r2_score(Ytest,y_pred_test)
	rmse_Test = np.sqrt(mean_squared_error(Ytest,y_pred_test))
	
	return r2_Train, rmse_Train, r2_Test, rmse_Test
	




if __name__ == '__main__':

	pred_method = sys.argv[1]
	data_name = sys.argv[2]
	job_name = sys.argv[3]
	
	print 'PREDICTING %s/%s WITH %s' %(data_name, job_name, pred_method)
	
	all_r2_Train = []
	all_rmse_Train = []
	all_r2_Test = []
	all_rmse_Test = []

	for i in range(1,11):
		
		job_path = '/nbu/liv/lweiner/%s/%s/%d' % (data_name, job_name,i)
		print job_path

		if data_name == 'MovieReviews':
			TrainData = MovieReviews_sLDA01.get_data()
			TrainData.name = 'Train'
			TestData = MovieReviews_sLDA01.get_test_data()
			TestData.name = 'Test'
			lap = 800

		if data_name == 'grid3x3_nD400_nW100':
			TrainData = grid3x3_nD400_nW100.get_data()
			TrainData.name = 'Train'
			TestData = grid3x3_nD400_nW100.get_data(seed=99)
			TestData.name = 'Test'
			lap = 500

		if data_name == 'grid3x3_nD900_nW200':
			TrainData = grid3x3_nD900_nW200.get_data()
			TrainData.name = 'Train'
			TestData = grid3x3_nD900_nW200.get_data(seed=99)
			TestData.name = 'Test'
			lap = 500
		if data_name == 'grid3x3_nD900_nW200_nV225':
			TrainData = grid3x3_nD900_nW200_nV225.get_data()
			TrainData.name = 'Train'
			TestData = grid3x3_nD900_nW200_nV225.get_data(seed=99)
			TestData.name = 'Test'
			lap = 500


		if pred_method == 'lin_reg':
			r2_Train, rmse_Train, r2_Test, rmse_Test = predict_with_lda_plus_linreg(job_path,TrainData,TestData,lap=lap)
		elif pred_method == 'lin_reg_on_pi':
			r2_Train, rmse_Train, r2_Test, rmse_Test = predict_with_lda_plus_linreg_on_pi(job_path,TrainData,TestData,lap=lap)
		
		elif pred_method == 'slda_orig':
			r2_Train, rmse_Train, r2_Test, rmse_Test = predict_with_orig_slda(job_path,TrainData,TestData,lap=lap)
		elif pred_method == 'slda_alt':
			r2_Train, rmse_Train, r2_Test, rmse_Test = predict_with_alt_slda(job_path,TrainData,TestData,lap=lap)

		#print 'r2_Train: ', r2_Train
		#print 'r2_Test: ', r2_Test

		all_r2_Train.append(r2_Train)
		all_rmse_Train.append(rmse_Train)
		all_r2_Test.append(r2_Test)
		all_rmse_Test.append(rmse_Test)
	
		print 'TRAIN'
		print 'R2: ', np.mean(all_r2_Train), np.max(all_r2_Train), np.min(all_r2_Train)
		print 'RMSE: ', np.mean(all_rmse_Train), np.max(all_rmse_Train), np.min(all_rmse_Train)

		print 'TEST'
		print 'R2: ', np.mean(all_r2_Test), np.max(all_r2_Test), np.min(all_r2_Test)
		print 'RMSE: ', np.mean(all_rmse_Test), np.max(all_rmse_Test), np.min(all_rmse_Test)
	
