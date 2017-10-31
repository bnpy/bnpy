''' 
Motorcycle crash data
'''

import numpy as np
import bnpy
import os
import scipy.io

def get_data(**kwargs):
	''' Load fixed dataset from file. 

	Args
	----
	None that matter. Kwargs only for compatibility.

	Returns
	-------
	Data : bnpy.data.WordsData instance
	
	'''
	data_path = '/data/liv/lweiner/motor.mat'
	mat = scipy.io.loadmat(data_path)

	Xtrain = mat['Xtrain']
	Ytrain = mat['Ytrain']

	Data = bnpy.data.XData(X=Xtrain, Y=Ytrain)
	Data.name = "motorcycle_crash_data" 

	return Data

def get_test_data(**kwargs):
	
	data_path = '/data/liv/lweiner/motor.mat'
	mat = scipy.io.loadmat(data_path)

	Xtest = mat['Xtest']
	Ytest = mat['Ytest']

	TestData = bnpy.data.XData(X=Xtest, Y=Ytest)
	TestData.name = "motorcycle_crash_data" 


	return TestData




   
if __name__ == '__main__':
    Data = get_data()
    print(Data.get_text_summary())
    print(Data.get_stats_summary())
