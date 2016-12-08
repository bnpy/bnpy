'''
StandardNormalK1.py

Simple toy dataset from standard normal distribution.
'''

import numpy as np

from bnpy.data import XData


def get_data(seed=8675309, nObsTotal=25000, **kwargs):
    ''' Create and return toy dataset from 1D standard normal distribution.

    Args
    -------
    seed : integer seed for random number generator,
        used for actually *generating* the data
    nObsTotal : total number of observations for the dataset.

    Returns
    -------
    Data : bnpy XData object, with nObsTotal observations
    '''
    X, TrueZ = generate_data(seed, nObsTotal)
    Data = XData(X=X, TrueZ=TrueZ)
    Data.name = get_short_name()
    Data.summary = get_data_info()
    return Data


def get_data_info():
    return 'Standard Normal Data. All from one true cluster.'


def get_short_name():
    return "StandardNormalK1"


def generate_data(seed, nObsTotal):
    PRNG = np.random.RandomState(seed)
    X = PRNG.randn(nObsTotal, 1)
    TrueZ = np.ones(nObsTotal)
    return X, TrueZ
