import numpy as np
import bnpy

def makeData(K=10, N=1000, D=10,
             **kwargs):
    ''' Create bag-of-words toy dataset for topic modeling
    '''
    PRNG = np.random.RandomState(0)
    X = PRNG.randn(N, D)
    Data = bnpy.data.XData(X=X)
    return Data

def makeModel(Data=None, K=10, **kwargs):
    # Create model and initialize global parameters
    APriorSpec = dict(gamma0=10)
    OPriorSpec = dict(sF=1.0, ECovMat='eye')
    hmodel = bnpy.HModel.CreateEntireModel(
        'VB', 'DPMixtureModel', 'Gauss', APriorSpec, OPriorSpec,
        Data=Data)
    hmodel.init_global_params(Data, K=K, initname='randexamples')
    return hmodel


def pprintProblemSpecStr(K=10, N=5000, D=10, **kwargs):
    return 'N=%d\nD=%d\nK=%d\n' % (N, D, K)
