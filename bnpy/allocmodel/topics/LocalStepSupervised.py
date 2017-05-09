import numpy as np
import copy
import time, sys, random

from scipy.special import digamma, gammaln
import scipy.sparse

import LocalStepLogger
from bnpy.util import NumericUtil

try:
    import theano.tensor as T
    from theano import function
except:
    pass

from scipy.optimize import fmin_l_bfgs_b

def calcNumDocFromSlice(Data, cslice):
    if cslice[1] is None:
        nDoc = Data.nDoc
    else:
        nDoc = cslice[1] - cslice[0]
    return int(nDoc)


class RespOptimizerTheano:
    def __init__(self):
        self.resp = T.dmatrix('resp')
        self.Nd = T.dscalar('Nd')
        self.Epi = T.dvector('Epi')
        self.Yd = T.dscalar('Yd')
        self.mean = T.dvector('mean')
        self.var = T.dvector('var')
        self.E_outer = T.dmatrix('E_outer')
        self.w_c = T.dcol('w_c')
        self.Lik = T.dmatrix('Lik')

        #Setup the "observation"
        #resp = T.exp(self.resp)
        #resp = self.resp ** 2
        #resp = self.resp
        #resp = T.log(T.exp(self.resp) + 1)
        resp = T.sqrt(self.resp ** 2 + 1) - 1

        resp = resp / T.sum(resp, axis=1, keepdims=True) + 1e-300
        sumN = T.sum(self.w_c * resp, axis=0)
        X = sumN / self.Nd

        #Compute the eta update and lambda from eta
        eta2 = T.sum((X ** 2) * self.var) + T.sum(X * self.mean) ** 2
        eta = T.sqrt(eta2)
        lam = T.tanh(eta / 2.0) / (4.0 * eta)

        #Terms from the regression model
        #term1 = (1.0 / (2 * self.Nd)) * T.sum(sumN * (self.Yd * self.mean))
        term1 = ((self.Yd - 0.5) / self.Nd) * T.sum(sumN * self.mean)
        
        resp_outer = T.outer(sumN, sumN)
        resp_outer_adj = T.nlinalg.alloc_diag(sumN - T.sum(self.w_c * resp ** 2, axis=0))
        resp_outer = resp_outer + resp_outer_adj
        term2 =  -(lam / (self.Nd ** 2)) * T.sum(self.E_outer * resp_outer)

        #P(z|pi) term
        term3 = T.sum(self.Epi * sumN)

        #Entropy Term
        term4 = -T.sum(resp * T.log(resp) * self.w_c)

        #Mixture likelihood term
        term5 = T.sum((self.w_c * resp) * self.Lik)

        obj = -(1 * (term1 + term2) + term3 + term4 + term5)
        grad = T.grad(obj, self.resp)

        self.obj_T = function([self.resp, self.Nd, self.Epi, self.Yd, self.mean, self.var, self.E_outer, self.w_c, self.Lik], obj, on_unused_input='ignore')
        self.grad_T = function([self.resp, self.Nd, self.Epi, self.Yd, self.mean, self.var, self.E_outer, self.w_c, self.Lik], grad, on_unused_input='ignore')
        
    def obj(self, resp, Nd, Epi, Yd, mean, var, E_outer, w_c, Lik):
        resp = resp.reshape((-1, mean.shape[0]))
        val = self.obj_T(resp, Nd, Epi, Yd, mean, var, E_outer, w_c, Lik)
        return val

    def grad(self, resp, Nd, Epi, Yd, mean, var, E_outer, w_c, Lik):
        resp = resp.reshape((-1, mean.shape[0]))
        val = self.grad_T(resp, Nd, Epi, Yd, mean, var, E_outer, w_c, Lik)
        return val.flatten()

    def optimize(self, resp, Nd, Epi, Yd, mean, var, w_c, Lik):
        E_outer = np.outer(mean, mean)
        np.fill_diagonal(E_outer, mean ** 2 + var)

        rshape = resp.shape
        resp = resp.flatten()

        #resp = np.log(resp)
        #resp = np.sqrt(resp)
        #resp = np.log(np.exp(resp) - 0.99)
        resp = np.sqrt((resp + 1) ** 2 - 1)

        obj1 = self.obj(resp, Nd, Epi, Yd, mean, var, E_outer, w_c, Lik)
        result = fmin_l_bfgs_b(self.obj, resp, fprime=self.grad, 
                  args=(Nd, Epi, Yd, mean, var, E_outer, w_c, Lik))
        resp = result[0].reshape(rshape)        
        obj2 = self.obj(resp, Nd, Epi, Yd, mean, var, E_outer, w_c, Lik)

        if obj2 > obj1:
            print 'Objective decreased! Start:', obj1, 'end:', obj2

        #print 'Optimization!', obj1, obj2
        if not np.isfinite(obj2):
            print resp 
            raise

        #resp = np.exp(resp)  
        #resp = resp ** 2
        #resp = np.log(np.exp(resp) + 1) 
        resp = np.sqrt(resp ** 2 + 1) - 1 

        resp = resp / resp.sum(axis=1).reshape((-1,1))
        return resp

try:
    Optimizer = RespOptimizerTheano()
except:
    pass

def updateLPWithResp_Supervised(LP, Data, Lik, Prior, alphaEbeta, alphaEbetaRem, sumRespTilde, cslice=(0, None), nCoordAscentItersLP=1):
    ''' Compute assignment responsibilities given output of local step.

    Args
    ----
    LP : dict
        Has other fields like 'E_log_soft_ev'
    Data : DataObj 
    Lik : 2D array, size N x K
        Will be overwritten and turned into resp.

    Returns
    -------
    LP : dict
        Add field 'resp' : N x K 2D array.
    '''

    LP['resp'] = Lik.copy()
    w_c = np.ones(Lik.shape[0])
    if hasattr(Data, 'word_count'):
        w_c = Data.word_count

    nDoc = calcNumDocFromSlice(Data, cslice)
    slice_start = Data.doc_range[cslice[0]]
    N = LP['resp'].shape[0]
    K = LP['resp'].shape[1]
    if N > Data.doc_range[-1]:
        assert False
    else:
        for d in xrange(nDoc):
            start = Data.doc_range[cslice[0] + d] - slice_start
            stop = Data.doc_range[cslice[0] + d + 1] - slice_start
            
            #Initialize using update without regression
            LP['resp'][start:stop] *= Prior[d]

            #If we don't have labels, skip the optimization
            if not hasattr(Data, 'Y') or Data.Y is None:
                LP['resp'][start:stop] = LP['resp'][start:stop] / np.sum(LP['resp'][start:stop], axis=1).reshape((-1, 1))
                continue
            
            #Setup optimization parameters
            Yd = Data.Y[d]
            Nd = np.sum(w_c[start:stop])

            Epi = Prior[d]

            lLik = LP['E_log_soft_ev'][start:stop]
            mean = np.asarray(LP['w_m'])
            var = np.asarray(LP['w_var'])
            w_c_loc = w_c[start:stop].reshape((-1, 1))

            if len(mean.shape) == 0 or mean.shape[0] < K:
                mean = np.ones(K) * mean.reshape((-1,))[0]
                var = np.ones(K) * var.reshape((-1,))[0]

            resp = LP['resp'][start:stop]
            resp *= np.exp(((Yd - 0.5) / Nd) * mean.reshape((1, -1))) #Adjustment for labels in closed form update (TODO: check!)

            #Iterate between updating z and pi
            for i in xrange(nCoordAscentItersLP):
                resp = resp / np.sum(resp, axis=1).reshape((-1, 1))

                Epi = LP['DocTopicCount'][d, :] + alphaEbeta   #This should be more correct, but leads to decreasing ELBO, so... idk wtf
                Epi = digamma(Epi) - digamma(np.sum(Epi))

                #Run gradient descent for document
                LP['resp'][start:stop] = Optimizer.optimize(resp, Nd, Epi, Yd, mean, var, w_c_loc, lLik)
                LP['DocTopicCount'][d, :] = (LP['resp'][start:stop] * w_c_loc.reshape((-1, 1))).sum(axis=0)
                resp = LP['resp'][start:stop]

    
    np.maximum(LP['resp'], 1e-300, out=LP['resp'])
    return LP

