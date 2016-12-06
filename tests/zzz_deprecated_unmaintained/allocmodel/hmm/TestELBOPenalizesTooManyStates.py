import numpy as np
import unittest
import sys

import bnpy
import BarsK10V900
from bnpy.allocmodel import HDPHMM

sys.path.append('../topics/HDP-point-estimation')
import HDPPE
HDPPE = HDPPE.HDPPE


def pprintProbVector(xvec, fmt='%.4f', Kmax=10):
  xvec = np.asarray(xvec)
  if xvec.ndim == 0:
    xvec = np.asarray([xvec])
  if xvec.size > Kmax:
      s_start = ' '.join([fmt % (x) for x in xvec[:Kmax/2]])
      s_end = ' '.join([fmt % (x) for x in xvec[-Kmax/2:]])
      return s_start + '...' + s_end
  else:
      return ' '.join([fmt % (x) for x in xvec[:Kmax]])


def resp2ELBO_HDPHMM(Data, resp, 
                     gamma=10, alpha=0.5, initprobs='fromdata',
                     verbose=0,
                     **kwargs):
  ## Create a new HDPHMM
  amodel = HDPHMM('VB', dict(alpha=alpha, gamma=gamma, startAlpha=alpha))

  K = resp.shape[1]
  if initprobs == 'fromdata':
    init_probs = np.sum(resp, axis=0) + gamma
  elif initprobs == 'uniform':
    init_probs = np.ones(K) / K
  init_probs = init_probs / np.sum(init_probs)
  init_probs = np.hstack([0.99*init_probs, 0.01])

  rho = bnpy.util.StickBreakUtil.beta2rho(init_probs, K)
  amodel.rho = rho

  ## Create a local params dict and suff stats
  ## These will remain fixed, used to update amodel
  LP = dict(resp=resp)
  LP = amodel.initLPFromResp(Data, LP, limitMemoryLP=0)

  ## Loop over alternating updates to local and global parameters
  ## until we've converged 
  prevELBO = -1 * np.inf
  ELBO = 0
  while np.abs(ELBO - prevELBO) > 1e-7:
      Ebeta = amodel.get_active_comp_probs()

      SS = amodel.get_global_suff_stats(Data, LP, doPrecompEntropy=0)
      amodel.update_global_params(SS)

      prevELBO = ELBO
      ELBO = amodel.calc_evidence(Data, SS, LP)
      if verbose:
          print '%.6f  %s' % (ELBO, pprintProbVector(Ebeta))
  return ELBO


def resp2ELBO_PointEstimate(Data, resp, 
                     gamma=10, alpha=0.5, initprobs='fromdata',
                     verbose=0,
                     **kwargs):
  ## Create a new HDPHMM
  ## with initial global params set so we have a uniform distr over topics
  amodelPE = HDPPE('VB', dict(alpha=alpha, gamma=gamma))
  amodel = HDPHMM('VB', dict(alpha=alpha, gamma=gamma, startAlpha=alpha))

  K = resp.shape[1]
  if initprobs == 'fromdata':
    init_probs = np.sum(resp, axis=0) + gamma
  elif initprobs == 'uniform':
    init_probs = np.ones(K) / K
  init_probs = init_probs / np.sum(init_probs)
  init_probs = np.hstack([0.99*init_probs, 0.01])

  rho = bnpy.util.StickBreakUtil.beta2rho(init_probs, K)
  amodelPE.uhat = rho


  ## Create a local params dict and suff stats
  ## These will remain fixed, used to update amodle
  LP = dict(resp=resp)
  LP = amodel.initLPFromResp(Data, LP)
  SS = amodel.get_global_suff_stats(Data, LP, doPrecompEntropy=0)

  DocTopicCount=np.vstack(
      [SS.StartStateCount[np.newaxis,:], 
       SS.TransStateCount])
  resp = np.maximum(resp, 1e-100)
  LPPE = dict(DocTopicCount=DocTopicCount, resp=resp)
  LPPE = amodelPE.updateLPGivenDocTopicCount(LPPE, LPPE['DocTopicCount'])

  ## Loop over alternating updates to local and global parameters
  ## until we've converged 
  prevELBO = -1 * np.inf
  ELBO = 0
  while np.abs(ELBO - prevELBO) > 1e-7:
      Ebeta = amodelPE.get_active_comp_probs()

      LP = amodelPE.updateLPGivenDocTopicCount(LPPE, LPPE['DocTopicCount'])
      SS = amodelPE.get_global_suff_stats(Data, LP, doPrecompEntropy=0)
      SS.setField('nDoc', DocTopicCount.shape[0], dims=None)

      amodelPE.update_global_params(SS)
      prevELBO = ELBO
      ELBO = amodelPE.calc_evidence(Data, SS, LP)
      if verbose:
          print '%.6f  %s' % (ELBO, pprintProbVector(Ebeta))
  return ELBO


def calcBernELBOFromResp(Data, resp, lam1=0.1, lam0=0.1, **kwargs):
    obsModel = bnpy.obsmodel.BernObsModel('VB', 
        lam1=lam1, lam0=lam0, Data=Data)
    LP = dict(resp=resp)
    SS = obsModel.get_global_suff_stats(Data, None, LP)
    obsModel.update_global_params(SS)
    return obsModel.calcELBO_Memoized(SS=SS)


def calcGaussELBOFromResp(Data, resp, nu=1.0, sF=1.0, **kwargs):
    obsModel = bnpy.obsmodel.GaussObsModel('VB', 
        sF=sF, ECovMat='eye', nu=nu, Data=Data)
    LP = dict(resp=resp)
    SS = obsModel.get_global_suff_stats(Data, None, LP)
    obsModel.update_global_params(SS)
    return obsModel.calcELBO_Memoized(SS=SS)

def makeGaussDataAndResp(seed=123, nDocTotal=1, N1=20, N2=20, **kwargs):
    PRNG = np.random.RandomState(0)
    X1 = 0.1 * PRNG.randn(N1)
    X2 = 0.2 * PRNG.randn(N2)
    X = np.vstack([X1[:,np.newaxis], X2[:,np.newaxis]])
    Data = bnpy.data.GroupXData(X=X, doc_range=[0,N1+N2])
    trueResp = np.zeros((X.shape[0], 2))
    trueResp[:X1.shape[0], 0] = 1.0
    trueResp[X1.shape[0]:, 1] = 1.0
    manyResp = np.eye(X.shape[0])
    return Data, trueResp, manyResp

def makeBernDataAndResp(seed=123, nDocTotal=1, N1=20, N2=20, **kwargs):
    PRNG = np.random.RandomState(0)
    X1 = PRNG.rand(N1) < 0.2
    X2 = PRNG.rand(N2) < 0.4
    X = np.vstack([X1[:,np.newaxis], X2[:,np.newaxis]])
    Data = bnpy.data.GroupXData(X=X, doc_range=[0,N1+N2])
    trueResp = np.zeros((X.shape[0], 2))
    trueResp[:X1.shape[0], 0] = 1.0
    trueResp[X1.shape[0]:, 1] = 1.0
    manyResp = np.eye(X.shape[0])
    return Data, trueResp, manyResp

def doExperiment(obsModel='Bern', **kwargs):
    if obsModel == 'Gauss':
        Data, trueResp, manyResp = makeGaussDataAndResp(**kwargs)
        print 'Gauss       obs elbo:'
        print '  true: %.6f' % calcGaussELBOFromResp(Data, trueResp, **kwargs)
        print '  many: %.6f' % calcGaussELBOFromResp(Data, manyResp, **kwargs)
    else:
        Data, trueResp, manyResp = makeBernDataAndResp(**kwargs)
        print 'Bern       obs elbo:'
        print '  true: %.6f' % calcBernELBOFromResp(Data, trueResp, **kwargs)
        print '  many: %.6f' % calcBernELBOFromResp(Data, manyResp, **kwargs)
    print np.squeeze(Data.X)

    trueELBO_pe = 

    print 'HDPPointEst alloc elbo:'
    print '  true: %.6f' % resp2ELBO_PointEstimate(Data, trueResp, **kwargs)
    print '  many: %.6f' % resp2ELBO_PointEstimate(Data, manyResp, **kwargs)
    print 'HDPHMM      alloc elbo:'
    print '  true: %.6f' % resp2ELBO_HDPHMM(Data, trueResp, **kwargs)
    print '  many: %.6f' % resp2ELBO_HDPHMM(Data, manyResp, **kwargs)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--obsModel', default='Bern')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=10)
    parser.add_argument('--initprobs', type=str, default='fromdata')
    parser.add_argument('--N1', type=int, default=10)
    parser.add_argument('--N2', type=int, default=10)
    parser.add_argument('--sF', type=float, default=1)
    parser.add_argument('--nu', type=float, default=1)
    args = parser.parse_args()

    doExperiment(**args.__dict__)
