'''
HDPPE.py
Bayesian nonparametric admixture model via the Hierarchical Dirichlet Process.
Uses a direct construction that maintains K active components.
with **point-estimation** for the top-level stick-lengths.

Attributes
-------
K : # of components
gamma : scalar positive real, global concentration 
alpha : scalar positive real, document-level concentration param

Local Model Parameters (document-specific)
--------
z :  one-of-K topic assignment indicator for tokens
     z_{dn} : binary indicator vector for assignment of token n in document d
              z_{dnk} = 1 iff assigned to topic k, 0 otherwise.

v : document-specific stick-breaking lengths for each active topic 
     v1 : 2D array, size D x K
     v0 : 2D array, size D x K

Local Variational Parameters
--------
resp :  q(z_dn) = Categorical( z_dn | resp_{dn1}, ... resp_{dnK} )
eta1, eta0 : q(v_d) = Beta( eta1[d,k], eta0[d,k])

Global Model Parameters (shared across all documents)
--------
uhat : 1D array, size K
q(u_k) = Point Mass at uhat[k]

References
-------
TODO
Latent Dirichlet Allocation, by Blei, Ng, and Jordan
introduces a classic admixture model with Dirichlet-Mult observations.
'''
import numpy as np

from HDPSB import HDPSB, c_Beta, gtsum

from bnpy.suffstats import SuffStatBag
from bnpy.util import digamma, gammaln
from bnpy.util import NumericUtil, as1D

import OptimizerHDPPE as OptimHDPPE
import LocalUtil

class HDPPE(HDPSB):

  def E_beta_active(self):
    ''' Return vector beta of appearance probabilities for all active topics
    '''
    beta = self.uhat.copy()
    beta[1:] *= np.cumprod( 1 - self.uhat[:-1])
    return beta

  def to_dict(self):
    return dict(uhat=self.uhat)

  def from_dict(self, Dict):
    self.inferType = Dict['inferType']
    self.K = Dict['K']
    self.uhat = as1D(Dict['uhat'])

  ####################################################### VB Local Step
  ####################################################### (E-step)
  def calc_local_params(self, Data, LP, **kwargs):
    ''' Calculate document-specific quantities (E-step)
         
          Returns
          -------
          LP : local params dict, with fields
          * resp
          * theta
          * ElogPi
          * DocTopicCount
    '''
    LP = LocalUtil.calcLocalParams(Data, LP, self, **kwargs)
    assert 'resp' in LP
    assert 'DocTopicCount' in LP
    return LP

  ### Inherited from HDPSB
  # def calcLogPrActiveCompsForDoc(self, DocTopicCount_d)
  # def updateLPGivenDocTopicCount(self, LP, DocTopicCount)
  # def initLPFromResp(self, Data, LP)

  ####################################################### Suff Stat Calc
  ####################################################### 
  def get_global_suff_stats(self, Data, LP, doPrecompEntropy=None, **kwargs):
    ''' Calculate sufficient statistics.
    '''
    resp = LP['resp']
    _, K = resp.shape
    SS = SuffStatBag(K=K, D=Data.get_dim())
    SS.setField('nDoc', Data.nDoc, dims=None)
    SS.setField('sumLogVd', np.sum(LP['ElogV'], axis=0), dims='K')
    SS.setField('sumLog1mVd', np.sum(LP['Elog1mV'], axis=0), dims='K')

    if doPrecompEntropy:
      ElogqZ = self.E_logqZ(Data, LP)
      VZlocal = self.E_logpVZ_logqV(Data, LP)
      SS.setELBOTerm('ElogqZ', ElogqZ, dims='K')
      SS.setELBOTerm('VZlocal', VZlocal, dims=None)
    return SS

  ####################################################### VB Global Step
  #######################################################
  def update_global_params_VB(self, SS, rho=None, **kwargs):
    ''' Update global parameters.
    '''
    uhat = self._find_optimum_uhat(SS, **kwargs)
    self.uhat = uhat
    self.K = SS.K
    self.ClearCache()

  def _find_optimum_uhat(self, SS, **kwargs):
    ''' Run numerical optimization to find optimal uhat point estimate

        Args
        --------
        SS : bnpy SuffStatBag, with K components

        Returns
        --------
        uhat : 1D array, length K
    '''
    if hasattr(self, 'uhat') and self.uhat.size == SS.K:
      inituhat = self.uhat
    else:
      inituhat = None

    try:
      uhat, f, Info = OptimHDPPE.find_optimum_multiple_tries(
                                        sumLogVd=SS.sumLogVd,
                                        sumLog1mVd=SS.sumLog1mVd,
                                        nDoc=SS.nDoc,
                                        gamma=self.gamma, alpha=self.alpha,
                                        inituhat=inituhat)

    except ValueError as error:
      if str(error).count('FAILURE') == 0:
        raise error
      if inituhat is not None:
        Log.error('***** Optim failed. Remain at cur val. ' + str(error))
        self.uhat = inituhat
      else:
        Log.error('***** Optim failed. Set to default init. ' + str(error))
        uhat = OptimHDPPE.create_inituhat(K)
    return uhat

  ####################################################### Set Global Params
  #######################################################
  def init_global_params(self, Data, K=0, **kwargs):
    self.K = K
    self.uhat = OptimHDPPE.create_inituhat(K)
    self.ClearCache()

  def set_global_params(self, hmodel=None, 
                              uhat=None,
                              **kwargs):
    if hmodel is not None:
      self.K = hmodel.allocModel.K
      if hasattr(hmodel.allocModel, 'rho'):
        self.rho = hmodel.allocModel.rho
      elif hasattr(hmodel.allocModel, 'uhat'):
        self.uhat = hmodel.allocModel.uhat
      else:
        raise AttributeError('Unrecognized hmodel')
    elif uhat is not None:
      self.uhat = uhat
      self.K = uhat.size
    else:
      self._set_global_params_from_scratch(**kwargs)
    self.ClearCache()

  def _set_global_params_from_scratch(self, beta=None, probs=None,
                                            Data=None, nDoc=None, **kwargs):
    ''' Set uhat to values that reproduce provided appearance probs
    '''
    if nDoc is None:
      nDoc = Data.nDoc
    if nDoc is None:
      raise ValueError('Bad parameters. nDoc not specified.')
    if probs is not None:
      beta = probs / probs.sum()
    if beta is not None:
      Ktmp = beta.size
      rem = np.minimum(0.05, 1./(Ktmp))
      beta = np.hstack([np.squeeze(beta), rem])
      beta = beta/np.sum(beta)
    else:
      raise ValueError('Bad parameters. Vector beta not specified.')
    self.K = beta.size - 1
    self.uhat = self._convert_beta2uhat(beta)
    assert self.uhat.size == self.K

  def _convert_beta2uhat(self, beta):
    ''' Find stick-lengths uhat that best recreate provided appearance probs beta

        Returns
        --------
        uhat : 1D array, size K
    '''
    assert abs(np.sum(beta) - 1.0) < 0.001
    return OptimHDPPE.beta2rho(beta, self.K)


  ####################################################### Calc ELBO
  #######################################################
  def calc_evidence(self, Data, SS, LP, **kwargs):
    ''' Calculate ELBO objective 
    '''
    cV_global = SS.nDoc * self.E_c_alphabeta()
    U_global = self.E_logpU()
    V_global = self.E_logpV__global(SS)
    if SS.hasELBOTerms():
      ElogqZ = SS.getELBOTerm('ElogqZ')
      VZlocal = SS.getELBOTerm('VZlocal')
    else:
      ElogqZ = self.E_logqZ(Data, LP)
      VZlocal = self.E_logpVZ_logqV(Data, LP)
    return U_global + cV_global + V_global + VZlocal - np.sum(ElogqZ)

  ## Inherited from HDPSB
  # def E_logqZ(self, Data, LP):
  # def E_logpVZ_logqV(self, Data, LP):
  # def E_logpV__global(self, SS)

  def E_logpU(self):
    ''' Calculate E[ log p(u) ]
    '''
    Elog1mU = np.log(1-self.uhat)
    return self.K * c_Beta(1, self.gamma) \
           + np.sum((self.gamma - 1) * Elog1mU)

  def E_c_alphabeta(self):
    ''' Calculate E[ \sum_k c_B( alpha beta_k, alpha beta_gtk) ]
    '''
    Ebeta, Ebeta_gt = self.E_beta_and_betagt()
    return c_Beta( self.alpha * Ebeta,
                   self.alpha * Ebeta_gt)

