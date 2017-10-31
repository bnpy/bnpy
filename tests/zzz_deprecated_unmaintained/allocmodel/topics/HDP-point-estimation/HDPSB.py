'''
HDPSB.py
Bayesian nonparametric admixture model via the Hierarchical Dirichlet Process.
Uses a direct construction that maintains K active components.

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
rho : 1D array, size K
omega : 1D array, size K

q(u_k) = Beta(rho[k]*omega[k], (1-rho[k])*omega[k])

References
-------
TODO
Latent Dirichlet Allocation, by Blei, Ng, and Jordan
introduces a classic admixture model with Dirichlet-Mult observations.
'''
import numpy as np

from bnpy.allocmodel.AllocModel import AllocModel
from bnpy.suffstats import SuffStatBag
from bnpy.util import digamma, gammaln
from bnpy.util import NumericUtil, as1D

import OptimizerHDPSB as OptimHDPSB
import LocalUtil

class HDPSB(AllocModel):
  def __init__(self, inferType, priorDict=None):
    if inferType == 'EM':
      raise ValueError('HDPSB cannot do EM.')
    self.inferType = inferType
    self.K = 0
    if priorDict is None:
      self.set_prior()
    else:
      self.set_prior(**priorDict)

  def get_keys_for_memoized_local_params(self):
    ''' Return list of string names of the LP fields
        that moVB needs to memoize across visits to a particular batch
    '''
    return ['DocTopicCount']
      
  def get_active_comp_probs(self):
    ''' Return K vector of appearance probabilities for each of the K comps
    '''
    return self.E_beta_active()

  def E_beta_active(self):
    ''' Return vector beta of appearance probabilities for active components
    '''
    if not hasattr(self, 'Ebeta'):
      self.Ebeta = self.rho.copy()
      self.Ebeta[1:] *= np.cumprod(1 - self.rho[:-1])
    return self.Ebeta

  def E_beta_and_betagt(self):
    ''' Return vectors beta, beta_gt that define conditional appearance probs

        Returns
        --------
        beta : 1D array, size K
        beta_gt : 1D array, size K
    '''
    if not 'Ebeta_gt' in self.__dict__:
      self.Ebeta = self.E_beta_active()
      self.Ebeta_gt = gtsum(self.Ebeta) + (1-np.sum(self.Ebeta))
    return self.Ebeta, self.Ebeta_gt 

  def ClearCache(self):
    if hasattr(self, 'Ebeta'):
      del self.Ebeta
    if hasattr(self, 'Ebeta_gt'):
      del self.Ebeta_gt

  def set_prior(self, gamma=1.0, alpha=1.0, **kwargs):
    self.alpha = float(alpha)
    self.gamma = float(gamma)

  def to_dict(self):
    return dict(rho=self.rho, omega=self.omega)              

  def from_dict(self, Dict):
    self.inferType = Dict['inferType']
    self.K = Dict['K']
    self.rho = as1D(Dict['rho'])
    self.omega = as1D(Dict['omega'])

  def get_prior_dict(self):
    return dict(alpha=self.alpha, gamma=self.gamma,
                K=self.K,
                inferType=self.inferType)
    
  def get_info_string(self):
    ''' Returns human-readable name of this object
    '''
    return 'HDP model with K=%d active comps. gamma=%.2f. alpha=%.2f' \
            % (self.K, self.gamma, self.alpha)
    

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

  def calcLogPrActiveCompsForDoc(self, DocTopicCount_d, out):
    ''' Calculate log prob of each of the K active topics given doc-topic counts

        Returns
        -------
        logp : 1D array, size K
               logp[k] gives probability of topic k in provided doc
    '''
    Ebeta, Ebeta_gt = self.E_beta_and_betagt()

    eta1 = DocTopicCount_d + self.alpha * Ebeta
    eta0 = gtsum(DocTopicCount_d) + self.alpha * Ebeta_gt

    digammaBoth = digamma(eta1+eta0)
    ElogVd = digamma(eta1) - digammaBoth
    Elog1mVd = digamma(eta0) - digammaBoth

    out[:] = ElogVd
    out[1:] += np.cumsum(Elog1mVd[:-1])
    return out

  def calcLogPrActiveComps_Fast(self, DocTopicCount, activeDocs=None, LP=dict(),
                                      out=None):
    ''' Calculate log prob of each active topic for each active document
    '''
    Ebeta, Ebeta_gt = self.E_beta_and_betagt()

    if activeDocs is None:
      activeDocTopicCount = DocTopicCount
    else:
      activeDocTopicCount = np.take(DocTopicCount, activeDocs, axis=0)

    if 'eta1' in LP:
      LP['eta1'][activeDocs] = activeDocTopicCount + self.alpha * Ebeta
    else:
      LP['eta1'] = DocTopicCount + self.alpha * Ebeta

    if 'eta0' in LP:
      LP['eta0'][activeDocs] = gtsum(activeDocTopicCount) \
                               + self.alpha * Ebeta_gt
    else:
      LP['eta0'] = gtsum(DocTopicCount) + self.alpha * Ebeta_gt

    eta1 = LP['eta1']
    eta0 = LP['eta0']
    digammaBoth = digamma(eta1+eta0)
    ElogVd = digamma(eta1) - digammaBoth
    Elog1mVd = digamma(eta0) - digammaBoth
    if out is None:
      ElogPi = ElogVd.copy()
    else:
      ElogPi = out
      ElogPi[activeDocs] = ElogVd[activeDocs]
    if activeDocs is None:
      ElogPi[:,1:] += np.cumsum(Elog1mVd[:,:-1], axis=1)
    else:
      ElogPi[activeDocs,1:] += np.cumsum(Elog1mVd[activeDocs,:-1], axis=1)
    return ElogPi


  def updateLPGivenDocTopicCount(self, LP, DocTopicCount):
    ''' Update all local parameters, given topic counts for all docs in set.

        Returns
        --------
        LP : dict of local params, with updated fields
        * eta1, eta0
        * ElogVd, Elog1mVd
        * ElogPi
    '''
    DocTopicCount_gt = gtsum(DocTopicCount)

    Ebeta, Ebeta_gt = self.E_beta_and_betagt()

    eta1 = DocTopicCount + self.alpha * Ebeta
    eta0 = DocTopicCount_gt + self.alpha * Ebeta_gt

    ## Double-check!
    Ebeta2, Ebeta_gt2 = self.E_beta_and_betagt()
    assert np.allclose(Ebeta2, Ebeta)
    assert np.allclose(Ebeta_gt2, Ebeta_gt)

    digammaBoth = digamma(eta1+eta0)
    ElogV = digamma(eta1) - digammaBoth
    Elog1mV = digamma(eta0) - digammaBoth
    ElogPi = ElogV.copy()
    ElogPi[:, 1:] += np.cumsum(Elog1mV[:, :-1], axis=1)

    LP['DocTopicCount_gt'] = DocTopicCount_gt
    LP['eta1'] = eta1
    LP['eta0'] = eta0
    LP['ElogV'] = ElogV
    LP['Elog1mV'] = Elog1mV
    LP['ElogPi'] = ElogPi
    return LP

  def initLPFromResp(self, Data, LP):
    ''' Obtain initial local params for initializing this model.
    '''
    resp = LP['resp']
    K = resp.shape[1]
    DocTopicCount = np.zeros( (Data.nDoc, K))
    for d in range(Data.nDoc):
      start = Data.doc_range[d]
      stop = Data.doc_range[d+1]
      if hasattr(Data, 'word_count'):
        DocTopicCount[d,:] = np.dot(Data.word_count[start:stop],
                                    resp[start:stop,:])
      else:
        DocTopicCount[d,:] = np.sum(resp[start:stop,:], axis=0)
    DocTopicCount_gt = gtsum(DocTopicCount)

    remMass = np.minimum(0.1, 1.0/(K*K))
    Ebeta = (1 - remMass) / float(K) * np.ones(K)
    Ebeta_gt = gtsum(Ebeta) + remMass

    eta1 = DocTopicCount + self.alpha * Ebeta
    eta0 = DocTopicCount_gt + self.alpha * Ebeta_gt

    digammaBoth = digamma(eta1+eta0)
    ElogV = digamma(eta1) - digammaBoth
    Elog1mV = digamma(eta0) - digammaBoth
    ElogPi = ElogV.copy()
    ElogPi[:, 1:] += np.cumsum(Elog1mV[:, :-1], axis=1)

    LP['DocTopicCount'] = DocTopicCount
    LP['DocTopicCount_gt'] = DocTopicCount_gt
    LP['eta1'] = eta1
    LP['eta0'] = eta0
    LP['ElogV'] = ElogV
    LP['Elog1mV'] = Elog1mV
    LP['ElogPi'] = ElogPi
    return LP

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
    rho, omega = self._find_optimum_rhoomega(SS, **kwargs)
    self.rho = rho
    self.omega = omega
    self.K = SS.K
    self.ClearCache()

  def _find_optimum_rhoomega(self, SS, **kwargs):
    ''' Run numerical optimization to find optimal rho, omega parameters

        Args
        --------
        SS : bnpy SuffStatBag, with K components

        Returns
        --------
        rho : 1D array, length K
        omega : 1D array, length K
    '''
    if hasattr(self, 'rho') and self.rho.size == SS.K:
      initrho = self.rho
      initomega = self.omega
    else:
      initrho = None   # default initialization
      initomega = None

    try:
      rho, omega, f, Info = OptimHDPSB.find_optimum_multiple_tries(
                                        sumLogVd=SS.sumLogVd,
                                        sumLog1mVd=SS.sumLog1mVd,
                                        nDoc=SS.nDoc,
                                        gamma=self.gamma, alpha=self.alpha,
                                        initrho=initrho, initomega=initomega)
    except ValueError as error:
      if str(error).count('FAILURE') == 0:
        raise error
      if hasattr(self, 'rho') and self.rho.size == SS.K:
        Log.error('***** Optim failed. Remain at cur val. ' + str(error))
        rho = self.rho
        omega = self.omega
      else:
        Log.error('***** Optim failed. Set to default init. ' + str(error))
        omega = (1 + self.gamma) * np.ones(SS.K)
        rho = OptimHDPSB.create_initrho(K)
    return rho, omega


  ####################################################### Set Global Params
  #######################################################
  def init_global_params(self, Data, K=0, **kwargs):
    ''' Initialize rho, omega to reasonable values
    '''
    self.K = K
    self.rho = OptimHDPSB.create_initrho(K)
    self.omega = (1.0 + self.gamma) * np.ones(K)
    self.ClearCache()

  def set_global_params(self, hmodel=None, rho=None, omega=None, 
                              **kwargs):
    ''' Set rho, omega to provided values.
    '''
    if hmodel is not None:
      self.K = hmodel.allocModel.K
      if hasattr(hmodel.allocModel, 'rho'):
        self.rho = hmodel.allocModel.rho
        self.omega = hmodel.allocModel.omega
      else:
        raise AttributeError('Unrecognized hmodel')
    elif rho is not None and omega is not None:
      self.rho = rho
      self.omega = omega
      self.K = omega.size
    else:
      self._set_global_params_from_scratch(**kwargs)
    self.ClearCache()


  def _set_global_params_from_scratch(self, beta=None, topic_prior=None,
                                            Data=None, **kwargs):
    ''' Set rho, omega to values that reproduce provided appearance probs
    '''
    if topic_prior is not None:
      beta = topic_prior / topic_prior.sum()
    if beta is not None:
      Ktmp = beta.size
      rem = np.minimum(0.05, 1./(Ktmp))
      beta = np.hstack([np.squeeze(beta), rem])
      beta = beta/np.sum(beta)
    else:
      raise ValueError('Bad parameters. Vector beta not specified.')
    self.K = beta.size - 1
    self.rho, self.omega = self._convert_beta2rhoomega(beta, Data.nDoc)
    assert self.rho.size == self.K
    assert self.omega.size == self.K

  def _convert_beta2rhoomega(self, beta, nDoc=10):
    ''' Find vectors rho, omega that are probable given beta

        Returns
        --------
        rho : 1D array, size K
        omega : 1D array, size K
    '''
    assert abs(np.sum(beta) - 1.0) < 0.001
    rho = OptimHDPSB.beta2rho(beta, self.K)
    omega = (nDoc + self.gamma) * np.ones(rho.size)
    return rho, omega


  ####################################################### Calc ELBO
  #######################################################
  def calc_evidence(self, Data, SS, LP, **kwargs):
    ''' Calculate ELBO objective 
    '''
    UandcV_global = self.E_logpU_logqU_c(SS)
    V_global = self.E_logpV__global(SS)
    if SS.hasELBOTerms():
      ElogqZ = SS.getELBOTerm('ElogqZ')
      VZlocal = SS.getELBOTerm('VZlocal')
    else:
      ElogqZ = self.E_logqZ(Data, LP)
      VZlocal = self.E_logpVZ_logqV(Data, LP)
    return UandcV_global + V_global + VZlocal - np.sum(ElogqZ)

  def E_logqZ(self, Data, LP):
    ''' Calculate E[ log q(z)] for each active topic

        Returns
        -------
        ElogqZ : 1D array, size K
    '''
    if hasattr(Data, 'word_count'):
      return NumericUtil.calcRlogRdotv(LP['resp'], Data.word_count)
    else:
      return NumericUtil.calcRlogR(LP['resp'])

  def E_logpV__global(self, SS):
    ''' Calculate the part of E[ log p(v) ] that depends on global topic probs

        Returns
        --------
        Elogstuff : real scalar
    ''' 
    Ebeta, Ebeta_gt = self.E_beta_and_betagt()
    return np.inner(self.alpha * Ebeta, SS.sumLogVd) \
           + np.inner(self.alpha * Ebeta_gt, SS.sumLog1mVd)

  def E_logpVZ_logqV(self, Data, LP):
    ''' Calculate E[ log p(v) + log p(z) - log q(v) ]

        Returns
        -------
        Elogstuff : real scalar
    '''
    cDiff = -1 * c_Beta(LP['eta1'], LP['eta0'])
    ONcoef = LP['DocTopicCount']
    OFFcoef = LP['DocTopicCount_gt']
    logBetaPDF = np.sum((ONcoef - LP['eta1']) * LP['ElogV']) \
                 + np.sum((OFFcoef - LP['eta0']) * LP['Elog1mV'])
    return cDiff + np.sum(logBetaPDF)

  def E_logpU_logqU_c(self, SS):
    ''' Calculate E[ log p(u) - log q(u) ]

        Returns
        ---------
        Elogstuff : real scalar
    '''
    g1 = self.rho * self.omega
    g0 = (1-self.rho) * self.omega
    digammaBoth = digamma(g1+g0)
    ElogU = digamma(g1) - digammaBoth
    Elog1mU = digamma(g0) - digammaBoth

    ONcoef = SS.nDoc + 1.0 - g1
    OFFcoef = SS.nDoc * OptimHDPSB.kvec(self.K) + self.gamma - g0

    cDiff = SS.K * c_Beta(1, self.gamma) - c_Beta(g1, g0)
    logBetaPDF = np.inner(ONcoef, ElogU) \
                 + np.inner(OFFcoef, Elog1mU)
    return cDiff + logBetaPDF

def gtsum(Nvec):
  ''' Calculate new vector where each entry k holds the sum of Nvec[k+1:] 

      Example
      --------
      >> gtsum([5, 6, 10])
      [16, 10, 0]
  '''
  if Nvec.ndim == 1:
    Ngt = np.cumsum(Nvec[::-1])[::-1]
    Ngt[:-1] = Ngt[1:]
    Ngt[-1] = 0
    return Ngt
    #return np.hstack([Ngt[1:], 0])
  elif Nvec.ndim == 2:
    Ngt = np.fliplr(np.cumsum(np.fliplr(Nvec), axis=1))
    zeroCol = np.zeros((Ngt.shape[0],1))
    return np.hstack([Ngt[:, 1:], zeroCol])

def c_Beta(a1, a0):
  ''' Evaluate cumulant function of the Beta distribution

      When input is vectorized, we compute sum over all entries.

      Returns
      -------
      c : scalar real
  '''
  return np.sum(gammaln(a1 + a0)) - np.sum(gammaln(a1)) - np.sum(gammaln(a0))  
