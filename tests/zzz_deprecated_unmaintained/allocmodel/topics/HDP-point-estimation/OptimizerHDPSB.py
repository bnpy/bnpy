'''
OptimizerHDPSB.py

CONSTRAINED Optimization Problem
----------
Variables:
Two K-length vectors
* rho = rho[0], rho[1], rho[2], ... rho[K-1]
* omega = omega[0], omega[1], ... omega[K-1]

Objective:
* argmin ELBO(rho, omega)

Constraints: 
* rho satisfies: 0 < rho[k] < 1
* omega satisfies: 0 < omega[k]
'''

import warnings
import numpy as np
import scipy.optimize
import scipy.io
from scipy.special import gammaln, digamma, polygamma
import datetime
import logging

from bnpy.util.StickBreakUtil import rho2beta_active, beta2rho, sigmoid, invsigmoid
from bnpy.util.StickBreakUtil import forceRhoInBounds, forceOmegaInBounds
from bnpy.util.StickBreakUtil import create_initrho, create_initomega

Log = logging.getLogger('bnpy')

def find_optimum_multiple_tries(sumLogVd=0, sumLog1mVd=0, nDoc=0, 
                                gamma=1.0, alpha=1.0,
                                initrho=None, initomega=None,
                                approx_grad=False,
                                factrList=[1e5, 1e7, 1e9, 1e10, 1e11],
                                **kwargs):
  ''' Estimate vectors rho and omega via gradient descent,
        gracefully using multiple restarts
        with progressively weaker tolerances until one succeeds

      Returns
      --------
      rho : 1D array, length K
      omega : 1D array, length K
      f : scalar value of minimization objective
      Info : dict

      Raises
      --------
      ValueError with FAILURE in message if all restarts fail
  '''
  rhoomega = None
  Info = dict()
  msg = ''
  nOverflow = 0
  for trial, factr in enumerate(factrList):
    try:
      rhoomega, f, Info = find_optimum(sumLogVd, sumLog1mVd, nDoc,
                                       gamma=gamma, alpha=alpha,
                                       initrho=initrho, initomega=initomega,
                                       factr=factr, approx_grad=approx_grad,
                                       **kwargs)
      Info['nRestarts'] = trial
      Info['factr'] = factr
      Info['msg'] = Info['task']
      del Info['grad']
      del Info['task']
      break
    except ValueError as err:
      msg = str(err)
      if str(err).count('FAILURE') > 0:
        # Catch line search problems
        pass
      elif str(err).count('overflow') > 0:
        nOverflow += 1
      else:
        raise err

  if rhoomega is None:
    if initrho is not None:      
      # Last ditch effort, try different initialization
      return find_optimum_multiple_tries(sumLogVd, sumLog1mVd, nDoc, 
                                gamma=gamma, alpha=alpha,
                                initrho=None, initomega=None,
                                approx_grad=approx_grad, **kwargs)
    else:
      raise ValueError(msg)
  Info['nOverflow'] = nOverflow
  rho, omega, K = _unpack(rhoomega)
  return rho, omega, f, Info


def find_optimum(sumLogVd=0, sumLog1mVd=0, nDoc=0, gamma=1.0, alpha=1.0,
                 initrho=None, initomega=None, scaleVector=None,
                 approx_grad=False, factr=1.0e5, **kwargs):
  ''' Run gradient optimization to estimate best parameters rho, omega

      Returns
      --------
      rhoomega : 1D array, length 2*K
      f : scalar value of minimization objective
      Info : dict

      Raises
      --------
      ValueError on an overflow, any NaN, or failure to converge
  '''
  if sumLogVd.ndim > 1:
    sumLogVd = np.squeeze(np.asarray(sumLogVd, dtype=np.float64))
    sumLog1mVd = np.squeeze(np.asarray(sumLog1mVd, dtype=np.float64))

  assert sumLogVd.ndim == 1
  K = sumLogVd.size

  ## Determine initial value
  if initrho is None:
    initrho = create_initrho(K)
  initrho = forceRhoInBounds(initrho)  
  if initomega is None:
    initomega = create_initomega(K, nDoc, gamma)
  initomega = forceOmegaInBounds(initomega)
  assert initrho.size == K
  assert initomega.size == K

  ## Initialize rescaling vector
  if scaleVector is None:
    scaleVector = np.hstack([np.ones(K), np.ones(K)])

  ## Create init vector in unconstrained space
  initrhoomega = np.hstack([initrho, initomega])
  initc = rhoomega2c(initrhoomega, scaleVector=scaleVector)

  ## Define objective function (unconstrained!)
  objArgs = dict(sumLogVd=sumLogVd, sumLog1mVd=sumLog1mVd,
                  nDoc=nDoc, gamma=gamma, alpha=alpha,
                  approx_grad=approx_grad, scaleVector=scaleVector)

  c_objFunc = lambda c: objFunc_unconstrained(c, **objArgs)
  
  ## Run optimization, raising special error on any overflow or NaN issues
  with warnings.catch_warnings():
    warnings.filterwarnings('error', category=RuntimeWarning,
                               message='overflow')
    try:
      chat, fhat, Info = scipy.optimize.fmin_l_bfgs_b(c_objFunc, initc,
                                                  disp=None,
                                                  approx_grad=approx_grad,
                                                  factr=factr,
                                                  **kwargs)
    except RuntimeWarning:
      raise ValueError("FAILURE: overflow!" )
    except AssertionError:
      raise ValueError("FAILURE: NaN/Inf detected!")
  ## Raise error on abnormal warnings (like bad line search)
  if Info['warnflag'] > 1:
    raise ValueError("FAILURE: " + Info['task'])

  ## Convert final answer back to rhoomega (safely)
  Info['init'] = initrhoomega
  rhoomega = c2rhoomega(chat, scaleVector=scaleVector, returnSingleVector=1)
  rhoomega[:K] = forceRhoInBounds(rhoomega[:K])
  return rhoomega, fhat, Info

########################################################### Objective
###########################################################  unconstrained
def objFunc_unconstrained(c, scaleVector=None, approx_grad=False, **kwargs):
  rho, omega = c2rhoomega(c, scaleVector)
  rhoomega = np.hstack([rho, omega])
  if approx_grad:
    f = objFunc_constrained(rhoomega, approx_grad=1, **kwargs)
    return f
  else:
    f, grad = objFunc_constrained(rhoomega, approx_grad=0, **kwargs)
    drodc = np.hstack([rho*(1-rho), omega])
    return f, grad * drodc

def c2rhoomega(c, scaleVector=None, returnSingleVector=False):
  ''' Transform unconstrained variable c into constrained rho, omega

      Returns
      --------
      rho : 1D array, size K, entries between [0, 1]
      omega : 1D array, size K, positive entries

      OPTIONAL: may return as one concatenated vector (length 2K)
  '''
  K = c.size/2
  rho = sigmoid(c[:K])
  omega = np.exp(c[K:])
  if scaleVector is not None:
    rho *= scaleVector[:K]
    omega *= scaleVector[K:]
  if returnSingleVector:
    return np.hstack([rho, omega])
  return rho, omega

def rhoomega2c(rhoomega, scaleVector=None):
  K = rhoomega.size/2
  if scaleVector is not None:
    rhoomega = rhoomega / scaleVector
  return np.hstack([invsigmoid(rhoomega[:K]), np.log(rhoomega[K:])])

########################################################### Objective
###########################################################  constrained
def objFunc_constrained(rhoomega,
                     sumLogVd=0, sumLog1mVd=0, nDoc=0, gamma=1.0, alpha=1.0,
                     approx_grad=False, **kwargs):
  ''' Returns constrained objective function and its gradient

      Args
      -------
      rhoomega := 1D array, size 2*K

      Returns
      -------
      f := -1 * L(rhoomega), 
           where L is ELBO objective function (log posterior prob)
      g := gradient of f
  '''
  assert not np.any(np.isnan(rhoomega))
  assert not np.any(np.isinf(rhoomega))
  rho, omega, K = _unpack(rhoomega)
  
  g1 = rho * omega
  g0 = (1 - rho) * omega
  digammaomega =  digamma(omega)
  assert not np.any(np.isinf(digammaomega))

  Elogu = digamma(g1) - digammaomega
  Elog1mu = digamma(g0) - digammaomega

  if nDoc > 0:
    scale = nDoc
    ONcoef = 1 + (1.0 - g1)/nDoc
    OFFcoef = kvec(K) + (gamma - g0)/nDoc
    P = sumLogVd/nDoc
    Q = sumLog1mVd/nDoc

    ## Calc term related to \sum_{k=1}^K P_k E[\beta_k] + Q_k E[\beta_{>k}]
    cumprod1mrho = np.ones(K)
    cumprod1mrho[1:] = np.cumprod( 1 - rho[:-1])
    rPand1mrQ = rho * P + (1-rho) * Q
    elbo_beta = np.inner(cumprod1mrho, rPand1mrQ) 

  else:
    scale = 1
    ONcoef = 1 - g1
    OFFcoef = gamma - g0
    elbo_beta = 0

  elbo = -1 * c_Beta(g1, g0)/scale \
           + np.inner(ONcoef, Elogu) \
           + np.inner(OFFcoef, Elog1mu) \
           + alpha * elbo_beta

  if approx_grad:
    return -1.0 * elbo

  ## Gradient computation!  
  trigamma_omega = polygamma(1, omega)
  trigamma_g1 = polygamma(1, g1)
  trigamma_g0 = polygamma(1, g0)
  assert np.all(np.isfinite(trigamma_omega))
  assert np.all(np.isfinite(trigamma_g1))


  gradrho = ONcoef * omega * trigamma_g1 \
            - OFFcoef * omega * trigamma_g0
  gradomega = ONcoef * (rho * trigamma_g1 - trigamma_omega) \
            + OFFcoef * ((1-rho) * trigamma_g0 - trigamma_omega)
  if nDoc > 0:
    RMat = calc_drho_dcumprod1mrho(cumprod1mrho, rho, K)
    gB = np.dot(RMat, rPand1mrQ) + cumprod1mrho * (P-Q)
    gradrho += alpha * gB
  grad = np.hstack([gradrho, gradomega])

  return -1.0 * elbo, -1.0 * grad
  
########################################################### Util fcns
###########################################################
def _unpack(rhoomega):
  K = rhoomega.size / 2
  rho = rhoomega[:K]
  omega = rhoomega[-K:]
  return rho, omega, K

def kvec(K):
  ''' Obtain descending vector of [K, K-1, ... 1]

      Returns
      --------
      kvec : 1D array, size K
  '''
  return K + 1 - np.arange(1, K+1)

def c_Beta(g1, g0):
  ''' Calculate cumulant function of the Beta distribution

      Returns
      -------
      c : scalar sum of the cumulants defined by provided parameters
  '''
  return np.sum(gammaln(g1 + g0) - gammaln(g1) - gammaln(g0))

def _v2beta(v):
  ''' Convert to stick-breaking fractions v to probability vector beta
      Args
      --------
      v : K-len vector, rho[k] in interval [0, 1]
      
      Returns
      --------
      beta : K+1-len vector, with positive entries that sum to 1
  '''
  beta = np.hstack([1.0, np.cumprod(1-v)])
  beta[:-1] *= v
  return beta

lowTriIDsDict = dict()
def _get_lowTriIDs(K):
  if K in lowTriIDsDict:
    return lowTriIDsDict[K]
  else:
    ltIDs = np.tril_indices(K)
    lowTriIDsDict[K] = ltIDs
    return ltIDs


def calc_drho_dcumprod1mrho(cumprod1mrho, rho, K):
  ''' Calculate partial derivative of cumprod1mrho w.r.t. rho

      Returns
      ---------
      RMat : 2D array, size K x K
  '''
  RMat = np.tile(-1*cumprod1mrho, (K,1))
  RMat /= (1-rho)[:,np.newaxis]
  RMat[_get_lowTriIDs(K)] = 0
  return RMat

