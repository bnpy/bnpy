'''
OptimizerHDPPE.py

CONSTRAINED Optimization Problem
----------
Variables:
* uhat : 1D array, size K

Objective:
* argmin -1 * E_{q(u|uhat)} [ log p( v_k | u, \alpha) 
                               + log Beta(u_k | 1, gamma)]

Constraints: 
* uhat satisfies: 0 < uhat[k] < 1
'''

import warnings
import numpy as np
import scipy.optimize
import scipy.io
from scipy.special import gammaln, digamma, polygamma
import datetime
import logging

from bnpy.util.StickBreakUtil import rho2beta_active, beta2rho, sigmoid, invsigmoid
from bnpy.util.StickBreakUtil import forceRhoInBounds
from bnpy.util.StickBreakUtil import create_initrho 
Log = logging.getLogger('bnpy')

def find_optimum_multiple_tries(sumLogVd=0, sumLog1mVd=0, nDoc=0, 
                                gamma=1.0, alpha=1.0,
                                inituhat=None,
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
  uhat = None
  Info = dict()
  msg = ''
  nOverflow = 0
  for trial, factr in enumerate(factrList):
    try:
      uhat, f, Info = find_optimum(sumLogVd, sumLog1mVd, nDoc,
                                       gamma=gamma, alpha=alpha,
                                       inituhat=inituhat,
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

  if uhat is None:
    raise ValueError(msg)
  Info['nOverflow'] = nOverflow
  return uhat, f, Info


def find_optimum(sumLogVd=0, sumLog1mVd=0, nDoc=0, gamma=1.0, alpha=1.0,
                 inituhat=None, 
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
  if inituhat is None:
    inituhat = create_inituhat(K)
  inituhat = forceRhoInBounds(inituhat)
  assert inituhat.size == K

  initc = uhat2c(inituhat)

  ## Define objective function (unconstrained!)
  objArgs = dict(sumLogVd=sumLogVd, sumLog1mVd=sumLog1mVd,
                  nDoc=nDoc, gamma=gamma, alpha=alpha,
                  approx_grad=approx_grad)

  c_objFunc = lambda c: objFunc_unconstrained(c, **objArgs)
  
  ## Run optimization and catch any overflow or NaN issues
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
      
  if Info['warnflag'] > 1:
    raise ValueError("FAILURE: " + Info['task'])

  Info['init'] = inituhat
  uhat = c2uhat(chat)
  uhat = forceRhoInBounds(uhat)
  return uhat, fhat, Info

def create_inituhat(K):
  ''' Make initial guess for uhat s.t. E[beta_k] \approx uniform (1/K)
      except that a small amount of remaining/leftover mass is reserved
  '''
  remMass = np.minimum(0.1, 1.0/(K*K))
  # delta = 0, -1 + r, -2 + 2r, ...
  delta = (-1 + remMass) * np.arange(0, K, 1, dtype=np.float)
  uhat = (1-remMass)/(K+delta)
  return uhat


########################################################### Objective
###########################################################  unconstrained
def objFunc_unconstrained(c, approx_grad=False, **kwargs):
  uhat = c2uhat(c)
  if approx_grad:
    f = objFunc_constrained(uhat, approx_grad=1, **kwargs)
    return f
  else:
    f, grad = objFunc_constrained(uhat, approx_grad=0, **kwargs)
    dudc = uhat * (1-uhat)
    return f, grad * dudc

def c2uhat(c, returnSingleVector=False):
  ''' Transform unconstrained variable c into constrained uhat

      Returns
      --------
      uhat : 1D array, size K, entries between [0, 1]
  '''
  uhat = sigmoid(c)
  return uhat

def uhat2c(uhat):
  return invsigmoid(uhat)

########################################################### Objective
###########################################################  constrained
def objFunc_constrained(uhat,
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
  rho = uhat
  assert np.all(np.isfinite(rho))
  K = rho.size

  if nDoc > 0:
    OFFcoef = (gamma-1)/nDoc
    P = sumLogVd/nDoc
    Q = sumLog1mVd/nDoc

    cumprod1mrho = np.ones(K)
    cumprod1mrho[1:] = np.cumprod( 1 - rho[:-1])

    rPand1mrQ = rho * P + (1-rho) * Q
    elbo_beta = alpha * np.inner(cumprod1mrho, rPand1mrQ) \
                + np.sum(c_Beta(alpha * cumprod1mrho * rho,
                                alpha * cumprod1mrho * (1-rho)))

  else:
    OFFcoef = gamma-1
    elbo_beta = 0

  Elog1mu = np.log(1-rho)
  elbo = np.sum(OFFcoef * Elog1mu) \
           + elbo_beta

  if approx_grad:
    return -1.0 * elbo

  ## Gradient computation!  
  gradrho = - OFFcoef / (1 - rho)
  if nDoc > 0:
    Delta = calc_drho_dcumprod1mrho(cumprod1mrho, rho, K)
    DeltaON = calc_drho_dON(Delta, cumprod1mrho, rho, K)
    DeltaOFF = calc_drho_dOFF(Delta, cumprod1mrho, rho, K)

    a = digamma( alpha * cumprod1mrho )
    b = digamma( alpha * cumprod1mrho * rho)
    c = digamma( alpha * cumprod1mrho * (1-rho) )
    gradrho += alpha * (np.dot(Delta, a) \
                        + np.dot(DeltaON, P - b) \
                        + np.dot(DeltaOFF, Q - c))
  return -1.0 * elbo, -1.0 * gradrho
  
########################################################### Util fcns
###########################################################

def c_Beta(g1, g0):
  ''' Calculate cumulant function of the Beta distribution

      Returns
      -------
      c : scalar sum of the cumulants defined by provided parameters
  '''
  return np.sum(gammaln(g1 + g0) - gammaln(g1) - gammaln(g0))


lowTriIDsDict = dict()
def _get_lowTriIDs(K):
  if K in lowTriIDsDict:
    return lowTriIDsDict[K]
  else:
    ltIDs = np.tril_indices(K)
    lowTriIDsDict[K] = ltIDs
    return ltIDs

diagIDsDict = dict()
def _get_diagIDs(K):
  if K in diagIDsDict:
    return diagIDsDict[K]
  else:
    diagIDs = np.diag_indices(K)
    diagIDsDict[K] = diagIDs
    return diagIDs

def calc_drho_dcumprod1mrho(cumprod1mrho, rho, K):
  ''' Calculate matrix of partial derivatives for expr \prod_{j < k} (1-\rho_j)

      Returns
      ---------
      RMat : 2D array, size K x K
             RMat[m, k] = d/d rho[m] [ expr[k] ]
  '''
  RMat = np.tile(-1*cumprod1mrho, (K,1))
  RMat /= (1-rho)[:,np.newaxis]
  RMat[_get_lowTriIDs(K)] = 0
  return RMat

def calc_drho_dON(RMat, cumprod1mrho, rho, K):
  DMat = RMat.copy()
  DMat *= rho[np.newaxis, :]
  DMat[_get_diagIDs(K)] = cumprod1mrho
  return DMat

def calc_drho_dOFF(RMat, cumprod1mrho, rho, K):
  DMat = RMat.copy()
  DMat *= (1-rho)[np.newaxis, :]
  DMat[_get_diagIDs(K)] = -1 * cumprod1mrho
  return DMat
