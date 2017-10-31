'''
OptimizerRhoOmegaBetter.py

Constrained Optimization Problem
--------------------------------
Variables:
Two K-length vectors
* rho = rho[0], rho[1], rho[2], ... rho[K-1]
* omega = omega[0], omega[1], ... omega[K-1]

Objective:
* argmax L(rho, omega)
or equivalently,
* argmin -1 * L(rho, omega)

Constraints:
* rho satisfies: 0 < rho[k] < 1
* omega satisfies: 0 < omega[k]
'''
from builtins import *
import warnings
import numpy as np
import scipy.optimize
import scipy.io
from scipy.special import gammaln, digamma, polygamma
import datetime
import logging

from bnpy.util.StickBreakUtil import rho2beta_active, beta2rho
from bnpy.util.StickBreakUtil import sigmoid, invsigmoid
from bnpy.util.StickBreakUtil import forceRhoInBounds, forceOmegaInBounds

Log = logging.getLogger('bnpy')

def negL_rho(rho=None, omega=None, initomega=None, **kwargs):
    if omega is None:
        omega = initomega
    kwargs['do_grad_rho'] = 1
    kwargs['do_grad_omega'] = 0
    return negL_rhoomega(rho=rho, omega=omega, **kwargs)

def negL_omega(omega=None, rho=None, initrho=None, **kwargs):
    if rho is None:
        rho = initrho
    kwargs['do_grad_rho'] = 0
    kwargs['do_grad_omega'] = 1
    return negL_rhoomega(rho=rho, omega=omega, **kwargs)

def negL_rhoomega(rhoomega=None, rho=None, omega=None,
            sumLogPiActiveVec=None,
            sumLogPiRemVec=None,
            sumLogPiRem=None,
            nDoc=0, gamma=1.0, alpha=1.0,
            approx_grad=False,
            do_grad_omega=1,
            do_grad_rho=1,
            **kwargs):
    ''' Returns negative ELBO objective function and its gradient.

    Args
    -------
    rhoomega := 1D array, size 2*K
        First K entries are vector rho
        Final K entries are vector omega

    Returns
    -------
    f := -1 * L(rho, omega), up to additive constant
         where L is ELBO objective function (log posterior prob)
    g := gradient of f
    '''
    if rhoomega is not None:
        assert not np.any(np.isnan(rhoomega))
        assert not np.any(np.isinf(rhoomega))
        rho, omega, K = _unpack(rhoomega)
    else:
        assert np.all(np.isfinite(rho))
        assert np.all(np.isfinite(omega))
        K = rho.size
        assert K == omega.size
    eta1 = rho * omega
    eta0 = (1 - rho) * omega
    digammaomega = digamma(omega)
    assert not np.any(np.isinf(digammaomega))
    Elogu = digamma(eta1) - digammaomega
    Elog1mu = digamma(eta0) - digammaomega
    if nDoc > 0:
        if sumLogPiRem is not None:
            sumLogPiRemVec = np.zeros(K)
            sumLogPiRemVec[-1] = sumLogPiRem

        ONcoef = nDoc + 1.0 - eta1
        OFFcoef = nDoc * kvec(K) + gamma - eta0

        Tvec = alpha * sumLogPiActiveVec
        Uvec = alpha * sumLogPiRemVec

        Ebeta_gtm1 = np.hstack([1.0, np.cumprod(1 - rho[:-1])])
        Ebeta = rho * Ebeta_gtm1
        assert Ebeta.size == Tvec.size
        Ebeta_gt = (1-rho) * Ebeta_gtm1
        L_local = np.inner(Ebeta, Tvec) + np.inner(Ebeta_gt, Uvec)
    else:
        # This is special case for unit tests that make sure the optimizer
        # finds the parameters that set q(u) equal to its prior when nDoc=0
        ONcoef = 1 - eta1
        OFFcoef = gamma - eta0
        L_local = 0
    # Compute total objective score L
    L = -1 * c_Beta(eta1, eta0) + \
        np.inner(ONcoef, Elogu) + \
        np.inner(OFFcoef, Elog1mu) + \
        L_local
    negL = -1.0 * L

    # When using approximate gradients, only the objective value is needed.
    if approx_grad:
        return negL

    # Gradient computation!
    trigamma_omega = polygamma(1, omega)
    trigamma_eta1 = polygamma(1, eta1)
    trigamma_eta0 = polygamma(1, eta0)
    assert np.all(np.isfinite(trigamma_omega))
    assert np.all(np.isfinite(trigamma_eta1))

    # First, compute omega gradients in closed form
    if do_grad_omega:
        gradomega = \
            ONcoef * (rho * trigamma_eta1 - trigamma_omega) + \
            OFFcoef * ((1 - rho) * trigamma_eta0 - trigamma_omega)

    if do_grad_rho:
        gradrho = omega * (
            ONcoef * trigamma_eta1 - OFFcoef * trigamma_eta0)
        if nDoc > 0:
            Psi = calc_Psi(Ebeta, rho, K)
            gradrho += np.dot(Psi, Uvec)

            Delta = calc_dEbeta_drho(Ebeta, rho, K)[:, :K]
            gradrho += np.dot(Delta, Tvec)

    # Return computed objective and (optionally) a gradient vector
    if do_grad_rho and do_grad_omega:
        grad = np.hstack([gradrho, gradomega])
        return negL, -1.0 * grad
    elif do_grad_rho:
        return negL, -1.0 * gradrho
    elif do_grad_omega:
        return negL, -1.0 * gradomega
    else:
        return negL


def find_optimum_multiple_tries(
        factrList=[1e4, 1e6, 1e8, 1e10, 1e12],
        **kwargs):
    ''' Robustly estimate optimal rho/omega via gradient descent on ELBO.

    Will gracefully using multiple restarts with progressively
    weaker tolerances until one succeeds.

    Args
    ----
    factrList : list of progressively weaker tolerances to try
        According to fmin_l_bfgs_b documentation:
            factr ~= 1e12 yields low accuracy,
            factr ~= 1e7 yields moderate accuracy
            factr ~= 1e2 yields extremely high accuracy

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
    rho_opt = None
    omega_opt = None
    Info = dict()
    errmsg = ''
    nOverflow = 0
    for trial, factr in enumerate(factrList):
        try:
            rho_opt, omega_opt, f_opt, Info = find_optimum(
                factr=factr,
                **kwargs)
            Info['nRestarts'] = trial
            Info['factr'] = factr
            Info['msg'] = Info['task']
            del Info['grad']
            del Info['task']
            break
        except ValueError as err:
            errmsg = str(err)
            Info['errmsg'] = errmsg
            if errmsg.count('overflow') > 0:
                # Eat any overflow problems.
                # Just discard this result and try again with diff factr val.
                nOverflow += 1
            elif errmsg.count('ABNORMAL_TERMINATION_IN_LNSRCH') > 0:
                # Eat any line search problems.
                # Just discard this result and try again with diff factr val.
                pass
            else:
                raise err

    if rho_opt is None:
        raise ValueError(errmsg)
    Info['nOverflow'] = nOverflow
    return rho_opt, omega_opt, f_opt, Info


def find_optimum(
        initrho=None, initomega=None,
        do_grad_rho=1, do_grad_omega=1, approx_grad=0,
        nDoc=None, sumLogPiActiveVec=None,
        sumLogPiRemVec=None, sumLogPiRem=None,
        alpha=1.0, gamma=1.0,
        factr=100.0,
        Log=None,
        **kwargs):
    ''' Estimate optimal rho and omega via gradient descent on ELBO objective.

    Returns
    --------
    rho : 1D array, length K
    omega : 1D array, length K
    f : scalar value of minimization objective
    Info : dict

    Raises
    --------
    ValueError on an overflow, any NaN, or failure to converge.

    Examples
    --------
    When no documents exist, we recover the prior parameters
    >>> r_opt, o_opt, f_opt, Info = find_optimum(
    ...     nDoc=0,
    ...     sumLogPiActiveVec=np.zeros(3),
    ...     sumLogPiRemVec=np.zeros(3),
    ...     alpha=0.5, gamma=1.0)
    >>> print r_opt
    [ 0.5  0.5  0.5]
    >>> print o_opt
    [ 2.  2.  2.]

    We can optimize for just rho by turning do_grad_omega off.
    This fixes omega at its initial value, but optimizes rho.
    >>> r_opt, o_opt, f_opt, Info = find_optimum(
    ...     do_grad_omega=0,
    ...     nDoc=10,
    ...     sumLogPiActiveVec=np.asarray([-2., -4., -6.]),
    ...     sumLogPiRemVec=np.asarray([0, 0, -20.]),
    ...     alpha=0.5,
    ...     gamma=5.0)
    >>> print o_opt
    [ 46.  36.  26.]
    >>> np.allclose(o_opt, Info['initomega'])
    True

    We can optimize for just omega by turning do_grad_rho off.
    This fixes rho at its initial value, but optimizes omega
    >>> r_opt2, o_opt2, f_opt2, Info = find_optimum(
    ...     do_grad_rho=0,
    ...     initrho=r_opt,
    ...     nDoc=10,
    ...     sumLogPiActiveVec=np.asarray([-2., -4., -6.]),
    ...     sumLogPiRemVec=np.asarray([0, 0, -20.]),
    ...     alpha=0.5,
    ...     gamma=5.0)
    >>> np.allclose(r_opt, r_opt2)
    True
    >>> np.allclose(o_opt2, o_opt, atol=10, rtol=0)
    True
    '''
    assert sumLogPiActiveVec.ndim == 1
    K = sumLogPiActiveVec.size
    if sumLogPiRem is not None:
        sumLogPiRemVec = np.zeros(K)
        sumLogPiRemVec[-1] = sumLogPiRem
    assert sumLogPiActiveVec.shape == sumLogPiRemVec.shape


    if nDoc > 0:
        maxOmegaVal = 1000.0 * (nDoc * (K+1) + gamma)
    else:
        maxOmegaVal = 1000.0 * (K + 1 + gamma)

    # Determine initial values for rho, omega
    if initrho is None:
        initrho = make_initrho(K, nDoc, gamma)
    initrho = forceRhoInBounds(initrho)
    if initomega is None:
        initomega = make_initomega(K, nDoc, gamma)
    initomega = forceOmegaInBounds(initomega, maxOmegaVal=0.5*maxOmegaVal)
    assert initrho.size == K
    assert initomega.size == K

    # Define keyword args for the objective function
    objFuncKwargs = dict(
        sumLogPiActiveVec=sumLogPiActiveVec,
        sumLogPiRemVec=sumLogPiRemVec,
        nDoc=nDoc,
        gamma=gamma,
        alpha=alpha,
        approx_grad=approx_grad,
        do_grad_rho=do_grad_rho,
        do_grad_omega=do_grad_omega,
        initrho=initrho,
        initomega=initomega)
    # Transform initial rho/omega into unconstrained vector c
    if do_grad_rho and do_grad_omega:
        rhoomega_init = np.hstack([initrho, initomega])
        c_init = rhoomega2c(rhoomega_init)
    elif do_grad_rho:
        c_init = rho2c(initrho)
        objFuncKwargs['omega'] = initomega
    else:
        c_init = omega2c(initomega)
        objFuncKwargs['rho'] = initrho
    # Define the objective function (in unconstrained space)
    def objFunc(c):
        return negL_c(c, **objFuncKwargs)

    # Define keyword args for the optimization package (fmin_l_bfgs_b)
    fminKwargs = dict(
        factr=factr,
        approx_grad=approx_grad,
        disp=None,
        )
    fminPossibleKwargs = set(scipy.optimize.fmin_l_bfgs_b.__code__.co_varnames)
    for key in kwargs:
        if key in fminPossibleKwargs:
            fminKwargs[key] = kwargs[key]
    # Run optimization, raising special error on any overflow or NaN issues
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            c_opt, f_opt, Info = scipy.optimize.fmin_l_bfgs_b(
                objFunc, c_init, **fminKwargs)
        except RuntimeWarning as e:
            # Any warnings are probably related to overflow.
            # Raise them as errors! We don't want a result with overflow.
            raise ValueError("FAILURE: " + str(e))
        except AssertionError as e:
            # Any assertions that failed mean that
            # rho/omega or some other derived quantity
            # reached a very bad place numerically. Raise an error!
            raise ValueError("FAILURE: NaN/Inf detected!")
    # Raise error on abnormal optimization warnings (like bad line search)
    if Info['warnflag'] > 1:
        raise ValueError("FAILURE: " + Info['task'])

    # Convert final answer back to rhoomega (safely)
    Info['initrho'] = initrho
    Info['initomega'] = initomega
    if do_grad_rho and do_grad_omega:
        rho_opt, omega_opt = c2rhoomega(c_opt)
    elif do_grad_rho:
        rho_opt = c2rho(c_opt)
        omega_opt = initomega
    else:
        omega_opt = c2omega(c_opt)
        rho_opt = initrho

    Info['estrho'] = rho_opt
    Info['estomega'] = omega_opt
    rho_safe = forceRhoInBounds(rho_opt)
    omega_safe = forceOmegaInBounds(
        omega_opt, maxOmegaVal=maxOmegaVal, Log=Log)
    objFuncKwargs['approx_grad'] = 1.0

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        objFuncKwargs['rho'] = initrho
        objFuncKwargs['omega'] = initomega
        f_init = negL_rhoomega(**objFuncKwargs)

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        objFuncKwargs['rho'] = rho_safe
        objFuncKwargs['omega'] = omega_safe
        f_safe = negL_rhoomega(**objFuncKwargs)
    if not np.allclose(rho_safe, rho_opt):
        if Log:
            Log.error('rho_opt_CHANGED_TO_LIE_IN_BOUNDS')
        Info['rho_opt_CHANGED_TO_LIE_IN_BOUNDS'] = 1
    if not np.allclose(omega_safe, omega_opt):
        if Log:
            Log.error('omega_opt_CHANGED_TO_LIE_IN_BOUNDS')
        Info['omega_opt_CHANGED_TO_LIE_IN_BOUNDS'] = 1
    if f_safe < f_init:
        return rho_safe, omega_safe, f_safe, Info
    else:
        return initrho, initomega, f_init, Info

def negL_c(c, do_grad_rho=1, do_grad_omega=1, approx_grad=0, **kwargs):
    if do_grad_rho and do_grad_omega:
        rhoomega = c2rhoomega(c, returnSingleVector=1)
        if approx_grad:
            f = negL_rhoomega(rhoomega, approx_grad=1, **kwargs)
            return f
        else:
            f, grad = negL_rhoomega(rhoomega, approx_grad=0, **kwargs)
            rho, omega, K = _unpack(rhoomega)
            drodc = np.hstack([rho * (1 - rho), omega])
            return f, grad * drodc
    elif do_grad_rho:
        rho = c2rho(c)
        if approx_grad:
            f = negL_rho(rho, approx_grad=1, **kwargs)
            return f
        else:
            f, grad = negL_rho(rho, approx_grad=0, **kwargs)
            drhodc = rho * (1 - rho)
            return f, grad * drhodc
    elif do_grad_omega:
        omega = c2omega(c)
        if approx_grad:
            f = negL_omega(omega, approx_grad=1, **kwargs)
            return f
        else:
            f, grad = negL_omega(omega, approx_grad=0, **kwargs)
            return f, grad * omega
    else:
        raise ValueError("Need to select at least one variable to infer.")

def c2rhoomega(c, returnSingleVector=False):
    ''' Transform unconstrained variable c into constrained rho, omega

    Returns
    --------
    rho : 1D array, size K, entries between [0, 1]
    omega : 1D array, size K, positive entries

    OPTIONAL: may return as one concatenated vector (length 2K)
    '''
    K = c.size / 2
    rho = sigmoid(c[:K])
    omega = np.exp(c[K:])
    if returnSingleVector:
        return np.hstack([rho, omega])
    return rho, omega

def c2rho(c):
    return sigmoid(c)

def c2omega(c):
    return np.exp(c)

def rhoomega2c(rhoomega):
    K = rhoomega.size / 2
    return np.hstack([invsigmoid(rhoomega[:K]), np.log(rhoomega[K:])])

def rho2c(rho):
    return invsigmoid(rho)

def omega2c(omega):
    return np.log(omega)

def _unpack(rhoomega):
    K = rhoomega.size / 2
    rho = rhoomega[:K]
    omega = rhoomega[-K:]
    return rho, omega, K


def make_initrho(K, nDoc, gamma):
    ''' Make vector rho that is good guess for provided problem specs.

    Uses known optimal value for related problem.

    Returns
    --------
    rho : 1D array, size K
        Each entry satisfies 0 <= rho[k] <= 1.0

    Example
    -------
    >>> rho = make_initrho(3, 0, 1.0)
    >>> print rho
    [ 0.5  0.5  0.5]
    '''
    eta1 = (nDoc + 1) * np.ones(K)
    eta0 = nDoc * kvec(K) + gamma
    rho = eta1 / (eta1 + eta0)
    return rho

def make_initomega(K, nDoc, gamma):
    ''' Make vector omega that is good guess for provided problem specs.

    Uses known optimal value for related problem.

    Returns
    --------
    omega : 1D array, size K
        Each entry omega[k] >= 0.
    '''
    eta1 = (nDoc + 1) * np.ones(K)
    eta0 = nDoc * kvec(K) + gamma
    omega = eta1 + eta0
    return omega



kvecCache = dict()
def kvec(K):
    ''' Obtain descending vector of [K, K-1, ... 1]

    Returns
    --------
    kvec : 1D array, size K
    '''
    try:
        return kvecCache[K]
    except KeyError as e:
        kvec = K + 1 - np.arange(1, K + 1)
        kvecCache[K] = kvec
        return kvec


def c_Beta(g1, g0):
    ''' Calculate cumulant function of the Beta distribution

    Input can be vectors, in which case we compute sum over
    several cumulant functions of the independent distributions:
    \prod_k Beta(g1[k], g0[k])

    Args
    ----
    g1 : 1D array, size K
        first parameter of a Beta distribution
    g0 : 1D array, size K
        second parameter of a Beta distribution

    Returns
    -------
    c : scalar sum of the cumulants defined by provided parameters
    '''
    return np.sum(gammaln(g1 + g0) - gammaln(g1) - gammaln(g0))


def calc_dEbeta_drho(Ebeta, rho, K):
    ''' Calculate partial derivative of Ebeta w.r.t. rho

    Returns
    ---------
    Delta : 2D array, size K x K
    '''
    Delta = np.tile(-1 * Ebeta, (K, 1))
    Delta /= (1 - rho)[:, np.newaxis]
    Delta[_get_diagIDs(K)] *= -1 * (1 - rho) / rho

    # Using flat indexing seems to be faster (about x2)
    Delta.ravel()[_get_flatLowTriIDs_KxK(K)] = 0
    return Delta


def calc_Psi(Ebeta, rho, K):
    ''' Calculate partial derivative of Ebeta_gt w.r.t. rho

    Returns
    ---------
    Psi : 2D array, size K x K
    '''
    Ebeta_gt = 1.0 - np.cumsum(Ebeta[:K])
    Psi = np.tile(-1 * Ebeta_gt, (K, 1))
    Psi /= (1 - rho)[:, np.newaxis]
    Psi.ravel()[_get_flatLowTriIDs_KxK(K)] = 0
    return Psi


flatlowTriIDsDict = dict()
flatlowTriIDsDict_KxK = dict()
diagIDsDict = dict()


def _get_diagIDs(K):
    if K in diagIDsDict:
        return diagIDsDict[K]
    else:
        diagIDs = np.diag_indices(K)
        diagIDsDict[K] = diagIDs
        return diagIDs


def _get_flatLowTriIDs_KxK(K):
    if K in flatlowTriIDsDict_KxK:
        return flatlowTriIDsDict_KxK[K]
    flatIDs = np.ravel_multi_index(np.tril_indices(K, -1), (K, K))
    flatlowTriIDsDict_KxK[K] = flatIDs
    return flatIDs


def calc_fgrid(o_grid=None, o_pos=None,
               r_grid=None, r_pos=None,
               omega=None, rho=None, **kwargs):
    ''' Evaluate the objective across range of values for one entry
    '''
    K = omega.size
    if o_grid is not None:
        assert o_pos >= 0 and o_pos < K
        f_grid = np.zeros_like(o_grid)
        omega_n = omega.copy()
        for n in range(o_grid.size):
            omega_n[o_pos] = o_grid[n]
            f_grid[n] = negL_omega(rho=rho, omega=omega_n,
                approx_grad=1, **kwargs)
    elif r_grid is not None:
        assert r_pos >= 0 and r_pos < K
        f_grid = np.zeros_like(r_grid)
        rho_n = rho.copy()
        for n in range(r_grid.size):
            rho_n[o_pos] = r_grid[n]
            f_grid[n] = negL_rho(rho=rho_n, omega=omega,
                approx_grad=1, **kwargs)
    else:
        raise ValueError("Must specify either o_grid or r_grid")

    return f_grid


def negL_rhoomega_viaHDPTopicUtil(
        rho=None, omega=None,
        nDoc=0,
        sumLogPiActiveVec=None,
        sumLogPiRemVec=None,
        alpha=0.5,
        gamma=1.0,
        **kwargs):
    ''' Compute minimization objective another way, using utility funcs.

    This allows verifying that our negL_rhoomega function is correct.

    Returns
    -------
    negL : -1 * L(rho, omega, ...)
        Should be the same value as negL_rhoomega.
    '''
    K = rho.size

    from .HDPTopicUtil import L_alloc
    Ldict = L_alloc(todict=1,
        rho=rho, omega=omega, nDoc=nDoc, alpha=alpha, gamma=gamma)

    from .HDPTopicUtil import calcELBO_NonlinearTerms
    Ldict2 = calcELBO_NonlinearTerms(todict=1,
        rho=rho,
        alpha=alpha,
        gamma=gamma,
        nDoc=nDoc,
        sumLogPi=sumLogPiActiveVec,
        sumLogPiRemVec=sumLogPiRemVec,
        gammalnTheta=np.zeros(K),
        gammalnSumTheta=0,
        gammalnThetaRem=0,
        slackTheta=np.zeros(K),
        slackThetaRem=0,
        Hresp=np.zeros(K),
        )
    Lrhoomega = Ldict['Lalloc_rhoomega'] + \
        Ldict2['Lslack_alphaEbeta']
    return -1 * Lrhoomega
