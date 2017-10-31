'''
VerificationUtil.py

Verification utilities, for checking whether numerical variables are "equal".
'''
from builtins import *
import numpy as np


def isEvenlyDivisibleFloat(a, b, margin=1e-6):
    ''' Get true/false if a is evenly divisible by b

    Uses (small) numerical tolerance to decide if divisible.

    Returns
    -------
    isDivisible : boolean

    Examples
    --------
    >>> isEvenlyDivisibleFloat( 1.5, 0.5)
    True
    >>> isEvenlyDivisibleFloat( 1.0, 1./3)
    True
    >>> isEvenlyDivisibleFloat( 1.0, 1./7)
    True
    >>> isEvenlyDivisibleFloat( 1.0, 1./17.0)
    True
    >>> isEvenlyDivisibleFloat( 5+2/17.0, 1./17.0)
    True
    >>> isEvenlyDivisibleFloat( 8.0/7, 1./17.0)
    False
    '''
    cexact = np.asarray(a) / float(b)
    cround = np.round(cexact)
    return abs(cexact - cround) < margin


def assert_allclose(a, b, atol=1e-8, rtol=0):
    """ Verify two arrays a,b are numerically indistinguishable.

    Returns
    -------
    isClose : boolean
    """
    isOK = np.allclose(a, b, atol=atol, rtol=rtol)
    if not OK:
        msg = np2flatstr(a)
        msg += "\n"
        msg += np2flatstr(b)
        print(msg)
    assert isOK
