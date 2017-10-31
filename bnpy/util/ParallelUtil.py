from builtins import *
import numpy as np
import multiprocessing.sharedctypes
import warnings


def sharedMemDictToNumpy(ShMem):
    """ Get views (not copies) of all shared-mem arrays in dict.

    Returns
    -------
    d : dict
    """
    ArrDict = dict()
    if ShMem is None:
        return ArrDict

    for key, ShArr in list(ShMem.items()):
        ArrDict[key] = sharedMemToNumpyArray(ShArr)
    return ArrDict


def numpyToSharedMemArray(X):
    """ Get copy of X accessible as shared memory

    Returns
    --------
    Xsh : RawArray (same size as X)
        Uses separate storage than original array X.
    """
    Xtmp = np.ctypeslib.as_ctypes(X)
    Xsh = multiprocessing.sharedctypes.RawArray(Xtmp._type_, Xtmp)
    return Xsh


def sharedMemToNumpyArray(Xsh):
    """ Get view (not copy) of shared memory as numpy array.

    Returns
    -------
    X : ND numpy array (same size as X)
        Any changes to X will also influence data stored in Xsh.
    """
    if isinstance(Xsh, int):
        return Xsh
    elif isinstance(Xsh, np.ndarray):
        return Xsh
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.ctypeslib.as_array(Xsh)


def fillSharedMemArray(Xsh, Xarr):
    """ Copy all data from a numpy array into provided shared memory

    Post Condition
    --------------
    Xsh updated in place.
    """
    Xsh_arr = sharedMemToNumpyArray(Xsh)
    K = Xarr.shape[0]
    assert Xsh_arr.shape[0] >= K
    Xsh_arr[:K] = Xarr
