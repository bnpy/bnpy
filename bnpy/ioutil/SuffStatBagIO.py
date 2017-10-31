from builtins import *
import os
import numpy as np
try:
    from sklearn.externals import joblib
    hasJoblibModule = True
except ImportError:
    hasJoblibModule = False
    pass


def saveSuffStatBag(file_path='', SS=None, compress=1):
    ''' Write SuffStatBag object to disk in reproducible way.

    Args
    ----
    save_file_path : str, valid possible path on file system
        Indicates file where SS object will be written to disk
    SS : SuffStatBag or ParamBag object
        object wish to write to disk
    compress : int
        Larger values indicate more compression and longer IO times
        Possible values: 1, 2, ... 9
        Default is 1

    Post Condition
    --------------
    Binary file with all relevant info will be written at file_path

    Examples
    --------
    >>> import bnpy
    >>> SS = bnpy.suffstats.SuffStatBag(K=3, D=2)
    >>> SS.setField('attrVec', np.ones((3,2)), dims=('K', 'D'))
    >>> saveSuffStatBag('/tmp/SS.dump', SS)
    >>> SS2 = loadSuffStatBag('/tmp/SS.dump')
    >>> SS.K == SS2.K
    True
    >>> SS.D == SS2.D
    True
    >>> np.allclose(SS.attrVec, SS2.attrVec)
    True
    '''
    if compress < 1:
        raise ValueError("Value >= 1 required for compress argument")
    if not hasJoblibModule:
        raise ImportError("Cannot find required python module joblib")
    joblib.dump(SS, file_path, compress=compress)


def loadSuffStatBag(file_path=''):
    ''' Load SuffStatBag object from disk in reproducible way.

    Args
    ----
    file_path : str, valid file path on file system
        Indicates binary file where SS object has been written to disk

    Returns
    -------
    SS : SuffStatBag

    Examples
    --------
    >>> import bnpy
    >>> SS = bnpy.suffstats.SuffStatBag(K=2, D=4, E=3)
    >>> SS.setField('someVec', np.random.rand(2,3), dims=('K', 'E'))
    >>> SS.setELBOTerm('elboVec', np.arange(2), dims=('K'))
    >>> saveSuffStatBag('/tmp/SS.dump', SS)
    >>> SS2 = loadSuffStatBag('/tmp/SS.dump')
    >>> SS.K == SS2.K
    True
    >>> SS.D == SS2.D
    True
    >>> np.allclose(SS.getELBOTerm('elboVec'), SS2.getELBOTerm('elboVec'))
    True
    '''
    if not hasJoblibModule:
        raise ImportError("Cannot find required python module joblib")
    SS = joblib.load(file_path)
    return SS
