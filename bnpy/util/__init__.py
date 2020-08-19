"""
The :mod:`util` module gathers utility functions
"""

from bnpy.util import RandUtil
from bnpy.util import OptimizerForPi

from bnpy.util.PrettyPrintUtil import np2flatstr, flatstr2np
from bnpy.util.PrettyPrintUtil import split_str_into_fixed_width_lines
from bnpy.util.MatMultUtil import dotATA, dotATB, dotABT
from bnpy.util.MemoryUtil import getMemUsageOfCurProcess_MiB, calcObjSize_MiB
from bnpy.util.RandUtil import choice, multinomial
from bnpy.util.SpecialFuncUtil import MVgammaln, MVdigamma, digamma, gammaln
from bnpy.util.SpecialFuncUtil import LOGTWO, LOGPI, LOGTWOPI, EPS
from bnpy.util.SpecialFuncUtil import logsumexp
from bnpy.util.VerificationUtil import isEvenlyDivisibleFloat, assert_allclose
from bnpy.util.ShapeUtil import as1D, as2D, as3D, toCArray
from bnpy.util.ShapeUtil import argsort_bigtosmall_stable, is_sorted_bigtosmall
from bnpy.util.ShapeUtil import argsortBigToSmallByTiers
from bnpy.util.ParallelUtil import numpyToSharedMemArray, sharedMemToNumpyArray
from bnpy.util.ParallelUtil import sharedMemDictToNumpy, fillSharedMemArray


__all__ = ['RandUtil', 'OptimizerForPi',
           'split_str_into_fixed_width_lines',
           'np2flatstr', 'flatstr2np',
           'dotATA', 'dotATB', 'dotABT',
           'choice', 'multinomial',
           'MVgammaln', 'MVdigamma', 'logsumexp', 'digamma', 'gammaln',
           'isEvenlyDivisibleFloat', 'assert_allclose',
           'LOGTWO', 'LOGTWOPI', 'LOGPI', 'EPS',
           'as1D', 'as2D', 'as3D', 'toCArray',
           'argsort_bigtosmall_stable', 'is_sorted_bigtosmall',
           'argsortBigToSmallByTiers',
           'numpyToSharedMemArray', 'sharedMemToNumpyArray',
           'sharedMemDictToNumpy', 'fillSharedMemArray',
           'getMemUsageOfCurProcess', 'calcObjSize_MiB',
           ]
