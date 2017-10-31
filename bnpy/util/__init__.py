"""
The :mod:`util` module gathers utility functions
"""
from builtins import *
from . import RandUtil
from . import OptimizerForPi

from .PrettyPrintUtil import np2flatstr, flatstr2np
from .PrettyPrintUtil import split_str_into_fixed_width_lines
from .MatMultUtil import dotATA, dotATB, dotABT
from .MemoryUtil import getMemUsageOfCurProcess_MiB, calcObjSize_MiB
from .RandUtil import choice, multinomial
from .SpecialFuncUtil import MVgammaln, MVdigamma, digamma, gammaln
from .SpecialFuncUtil import LOGTWO, LOGPI, LOGTWOPI, EPS
from .SpecialFuncUtil import logsumexp
from .VerificationUtil import isEvenlyDivisibleFloat, assert_allclose
from .ShapeUtil import as1D, as2D, as3D, toCArray
from .ShapeUtil import argsort_bigtosmall_stable, is_sorted_bigtosmall
from .ShapeUtil import argsortBigToSmallByTiers
from .ParallelUtil import numpyToSharedMemArray, sharedMemToNumpyArray
from .ParallelUtil import sharedMemDictToNumpy, fillSharedMemArray

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
