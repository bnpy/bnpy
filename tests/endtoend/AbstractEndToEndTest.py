'''
Generic whole-system tests for end-to-end model training with bnpy.

Usage
------
>>> python runtests.py

'''
import sys
import numpy as np
import unittest
from nose.plugins.attrib import attr

import bnpy


def arg2name(aArg):
    if isinstance(aArg, dict):
        aName = aArg['name']
    elif isinstance(aArg, str):
        aName = aArg
    return aName


class AbstractEndToEndTest(unittest.TestCase):

    """ Defines test exercises for executing bnpy.run on provided dataset.

    Attributes
    ----
    Data : bnpy.data.DataObj
        dataset under testing
    possibleAllocModelNames : list
    possibleObsModelNames : list
    possibleInitNames : list
    """

    __test__ = False  # Do not execute this abstract module!

    def shortDescription(self):
        return None

    def makeAllocKwArgs(self, aName, algName):
        return dict()

    def makeObsKwArgs(self, oName, algName):
        return dict()

    def makeInitKwArgs(self, iName):
        kwargs = dict(initname=iName)
        return kwargs

    def nextAllocKwArgsForVB(self):
        for name in self.possibleAllocModelNames:
            yield dict(name=name)

    def nextObsKwArgsForVB(self, aName):
        for name in self.possibleObsModelNames:
            yield dict(name=name)

    def nextInitKwArgs(self):
        for name in self.possibleInitNames:
            yield dict(initname=name)

    def makeAllKwArgs(self, aName, obsArg, algName, iArg):
        kwargs = dict(
            doSaveToDisk=False,
            doWriteStdOut=False,
            saveEvery=-1,
            printEvery=-1,
            traceEvery=1,
            convergeThr=0.0001,
            nLap=300,
        )
        kwargs.update(self.makeAllocKwArgs(aName, algName))

        if isinstance(obsArg, str):
            kwargs.update(self.makeObsKwArgs(obsArg, algName))
        elif isinstance(obsArg, dict):
            kwargs.update(obsArg)

        if isinstance(iArg, str):
            kwargs.update(self.makeInitKwArgs(iArg))
        elif isinstance(iArg, dict):
            kwargs.update(iArg)
        return kwargs

    def single_run_repeatable_and_monotonic(self, aArg, oArg, algName, iArg):
        """ Test a single call to bnpy.run, verify repeatability and monotonic.
        """
        self.pprintSingleRun(aArg, oArg, algName, iArg)

        kwargs = self.makeAllKwArgs(aArg, oArg, algName, iArg)
        model1, Info1 = bnpy.run(self.Data, arg2name(aArg), arg2name(oArg),
                                 algName, **kwargs)
        self.pprintResult(model1, Info1)

        evTrace = Info1['evTrace']
        if algName.count('moVB'):
            evTrace = evTrace[Info1['lapTrace'] >= 1.0]
        isMonotonic = self.isMonotonic(evTrace)
        assert isMonotonic

        model2, Info2 = bnpy.run(self.Data, arg2name(aArg), arg2name(oArg),
                                 algName, **kwargs)
        self.pprintResult(model2, Info2)
        isRepeatable = np.allclose(Info1['evTrace'], Info2['evTrace'])
        assert isRepeatable

    def single_run_monotonic(self, aArg, oArg, algName, iArg):
        """ Test a single call to bnpy.run, verify monotonicity only.
        """
        self.pprintSingleRun(aArg, oArg, algName, iArg)

        kwargs = self.makeAllKwArgs(aArg, oArg, algName, iArg)
        model1, Info1 = bnpy.run(self.Data, arg2name(aArg), arg2name(oArg),
                                 algName, **kwargs)
        self.pprintResult(model1, Info1)
        isMonotonic = self.isMonotonic(Info1['evTrace'])
        assert isMonotonic

    @attr('slow')
    def test_VB_long__monotonic(self):
        print('')
        for aKwArgs in self.nextAllocKwArgsForVB():
            aName = arg2name(aKwArgs)
            for oKwArgs in self.nextObsKwArgsForVB(aName):
                for iKwArgs in self.nextInitKwArgs(aName, oKwArgs['name']):
                    self.single_run_monotonic(aKwArgs, oKwArgs,
                                              'VB', iKwArgs)

    @attr('fast')
    def test_VB__repeatable_and_monotonic(self):
        print('')
        for aName in self.possibleAllocModelNames:
            for oName in self.possibleObsModelNames:
                for iName in self.possibleInitNames:
                    self.single_run_repeatable_and_monotonic(aName, oName,
                                                             'VB', iName)

    @attr('fast')
    def test_EM__repeatable_and_monotonic(self):
        print('')
        for aName in self.possibleAllocModelNames:
            if 'EM' not in self.possibleLearnAlgsForAllocModel[aName]:
                continue
            for oName in self.possibleObsModelNames:
                for iName in self.possibleInitNames:
                    self.single_run_repeatable_and_monotonic(aName, oName,
                                                             'EM', iName)

    @attr('fast')
    def test_moVB__repeatable_and_monotonic(self):
        print('')
        for aName in self.possibleAllocModelNames:
            for oName in self.possibleObsModelNames:
                for iName in self.possibleInitNames:
                    self.single_run_repeatable_and_monotonic(aName, oName,
                                                             'moVB', iName)

    def isMonotonic(self, ELBOvec, atol=1e-6, verbose=True):
        ''' Returns True if monotonically increasing, False otherwise.

        Returns
        -------
        result : boolean (True or False)
        '''
        ELBOvec = np.asarray(ELBOvec, dtype=np.float64)
        assert ELBOvec.ndim == 1
        diff = ELBOvec[1:] - ELBOvec[:-1]
        maskIncrease = diff > 0
        maskWithinTol = np.abs(diff) < atol
        maskOK = np.logical_or(maskIncrease, maskWithinTol)
        isMonotonic = np.all(maskOK)
        if not isMonotonic and verbose:
            print("NOT MONOTONIC!")
            print('  %d violations found in vector of size %d.' % (
                np.sum(1 - maskOK), ELBOvec.size))
        return isMonotonic

    def test__isMonotonic(self):
        """ Verify that the isMonotonic boolean function has correct output.
        """
        assert self.isMonotonic([502.3, 503.1, 504.01, 504.00999999])
        assert not self.isMonotonic([502.3, 503.1, 504.01, 504.00989999],
                                    verbose=False)
        assert not self.isMonotonic([401.3, 400.99, 405.12],
                                    verbose=False)

    def pprintResult(self, model, Info):
        """ Pretty print the result of a learning algorithm.
        """
        print(" %25s after %4.1f sec  ELBO=% 7.3f  nLap=%5d  K=%d" % (
            Info['status'][:25],
            Info['elapsedTimeInSec'],
            Info['evBound'],
            Info['lapTrace'][-1],
            model.allocModel.K,
        ))

    def pprintSingleRun(self, aArg, oArg, algName, iArg):
        """ Pretty print information about current call to bnpy.run
        """
        print(">>> Run: %s" % (algName))
        self.pprint(aArg)
        self.pprint(oArg)
        self.pprint(iArg)

    def pprint(self, val):
        """ Pretty print the provided value.
        """
        if isinstance(val, str):
            print('  %s' % (val[:40]))
        elif hasattr(val, 'items'):
            firstMsg = ''
            msg = ''
            for (k, v) in list(val.items()):
                if k.count('name'):
                    firstMsg = str(v)
                else:
                    msg += " %s=%s" % (k, str(v))
            print('  ' + firstMsg + ' ' + msg)
