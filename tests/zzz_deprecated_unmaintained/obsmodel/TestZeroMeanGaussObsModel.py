import numpy as np
import unittest
import copy

import bnpy
from bnpy.obsmodel.ZeroMeanGaussObsModel import c_Func
from bnpy.util import as2D


class Test(unittest.TestCase):

    """ Verify integral for a single pair of nu/B values
    """

    def shortDescription(self):
        return None

    def __init__(self, testname,
                 B=10, nu=2, D=1,
                 **kwargs):
        super(type(self), self).__init__(testname)
        self.B = float(B)
        self.nu = float(nu)
        self.D = D

    def setUp(self):
        mean = self.nu / self.B
        var = self.nu / (0.5 * self.B * self.B)
        self.gridmaxval = mean + 8 * np.sqrt(var)
        self.gridsize = 8e6  # lots of grid points! need accurate integral.
        print('')
        print('B=%9.4f  nu=%9.4f | %9.4f' % (self.B, self.nu, self.gridmaxval))
        print('----------------- setup complete.')

    def test_integral_of_pdf_equals_one(self):
        """ Verify expected relation -c_F == log(integral(pdf))
        """
        grid = np.linspace(1e-10, self.gridmaxval, self.gridsize)
        grid = np.hstack([1e-13, 1e-12, 1e-11, grid])
        pdf = self.pdf(grid)

        negC_numeric = np.log(np.trapz(pdf, grid))
        negC_exact = -1 * c_Func(self.nu, as2D(self.B), 1)
        print('negC_numeric: % 9.7f' % (negC_numeric))
        print('negC_exact:   % 9.7f' % (negC_exact))
        assert np.allclose(negC_numeric, negC_exact, atol=0.001)

    def pdf(self, grid):
        return np.exp(0.5 * (self.nu - 2) * np.log(grid) - 0.5 * self.B * grid)


class TestRange(unittest.TestCase):

    """ Verify integral for a range of possible nu/B values
    """

    def runTest(self):
        suite = unittest.TestSuite()
        for B in np.linspace(1e-5, 10, 3):
            for nu in np.linspace(2, 10, 3):
                kwargs = dict(nu=nu, B=B)
                suite.addTest(
                    Test(
                        "test_integral_of_pdf_equals_one",
                        **kwargs))
        return unittest.TextTestRunner().run(suite)
