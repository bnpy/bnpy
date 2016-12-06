'''
Unit-tests for ParamBag
'''

from bnpy.suffstats.ParamBag import ParamBag
import numpy as np
import unittest


class TestParamBag(unittest.TestCase):

    def shortDescription(self):
        return None

    def test_setAllFieldsToZero_K1_D1(self, K=1, D=1):
        A = ParamBag(K=K, D=D)
        s = 123
        N = np.ones(K)
        x = np.ones((K, D))
        xxT = np.ones((K, D, D))
        W = np.ones((K, K))
        A.setField('s', s)
        A.setField('N', N, dims='K')
        A.setField('x', x, dims=('K', 'D'))
        A.setField('xxT', xxT, dims=('K', 'D', 'D'))
        A.setField('W', W, dims=('K', 'K'))
        A.setAllFieldsToZero()
        assert np.allclose(A.s, 0.0)
        assert np.allclose(A.N, np.zeros(K))
        assert np.allclose(A.x, np.zeros(K))
        assert np.allclose(A.xxT, np.zeros(K))
        assert np.allclose(A.xxT, np.zeros((K, K)))

    # insertEmptyComps
    def test_insertEmptyComps_K1_D1(self, K=1, D=1):
        A = ParamBag(K=K, D=D)
        s = 123
        N = np.zeros(K)
        x = np.zeros((K, D))
        xxT = np.zeros((K, D, D))
        W = np.zeros((K, K))
        A.setField('s', s)
        A.setField('N', N, dims='K')
        A.setField('x', x, dims=('K', 'D'))
        A.setField('xxT', xxT, dims=('K', 'D', 'D'))
        A.setField('W', W, dims=('K', 'K'))
        A.insertEmptyComps(2)
        assert np.allclose(A.s, 123)
        assert np.allclose(A.N, np.zeros(K + 2))
        assert np.allclose(A.x, np.zeros(K + 2))
        assert np.allclose(A.xxT, np.zeros(K + 2))
        assert np.allclose(A.W, np.zeros((K + 2, K + 2)))

    def test_insertEmptyComps_K1_D1(self, K=1, D=2):
        A = ParamBag(K=K, D=D)
        s = 123
        N = np.zeros(K)
        x = np.zeros((K, D))
        xxT = np.zeros((K, D, D))
        W = np.zeros((K, K))
        A.setField('s', s)
        A.setField('N', N, dims='K')
        A.setField('x', x, dims=('K', 'D'))
        A.setField('xxT', xxT, dims=('K', 'D', 'D'))
        A.setField('W', W, dims=('K', 'K'))
        A.insertEmptyComps(2)
        assert np.allclose(A.s, 123)
        assert np.allclose(A.N, np.zeros(K + 2))
        assert np.allclose(A.x, np.zeros((K + 2, D)))
        assert np.allclose(A.xxT, np.zeros((K + 2, D, D)))
        assert np.allclose(A.W, np.zeros((K + 2, K + 2)))

    def test_insertEmptyComps_K3_D1(self, K=3, D=1):
        A = ParamBag(K=K, D=D)
        s = 123
        N = np.zeros(K)
        x = np.zeros((K, D))
        xxT = np.zeros((K, D, D))
        W = np.zeros((K, K))
        A.setField('s', s)
        A.setField('N', N, dims='K')
        A.setField('x', x, dims=('K', 'D'))
        A.setField('xxT', xxT, dims=('K', 'D', 'D'))
        A.setField('W', W, dims=('K', 'K'))
        A.insertEmptyComps(2)
        assert np.allclose(A.s, 123)
        assert np.allclose(A.N, np.zeros(K + 2))
        assert np.allclose(A.x, np.zeros(K + 2))
        assert np.allclose(A.xxT, np.zeros(K + 2))
        assert np.allclose(A.W, np.zeros((K + 2, K + 2)))

    def test_insertEmptyComps_K3_D3(self, K=3, D=3):
        A = ParamBag(K=K, D=D)
        s = 123
        N = np.zeros(K)
        x = np.zeros((K, D))
        xxT = np.zeros((K, D, D))
        W = np.zeros((K, K))
        A.setField('s', s)
        A.setField('N', N, dims='K')
        A.setField('x', x, dims=('K', 'D'))
        A.setField('xxT', xxT, dims=('K', 'D', 'D'))
        A.setField('W', W, dims=('K', 'K'))
        A.insertEmptyComps(2)
        assert np.allclose(A.s, 123)
        assert np.allclose(A.N, np.zeros(K + 2))
        assert np.allclose(A.x, np.zeros((K + 2, D)))
        assert np.allclose(A.xxT, np.zeros((K + 2, D, D)))
        assert np.allclose(A.W, np.zeros((K + 2, K + 2)))

    # Verify insert
    def test_insertComps_K1_D1(self):
        A = ParamBag(K=1, D=1)
        s = 123.456
        A.setField('scalar', s, dims=None)
        A.setField('N', [1], dims='K')
        A.setField('x', [[1]], dims=('K', 'D'))
        A.setField('xxT', [[[1]]], dims=('K', 'D', 'D'))

        Abig = A.copy()
        Abig.insertComps(A)
        assert Abig.K == 2
        assert np.allclose(Abig.N, np.hstack([A.N, A.N]))
        # Verify that after inserting
        # scalar field is unchanged
        assert Abig.scalar == s

        Abig.insertComps(A)
        assert Abig.K == 3
        assert np.allclose(Abig.N, np.hstack([A.N, A.N, A.N]))
        assert Abig.scalar == s

        A.insertComps(Abig)
        assert A.K == 4
        assert A.scalar == s
        assert np.allclose(A.N, np.hstack([1, 1, 1, 1]))

    def test_insertComps_K1_D3(self, K=1, D=3):
        A = ParamBag(K=K, D=D)
        s = 123.456
        A.setField('scalar', s, dims=None)
        A.setField('N', [1.0], dims='K')
        A.setField('x', np.random.rand(K, D), dims=('K', 'D'))
        A.setField('xxT', np.random.rand(K, D, D), dims=('K', 'D', 'D'))

        Abig = A.copy()
        Abig.insertComps(A)

        assert Abig.K == 2
        assert np.allclose(Abig.N, np.hstack([A.N, A.N]))
        assert Abig.scalar == s
        assert Abig.xxT.shape == (2, 3, 3)
        assert np.allclose(Abig.xxT[0], A.xxT)
        assert np.allclose(Abig.xxT[1], A.xxT)

        Abig.insertComps(A)
        assert Abig.K == 3
        assert np.allclose(Abig.N, np.hstack([A.N, A.N, A.N]))
        assert Abig.scalar == s

        assert Abig.xxT.shape == (3, 3, 3)
        assert np.allclose(Abig.xxT[0], A.xxT)
        assert np.allclose(Abig.xxT[1], A.xxT)

        A.insertComps(Abig)
        assert A.K == 4
        assert A.scalar == s
        assert np.allclose(A.N, np.hstack([1, 1, 1, 1]))

    # Verify remove
    def test_removeComp_K1_D1(self):
        A = ParamBag(K=1, D=1)
        A.setField('N', [1], dims='K')
        A.setField('x', [[1]], dims=('K', 'D'))
        with self.assertRaises(ValueError):
            A.removeComp(0)

    def test_removeComp_K3_D1(self):
        A = ParamBag(K=3, D=1)
        A.setField('N', [1, 2, 3], dims='K')
        A.setField('x', [[4], [5], [6]], dims=('K', 'D'))
        A.setField('W', np.ones((3, 3)), dims=('K', 'K'))
        Aorig = A.copy()
        A.removeComp(1)
        assert Aorig.K == A.K + 1
        assert A.N[0] == Aorig.N[0]
        assert A.N[1] == Aorig.N[2]
        assert np.allclose(A.x, [[4], [6]])
        assert np.allclose(A.W, np.ones((2, 2)))

    def test_remove_K3_D2(self, K=3, D=2):
        A = ParamBag(K=K, D=D)
        s = 123
        N = np.random.rand(K)
        x = np.random.rand(K, D)
        xxT = np.random.randn(K, D, D)
        A.setField('s', s)
        A.setField('N', N, dims='K')
        A.setField('x', x, dims=('K', 'D'))
        A.setField('xxT', xxT, dims=('K', 'D', 'D'))
        Abig = A.copy()
        # First remove a few fields
        for k in range(K - 1):
            A.removeComp(0)
            assert A.K == K - k - 1
            assert A.s == s
            assert np.allclose(A.getComp(0).x, x[k + 1])
            assert np.allclose(A.getComp(0).xxT, xxT[k + 1])

    # Verify get
    def test_getComp_K1_D1(self):
        A = ParamBag(K=1, D=1)
        A.setField('scalar', 1, dims=None)
        A.setField('N', [1], dims='K')
        A.setField('x', [[1]], dims=('K', 'D'))
        c = A.getComp(0)
        assert c.K == 1
        assert c.N == A.N
        assert c.x == A.x
        assert id(c.scalar) != id(A.scalar)
        assert id(c.N) != id(A.N)
        assert id(c.x) != id(A.x)

    def test_getComp_K3_D1(self):
        A = ParamBag(K=3, D=1)
        A.setField('N', [1, 2, 3], dims='K')
        A.setField('x', [[4], [5], [6]], dims=('K', 'D'))
        c = A.getComp(0)
        assert c.K == 1
        assert c.N == A.N[0]
        assert c.x == A.x[0]
        assert id(c.N) != id(A.N)
        assert id(c.x) != id(A.x)

    # Verify add/subtract
    def test_add_K1_D1(self):
        A = ParamBag(K=1, D=1)
        B = ParamBag(K=1, D=1)
        C = A + B
        assert C.K == A.K and C.D == A.D
        A.setField('N', [1], dims='K')
        B.setField('N', [10], dims='K')
        C = A + B
        assert C.N[0] == 11.0

    def test_add_K3_D2(self, K=3, D=2):
        A = ParamBag(K=K, D=D)
        A.setField('xxT', np.random.randn(K, D, D), dims=('K', 'D', 'D'))
        B = ParamBag(K=K, D=D)
        B.setField('xxT', np.random.randn(K, D, D), dims=('K', 'D', 'D'))
        C = A + B
        assert np.allclose(C.xxT, A.xxT + B.xxT)

    def test_sub_K3_D2(self, K=3, D=2):
        A = ParamBag(K=K, D=D)
        A.setField('xxT', np.random.randn(K, D, D), dims=('K', 'D', 'D'))
        B = ParamBag(K=K, D=D)
        B.setField('xxT', np.random.randn(K, D, D), dims=('K', 'D', 'D'))
        C = A - B
        assert np.allclose(C.xxT, A.xxT - B.xxT)

    def test_iadd_K3_D2(self, K=3, D=2):
        A = ParamBag(K=K, D=D)
        A.setField('xxT', np.random.randn(K, D, D), dims=('K', 'D', 'D'))
        A.setField('x', np.random.randn(K, D), dims=('K', 'D'))
        B = ParamBag(K=K, D=D)
        B.setField('x', np.random.randn(K, D), dims=('K', 'D'))
        B.setField('xxT', np.random.randn(K, D, D), dims=('K', 'D', 'D'))
        origID = hex(id(A))
        A += B
        newID = hex(id(A))
        assert origID == newID
        A = A + B
        newnewID = hex(id(A))
        assert newnewID != origID

    def test_isub_K3_D2(self, K=3, D=2):
        A = ParamBag(K=K, D=D)
        A.setField('xxT', np.random.randn(K, D, D), dims=('K', 'D', 'D'))
        A.setField('x', np.random.randn(K, D), dims=('K', 'D'))
        B = ParamBag(K=K, D=D)
        B.setField('x', np.random.randn(K, D), dims=('K', 'D'))
        B.setField('xxT', np.random.randn(K, D, D), dims=('K', 'D', 'D'))
        origID = hex(id(A))
        A -= B
        newID = hex(id(A))
        assert origID == newID
        A = A - B
        newnewID = hex(id(A))
        assert newnewID != origID

    # Dim 0 parsing
    def test_parseArr_dim0_passes(self):
        PB1 = ParamBag(K=1, D=1)
        x = PB1.parseArr(1.23, dims=None)
        assert x.ndim == 0 and x.size == 1
        x = PB1.parseArr([1.23], dims=('K'))
        assert x.ndim == 1 and x.size == 1

        PB2 = ParamBag(K=2, D=1)
        x = PB2.parseArr(1.23, dims=None)
        assert x.ndim == 0 and x.size == 1

        PB5 = ParamBag(K=5, D=40)
        x = PB5.parseArr(1.23, dims=None)
        assert x.ndim == 0 and x.size == 1

    def test_parseArr_dim0_fails(self):
        ''' Verify fails for 0-dim input when K > 1
        '''
        PB2 = ParamBag(K=2, D=1)
        with self.assertRaises(ValueError):
            x = PB2.parseArr(1.23, dims=('K'))
        with self.assertRaises(ValueError):
            x = PB2.parseArr(1.23, dims='K')

    # Dim 1 parsing
    def test_parseArr_dim1_passes(self):
        # K = 1, D = 1
        PB1 = ParamBag(K=1, D=1)
        x = PB1.parseArr([1.23], dims='K')
        assert x.ndim == 1 and x.size == 1
        x = PB1.parseArr([[1.23]], dims=('K', 'D'))
        assert x.ndim == 2 and x.size == 1

        # K = *, D = 1
        PB2 = ParamBag(K=2, D=1)
        x = PB2.parseArr([1., 2.], dims='K')
        assert x.ndim == 1 and x.size == 2
        x = PB2.parseArr([[1.], [2.]], dims=('K', 'D'))
        assert x.ndim == 2 and x.size == 2

        # K = 1, D = *
        PB3 = ParamBag(K=1, D=3)
        x = PB3.parseArr([[1., 2., 3.]], dims=('K', 'D'))
        assert x.ndim == 2 and x.size == 3

        # K = *, D = *
        PB2 = ParamBag(K=4, D=1)
        x = PB2.parseArr([[1.], [2.], [3.], [4.]], dims=('K', 'D'))
        assert x.ndim == 2 and x.size == 4
        N = PB2.parseArr([1., 2., 3., 4.], dims='K')
        assert N.ndim == 1 and N.size == 4

    def test_parseArr_dim1_fails(self):
        PB1 = ParamBag(K=1, D=1)
        with self.assertRaises(ValueError):
            x = PB1.parseArr([1.23], dims=('K', 'D'))

        PB2 = ParamBag(K=2, D=1)
        with self.assertRaises(ValueError):
            x = PB2.parseArr([1.23], dims=('K'))
        with self.assertRaises(ValueError):
            x = PB2.parseArr([1.23], dims=('K', 'D'))

        PB3 = ParamBag(K=1, D=3)
        with self.assertRaises(ValueError):
            x = PB3.parseArr([1., 2.], dims=('K', 'D'))

        PB3 = ParamBag(K=2, D=3)
        with self.assertRaises(ValueError):
            x = PB3.parseArr([1., 2., 3., 4., 5., 6.], dims=('K', 'D'))

    # Dim 2 parsing
    def test_parseArr_dim2_passes(self):
        PB2 = ParamBag(K=2, D=2)
        x = PB2.parseArr(np.eye(2), dims=('K', 'D'))
        assert x.ndim == 2 and x.size == 4

        PB31 = ParamBag(K=3, D=1)
        x = PB31.parseArr([[10], [11], [12]], dims=('K', 'D'))
        assert x.ndim == 2 and x.size == 3

    def test_parseArr_dim2_fails(self):
        PB2 = ParamBag(K=2, D=2)
        with self.assertRaises(ValueError):
            x = PB2.parseArr([[1., 2]], dims=('K'))

        with self.assertRaises(ValueError):
            x = PB2.parseArr([[1., 2]], dims=('K', 'D'))

        with self.assertRaises(ValueError):
            x = PB2.parseArr(np.eye(3), dims=('K', 'D'))

        PB1 = ParamBag(K=1, D=2)
        with self.assertRaises(ValueError):
            # should be 1x2x2, not 2x2
            x = PB1.parseArr(np.eye(2), dims=('K', 'D', 'D'))

    # Dim 3 parsing

    def test_parseArr_dim3_passes(self):
        K = 2
        D = 2
        PB = ParamBag(K=K, D=D)
        x = PB.parseArr(np.random.randn(K, D, D), dims=('K', 'D', 'D'))
        assert x.ndim == 3 and x.size == K * D * D

        K = 1
        D = 2
        PB = ParamBag(K=K, D=D)
        x = PB.parseArr(np.random.rand(K, D, D), dims=('K', 'D', 'D'))
        assert x.ndim == 3 and x.size == K * D * D

        K = 3
        D = 1
        PB = ParamBag(K=K, D=D)
        x = PB.parseArr(np.random.rand(K, D, D), dims=('K', 'D', 'D'))
        assert x.ndim == 3 and x.size == K * D * D

    def test_parseArr_dim3_fails(self):
        PB = ParamBag(K=2, D=2)
        with self.assertRaises(ValueError):
            x = PB.parseArr([[[1., 2]]], dims=('K'))

        with self.assertRaises(ValueError):
            x = PB.parseArr([[[1., 2]]], dims=('K', 'D'))

        with self.assertRaises(ValueError):
            x = PB.parseArr(np.random.randn(3, 3, 3), dims=('K', 'D'))

        with self.assertRaises(ValueError):
            x = PB.parseArr(np.random.randn(3, 3, 3), dims=('K', 'D', 'D'))
