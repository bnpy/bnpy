import numpy as np
import copy


class ParamBag(object):

    ''' Container object for groups of related parameters.

    For example, can keep mean/variance params for all components
    of a Gaussian mixture model (GMM).

    Key functionality
    * run-time dimensionality verification, ensure matrices have correct size
    * easy access to the parameters for one component
    * remove/delete a particular component
    * insert new components

    Usage
    --------
    Create a new ParamBag
    >>> D = 3
    >>> PB = ParamBag(K=1, D=D)

    Add K x D field for mean parameters
    >>> PB.setField('Mu', np.ones((1,D)), dims=('K','D'))

    Add K x D x D field for all covar matrices
    >>> PB.setField('Sigma', np.eye(D)[np.newaxis,:], dims=('K','D','D'))

    >>> PB.Sigma
    array([[[ 1.,  0.,  0.],
            [ 0.,  1.,  0.],
            [ 0.,  0.,  1.]]])

    Insert an empty component
    >>> PB.insertEmptyComps(1)

    >>> PB.K
    2
    >>> PB.Mu
    array([[ 1.,  1.,  1.],
           [ 0.,  0.,  0.]])
    '''

    def __init__(self, K=0, doCollapseK1=False, **kwargs):
        ''' Create a ParamBag object with specified number of components.

        Args
        --------
        K : integer number of components this bag will contain
        D : integer dimension of parameters this bag will contain
        '''
        self.K = K
        self.D = 0
        for key, val in kwargs.iteritems():
            setattr(self, key, val)
        self._FieldDims = dict()
        self.doCollapseK1 = doCollapseK1

    def copy(self):
        ''' Returns deep copy of this object with separate memory.
        '''
        return copy.deepcopy(self)

    def setField(self, key, rawArray, dims=None):
        ''' Set a named field to particular array value.

        Raises
        ------
        ValueError
            if provided rawArray cannot be parsed into
            shape expected by the provided dimensions tuple
        '''
        # Parse dims tuple
        if dims is None and key in self._FieldDims:
            dims = self._FieldDims[key]
        else:
            self._FieldDims[key] = dims
        # Parse value as numpy array
        setattr(self, key, self.parseArr(rawArray, dims=dims, key=key))

    def setAllFieldsToZero(self):
        ''' Update every field to be an array of all zeros.
        '''
        for key, dims in self._FieldDims.items():
            curShape = getattr(self, key).shape
            self.setField(key, np.zeros(curShape), dims=dims)

    def reorderComps(self, sortIDs, fieldsToIgnore=[]):
        ''' Rearrange internal order of all fields along dimension 'K'
        '''
        for key in self._FieldDims:
            if key in fieldsToIgnore:
                continue
            arr = getattr(self, key)
            dims = self._FieldDims[key]
            if arr.ndim == 0:
                continue
            if dims[0] == 'K' and 'K' not in dims[1:]:
                arr = arr[sortIDs]
            elif dims[0] == 'K' and dims[1] == 'K' and 'K' not in dims[2:]:
                arr = arr[sortIDs, :][:, sortIDs]
            elif 'K' not in dims:
                continue
            elif dims[0] != 'K' and dims[1] == 'K':
                arr = arr[:, sortIDs]
            elif dims[0] != 'K' and dims[2] == 'K':
                arr = arr[:, :, sortIDs]
            else:
                raise NotImplementedError('TODO' + key + str(dims))
            self.setField(key, arr, dims=dims)

    def insertEmptyComps(self, Kextra):
        ''' Insert Kextra empty components to self in-place.
        '''
        origK = self.K
        self.K += Kextra
        for key in self._FieldDims:
            dims = self._FieldDims[key]
            if dims is None:
                continue
            if self.doCollapseK1:
                arr = self._getExpandedField(key, dims, K=origK)
            else:
                arr = getattr(self, key)
            for dimID, dimName in enumerate(dims):
                if dimName == 'K':
                    curShape = list(arr.shape)
                    curShape[dimID] = Kextra
                    zeroFill = np.zeros(curShape)
                    arr = np.append(arr, zeroFill, axis=dimID)
            self.setField(key, arr, dims=dims)

    def insertComps(self, PB):
        ''' Insert components from provided ParamBag to self in-place.
        '''
        assert PB.D == self.D
        origK = self.K
        self.K += PB.K
        for key in self._FieldDims:
            dims = self._FieldDims[key]
            if dims is None:
                pass
            elif dims[0] == 'K':
                if self.doCollapseK1:
                    arrA = self._getExpandedField(key, dims, K=origK)
                    arrB = PB._getExpandedField(key, dims)
                else:
                    arrA = getattr(self, key)
                    arrB = getattr(PB, key)
                arrC = np.append(arrA, arrB, axis=0)
                self.setField(key, arrC, dims=dims)
            elif dims[0] == 'M':
                self.setField(key, getattr(PB, key), dims=dims)
            else:
                raise NotImplementedError(
                    "Unknown insert request. key %s" % (key))

    def removeComp(self, k):
        ''' Updates self in-place to remove component "k"
        '''
        if k < 0 or k >= self.K:
            msg = 'Bad compID. Expected [0, %d], got %d' % (self.K - 1, k)
            raise IndexError(msg)
        if self.K <= 1:
            raise ValueError('Cannot remove final component.')
        self.K -= 1
        for key in self._FieldDims:
            arr = getattr(self, key)
            dims = self._FieldDims[key]
            if dims is not None:
                for dimID, name in enumerate(dims):
                    if name == 'K':
                        arr = np.delete(arr, k, axis=dimID)
                self.setField(key, arr, dims)

    def removeField(self, key):
        ''' Remove a field
        '''
        arr = getattr(self, key)
        dims = self._FieldDims[key]
        delattr(self, key)
        del self._FieldDims[key]
        return arr, dims

    def setComp(self, k, compPB):
        ''' Set (in-place) component k of self to provided compPB object.
        '''
        if k < 0 or k >= self.K:
            emsg = 'Bad compID. Expected [0, %d] but provided %d'
            emsg = emsg % (self.K - 1, k)
            raise IndexError(emsg)
        if compPB.K != 1:
            raise ValueError('Expected compPB to have K=1')
        for key, dims in self._FieldDims.items():
            if dims is None:
                self.setField(key, getattr(compPB, key), dims=None)
            elif self.K == 1:
                self.setField(key, getattr(compPB, key), dims=dims)
            else:
                bigArr = getattr(self, key)
                bigArr[k] = getattr(compPB, key)  # in-place

    def getComp(self, k, doCollapseK1=True):
        ''' Returns ParamBag object for component "k" of self.
        '''
        if k < 0 or k >= self.K:
            emsg = 'Bad compID. Expected [0, %d] but provided %d'
            emsg = emsg % (self.K - 1, k)
            raise IndexError(emsg)
        cPB = ParamBag(K=1, D=self.D, doCollapseK1=doCollapseK1)
        for key in self._FieldDims:
            arr = getattr(self, key)
            dims = self._FieldDims[key]
            if dims is not None:
                if self.K == 1:
                    cPB.setField(key, arr, dims=dims)
                else:
                    singleArr = arr[k]
                    if doCollapseK1:
                        cPB.setField(key, singleArr, dims=dims)
                    elif singleArr.ndim == 0:
                        cPB.setField(key, singleArr[np.newaxis], dims=dims)
                    else:
                        cPB.setField(key, singleArr[np.newaxis, :], dims=dims)
            else:
                cPB.setField(key, arr)
        return cPB

    def __add__(self, PB):
        ''' Add. Returns new ParamBag, with fields equal to self + PB
        '''
        # TODO: Decide on what happens if PB has more fields than self
        if self.K != PB.K or self.D != PB.D:
            raise ValueError('Dimension mismatch')
        PBsum = ParamBag(K=self.K, D=self.D, doCollapseK1=self.doCollapseK1)
        for key in self._FieldDims:
            arrA = getattr(self, key)
            arrB = getattr(PB, key)
            PBsum.setField(key, arrA + arrB, dims=self._FieldDims[key])
        return PBsum

    def __iadd__(self, PB):
        ''' In-place add. Updates self, with fields equal to self + PB.
        '''
        if self.K != PB.K or self.D != PB.D:
            raise ValueError('Dimension mismatch')
        if len(self._FieldDims.keys()) < len(PB._FieldDims.keys()):
            for key in PB._FieldDims:
                arrB = getattr(PB, key)
                try:
                    arrA = getattr(self, key)
                    self.setField(key, arrA + arrB)
                except AttributeError:
                    self.setField(key, arrB.copy(), dims=PB._FieldDims[key])
        else:
            for key in self._FieldDims:
                arrA = getattr(self, key)
                arrB = getattr(PB, key)
                self.setField(key, arrA + arrB)

        return self

    def subtractSpecificComps(self, PB, compIDs):
        ''' Subtract (in-place) from self the entire bag PB
                self.Fields[compIDs] -= PB
        '''
        assert len(compIDs) == PB.K
        for key in self._FieldDims:
            arr = getattr(self, key)
            if arr.ndim > 0:
                arr[compIDs] -= getattr(PB, key)
            else:
                self.setField(key, arr - getattr(PB, key), dims=None)

    def __sub__(self, PB):
        ''' Subtract.

        Returns new ParamBag object with fields equal to self - PB.
        '''
        if self.K != PB.K or self.D != PB.D:
            raise ValueError('Dimension mismatch')
        PBdiff = ParamBag(K=self.K, D=self.D, doCollapseK1=self.doCollapseK1)
        for key in self._FieldDims:
            arrA = getattr(self, key)
            arrB = getattr(PB, key)
            PBdiff.setField(key, arrA - arrB, dims=self._FieldDims[key])
        return PBdiff

    def __isub__(self, PB):
        ''' In-place subtract. Updates self, with fields equal to self - PB.
        '''
        if self.K != PB.K or self.D != PB.D:
            raise ValueError('Dimension mismatch')
        for key in self._FieldDims:
            arrA = getattr(self, key)
            arrB = getattr(PB, key)
            self.setField(key, arrA - arrB)
        return self

    def parseArr(self, arr, dims=None, key=None):
        ''' Parse provided array-like variable into a standard numpy array
            with provided dimensions "dims", as a tuple

            Returns
            --------
            numpy array with expected dimensions
        '''
        K = self.K
        D = self.D
        arr = np.asarray(arr, dtype=np.float64)
        # Verify shape is acceptable given expected dimensions
        if dims is not None and isinstance(dims, str):
            dims = (dims)  # force to tuple
        expectedShape = self._getExpectedShape(dims=dims)
        if self.doCollapseK1:
            if arr.shape not in self._getAllowedShapes(expectedShape):
                self._raiseDimError(dims, arr, key)
            # Squeeze into most economical shape possible
            #  e.g. (3,1) --> (3,),  (1,1) --> ()
            if K == 1 or D == 1:
                arr = np.squeeze(arr)
        else:
            if arr.shape != expectedShape:
                self._raiseDimError(dims, arr, key)
        if arr.ndim == 0:
            arr = np.float64(arr)
        return arr

    def _getExpectedShape(self, key=None, dims=None):
        ''' Returns tuple of expected shape, given named dimensions.

        Example
        -------
        >>> PB = ParamBag(K=3, D=2)
        >>> PB._getExpectedShape(dims=('K','K'))
        (3, 3)
        >>> PB._getExpectedShape(dims=('K','D','D'))
        (3, 2, 2)
        '''
        if key is not None:
            dims = self._FieldDims[key]
        if dims is None:
            expectShape = ()
        else:
            shapeList = list()
            for dim in dims:
                if isinstance(dim, int):
                    shapeList.append(dim)
                else:
                    shapeList.append(getattr(self, dim))
            expectShape = tuple(shapeList)
        return expectShape

    def _getAllowedShapes(self, shape):
        ''' Return set of allowed shapes that can be squeezed into given shape.

        Examples
        --------
        >>> PB = ParamBag() # fixing K,D doesn't matter
        >>> PB._getAllowedShapes(())
        set([()])
        >>> PB._getAllowedShapes((1,))
        set([(), (1,)])
        >>> aSet = PB._getAllowedShapes((23,))
        >>> sorted(aSet)
        [(23,)]
        >>> sorted(PB._getAllowedShapes((3,1)))
        [(3,), (3, 1)]
        >>> sorted(PB._getAllowedShapes((1,1)))
        [(), (1,), (1, 1)]
        '''
        assert isinstance(shape, tuple)
        allowedShapes = set()
        if len(shape) == 0:
            allowedShapes.add(tuple())
            return allowedShapes
        shapeVec = np.asarray(shape, dtype=np.int32)
        onesMask = shapeVec == 1
        keepMask = np.logical_not(onesMask)
        nOnes = sum(onesMask)
        for b in range(2**nOnes):
            bStr = np.binary_repr(b)
            bStr = '0' * (nOnes - len(bStr)) + bStr
            keepMask[onesMask] = np.asarray([int(x) > 0 for x in bStr])
            curShape = shapeVec[keepMask]
            allowedShapes.add(tuple(curShape))
        return allowedShapes

    def _raiseDimError(self, dims, badArr, key=None):
        ''' Raise ValueError when expected dimensions for array are not met.
        '''
        expectShape = self._getExpectedShape(dims=dims)
        if key is None:
            msg = 'Bad Dims. Expected %s, got %s' % (expectShape, badArr.shape)
        else:
            msg = 'Bad Dims for field %s. Expected %s, got %s' % (
                key, expectShape, badArr.shape)
        raise ValueError(msg)

    def _getExpandedField(self, key, dims, K=None, doExpandD=False):
        ''' Returns array expanded from squeezed form.

        Example
        --------
        Suppose dims=('K','D') and K=1, D=2
        Field stored as shape (2,) would be expanded to (1,2)

        Args
        --------
        key : name of field to expand
        dims : tuple of named dimensions, like ('K') or ('K','D')
        K : [optional] value for K to use, overrides self.K if provided
        '''
        if K is None:
            K = self.K
        arr = getattr(self, key)
        if arr.ndim < len(dims):
            for dimID, dimName in enumerate(dims):
                if dimName == 'K' and K == 1:
                    arr = np.expand_dims(arr, axis=dimID)
                elif getattr(self, dimName) == 1 and doExpandD:
                    arr = np.expand_dims(arr, axis=dimID)
        return arr
