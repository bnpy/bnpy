'''
AnimalAttrib.py

------------------------------------------------------------------
Animals with Attributes Dataset, v1.0, May 13th 2009
------------------------------------------------------------------

Animal/attribute matrix for 50 animal categories and 85 attributes.
Animals and attributes are in the same order as in the text files
'classes.txt' and 'predictes.txt'.

The numeric data was originally collected by Osherson et al. [1],
and extended by Kemp et al. [2]. Augmenting image data was collected
by Lampert et al. [3] (http://attributes.kyb.tuebingen.mpg.de).

Missing values in the numeric table are marked by -1. The binary
matrix was created by thresholding the continuous table at the
overall mean.

[1] D. N. Osherson, J. Stern, O. Wilkie, M. Stob, and E. E.
    Smith. Default probability. Cognitive Science, 15(2), 1991.

[2] C. Kemp, J. B. Tenenbaum, T. L. Griffiths, T. Yamada, and
    N. Ueda. Learning systems of concepts with an infinite rela-
    tional model. In AAAI, 2006.

[3] C. H. Lampert, H. Nickisch, and S. Harmeling. Learning To
    Detect Unseen Object Classes by Between-Class Attribute
    Transfer. In CVPR, 2009
'''

import numpy as np
import scipy.linalg
import os

from bnpy.data import XData

# Auto-determine the dataset directory based on where this script is located
datasetdir = os.path.sep.join(
    os.path.abspath(__file__).split(
        os.path.sep)[
            :-
            1])
if not os.path.isdir(datasetdir):
    raise ValueError('CANNOT FIND DATASET DIRECTORY:\n' + datasetdir)

filepath = os.path.join(datasetdir, 'rawData',
                        'AnimalAttrib', 'predicate-matrix-binary.txt')
if not os.path.isfile(filepath):
    raise ValueError('CANNOT FIND DATASET FILE:\n' + filepath)


def get_short_name():
    ''' Return short string used in filepaths to store solutions
    '''
    return 'AnimalAttrib'


def get_data_info():
    return 'Animals Attribute Dataset. 50 animals and 85 binary features.'


def get_data(**kwargs):
    '''
      Args
      -------
      filepath

      Returns
      -------
        Data : bnpy XData object, with nObsTotal observations
    '''
    X = np.loadtxt(filepath, dtype=np.float64)
    Data = XData(X=X)
    Data.name = get_short_name()
    Data.summary = get_data_info()
    return Data


if __name__ == '__main__':
    Data = get_data()
    from matplotlib import pylab
    imshowArgs = dict(
        interpolation='nearest',
        cmap='bone',
        vmin=0.0,
        vmax=1.0)
    pylab.imshow(Data.X, **imshowArgs)
    pylab.show()
