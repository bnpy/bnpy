

import numpy as np
import sys
import os
import errno
from numpy.random import RandomState
import random_projection as rp
import gram_schmidt_stable as gs 

def findAnchors(Q, K, params, candidates):

    # row normalize Q
    row_sums = Q.sum(axis=1)
    for i in range(len(Q[:, 0])):
        Q[i, :] = Q[i, :]/float(row_sums[i] + 1e-100)    

    # Reduced dimension random projection method for recovering anchor words
    if params.lowerDim is None  or params.lowerDim >= Q.shape[1]:
      Q_red = Q.copy()
    else:
      # Random number generator for generating dimension reduction
      prng_W = RandomState(params.seed)
      Q_red = rp.Random_Projection(Q.T, params.lowerDim, prng_W)
      Q_red = Q_red.T
    (anchors, anchor_indices) = gs.Projection_Find(Q_red, K, candidates)

    # restore the original Q
    for i in range(len(Q[:, 0])):
        Q[i, :] = Q[i, :]*float(row_sums[i])

    return anchor_indices


