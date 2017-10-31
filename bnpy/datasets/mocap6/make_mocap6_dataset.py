from builtins import *
import numpy as np
import scipy.io
import bnpy

Q = scipy.io.loadmat('/Users/mhughes/git/mocap6dataset/mocap6.mat')
Q = Q['DataBySeq']

X_list = list()
X_prev_list = list()
T_list = list()
Z_list = list()
for doc in range(6):
    doc_X = np.asarray(Q[doc]['X'][0], dtype=np.float64).copy()
    doc_Xprev = np.asarray(Q[doc]['Xprev'][0], dtype=np.float64).copy()
    doc_Z = np.squeeze(
        np.asarray(Q[doc]['TrueZ'][0], dtype=np.float64).copy())
    X_list.append(doc_X)
    X_prev_list.append(doc_Xprev)
    T_list.append(doc_X.shape[0])
    Z_list.append(doc_Z.shape[0])

X = np.vstack(X_list)
Xprev = np.vstack(X_prev_list)
doc_range = np.hstack([0, np.cumsum(T_list)])
TrueZ = np.hstack(Z_list)

np.savez('dataset.npz',
    **dict(X=X, Xprev=Xprev, doc_range=doc_range, TrueZ=TrueZ))
