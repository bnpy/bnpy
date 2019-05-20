import numpy as np
import scipy.stats
import bnpy

if __name__ == '__main__':
    K = 3
    D = 2
    T = 5
    L = K

    mu_KD = np.zeros((K,D))
    covar_KDD = np.zeros((K, D, D))
    for k in range(K):
        mu_KD[k,0] = k
        covar_KDD[k] = np.eye(D)

    trans_pi_KK = 2 * np.eye(K) + 1.0/K * np.ones((K, K))
    trans_pi_KK /= np.sum(trans_pi_KK, axis=1)[:,np.newaxis]
    start_pi_K = 1.0/K * np.ones(K)

    prng = np.random.RandomState(3)
    x_TD = prng.multivariate_normal(mu_KD[K-2], covar_KDD[K-2], size=T)

    lik_TK = np.zeros((T,K))
    for k in range(K):
        lik_TK[:,k] = scipy.stats.multivariate_normal.pdf(x_TD, mu_KD[k], covar_KDD[k])

    fmsg_TK, marglik_T, _ = bnpy.allocmodel.hmm.lib.LibFwdBwd.FwdAlg_cpp(start_pi_K, trans_pi_KK, lik_TK)
    sp_fmsg_TL, sp_marglik_T, sp_col_TL = bnpy.allocmodel.hmm.lib.LibFwdBwd.FwdAlg_onepass_cpp(start_pi_K, trans_pi_KK, lik_TK, L)

    A = trans_pi_KK.T - np.eye(K)
    A[-1] = 1.0
    b = np.zeros(K)
    b[-1] = 1.0
    equilibrium_K = np.linalg.solve(A, b)
    sp_fmsg_TL_2, sp_marglik_T_2, sp_col_TL_2 = bnpy.allocmodel.hmm.lib.LibFwdBwd.FwdAlg_sparse_cpp(
                                                start_pi_K, trans_pi_KK, lik_TK, L, equilibrium_K)

    sp_fmsg_TK = np.zeros((T,K))
    sp_fmsg_TK_2 = np.zeros((T,K))
    for t in range(T):
        for m in range(L):
            sp_fmsg_TK[t, sp_col_TL[t,m]] = sp_fmsg_TL[t,m]
            sp_fmsg_TK_2[t, sp_col_TL_2[t,m]] = sp_fmsg_TL_2[t,m]
        print("DENSE  fmsg_TK[t=%d]: %s" % (t, fmsg_TK[t]))
        print("ONEPASS fmsg_TK[t=%d]: %s" % (t, sp_fmsg_TK[t]))
        print("O(L^2) fmsg_TK[t=%d]: %s" % (t, sp_fmsg_TK_2[t]))
        print


    print("DENSE marglik_T:")
    print(marglik_T)
    print("ONEPASS marglik_T:")
    print(sp_marglik_T)
    print("O(L^2) marglik_T:")
    print(sp_marglik_T_2)
    if L == K:
        assert np.allclose(sp_marglik_T, marglik_T)
        assert np.allclose(sp_fmsg_TK, fmsg_TK)
        assert np.allclose(sp_marglik_T_2, marglik_T)
        assert np.allclose(sp_fmsg_TK_2, fmsg_TK)
