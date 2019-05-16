import numpy as np
import scipy.stats
import bnpy

if __name__ == '__main__':
    K = 3
    D = 2
    T = 10
    L = 2

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

    fmsg_TK, marglik_T = bnpy.allocmodel.hmm.lib.LibFwdBwd.FwdAlg_cpp(start_pi_K, trans_pi_KK, lik_TK)
    sp_fmsg_TL, sp_col_TL, sp_marglik_T = bnpy.allocmodel.hmm.lib.LibFwdBwd.FwdAlg_onepass_cpp(start_pi_K, trans_pi_KK, lik_TK, L)

    sp_fmsg_TK = np.zeros((T,K))
    for t in range(T):
        for m in range(L):
            sp_fmsg_TK[t, sp_col_TL[t,m]] = sp_fmsg_TL[t,m]
        print("DENSE  fmsg_TK[t=%d]: %s" % (t, fmsg_TK[t]))
        print("SPARSE fmsg_TK[t=%d]: %s" % (t, sp_fmsg_TK[t]))


    print("DENSE marglik_T:")
    print(marglik_T)
    print("SPARSE marglik_T:")
    print(sp_marglik_T)
    if L == K:
        assert np.allclose(sp_marglik_T, marglik_T)
        assert np.allclose(sp_fmsg_TK, fmsg_TK)