import numpy as np

def checkWPost(w_m, w_var, K):
    w_m, w_var = np.asarray(w_m), np.asarray(w_var)

    w_m_t = np.zeros(K)
    w_m_t[:w_m.size] = w_m.flatten()[:K]
    w_m = w_m_t

    if len(w_var.shape) <= 1:
        w_var_t = np.ones(K)
        w_var_t[:w_var.size] = w_var.flatten()[:K]
        w_var = w_var_t
    else:
        w_var_t = np.eye(K)
        w_var_t[:w_var.shape[0], :w_var.shape[1]] = w_var[:K, :K]
        w_var = w_var_t

    return w_m, w_var

def lam(eta):
    return np.tanh(eta / 2.0) / (4.0 * eta)

def eta_update(m, S, X, XXT=None):
    if XXT is None:
        if m.size == S.size:
            eta2 = np.dot(X ** 2, S) + (np.dot(X, m) ** 2)
        else:
            eta2 = (X * np.dot(X.reshape((-1, m.size)), S)).sum(axis=1) + (np.dot(X, m) ** 2)
    else:
        if m.size == S.size:
            S = np.diag(S.flatten())
        eta2 = np.einsum('ijk,ijk->i', XXT.reshape((-1,) +  S.shape), S.reshape((1,) + S.shape)) + (np.dot(X, m) ** 2)
    
    eta2 = eta2.flatten()
    eta2 = eta2[0] if eta2.size == 1 else eta2
    return np.sqrt(eta2)

def calc_Zbar_ZZT_unnorm(resp, w_c):
    w_c = w_c.flatten()
    Zbar = np.dot(w_c, resp).flatten()
    ZZT = np.outer(Zbar, Zbar)
    ZZT_adj = (Zbar - np.sum(w_c.reshape((-1, 1)) * (resp ** 2), axis=0))
    ZZT.flat[::(ZZT_adj.size+1)] += ZZT_adj
    return Zbar, ZZT

def calc_Zbar_ZZT(resp, w_c):
    Nd = np.sum(w_c)
    Zbar, ZZT = calc_Zbar_ZZT_unnorm(resp, w_c)
    return Zbar / Nd, ZZT / (Nd ** 2)

def calc_Zbar_ZZT_manyDocs(resp, w_c, doc_range):
    Zbar, ZZT = [], []
    for start, end in zip(doc_range[:-1], doc_range[1:]):
        resp_d = resp[start:end]
        wc_d = w_c[start:end]

        Zbar_d, ZZT_d = calc_Zbar_ZZT(resp_d, wc_d)
        Zbar.append(Zbar_d)
        ZZT.append(ZZT_d)

    return np.stack(Zbar), np.stack(ZZT)

