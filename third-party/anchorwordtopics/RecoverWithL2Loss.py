import time
import sys
from numpy import *


def nonNegativeRecoverTopics(Q, anchors, divergence,
                             initial_stepsize=1, eps=10**(-7)):
    """ Recover topic-by-word matrix given Q and chosen anchor words.

        Returns
        --------
        topics : 2D array, size K x V. Each row sums to one.
    """
    V = Q.shape[0]
    K = len(anchors)

    # Compute p(word) vector before Q is normalized
    P_w = dot(Q, ones(V))
    P_w[isnan(P_w)] = 1e-16

    # Normalize the rows of Q
    Q /= (Q.sum(axis=1) + 1e-14)[:, newaxis]

    X = Q[anchors, :]
    XXT = dot(X, X.transpose())

    A = zeros((K, V))
    for w in range(V):
        y = Q[w, :]
        alpha = fastRecover(y, X, w, anchors, divergence,
                            XXT, initial_stepsize, eps)
        A[:, w] = alpha

    # Rescale A matrix
    topics = dot(A, diag(P_w))

    # Normalize rows
    topics /= topics.sum(axis=1)[:, newaxis] + 1e-14
    return topics


def fastRecover(y, x, v, anchors, divergence, XXT, initial_stepsize, epsilon):
    K = len(anchors)
    alpha = zeros(K)
    gap = None

    if v in anchors:
        alpha[anchors.index(v)] = 1
        it = -1
        dist = 0
        stepsize = 0

    else:
        try:
            if divergence == "L2":
                alpha, it, dist, stepsize, gap = quadSolveExpGrad(
                    y, x, epsilon, None, XXT)

            else:
                print("invalid divergence!")
                assert(0)
            if isnan(alpha).any():
                alpha = ones(K) / K

        except Exception as inst:
            print(type(inst))     # the exception instance
            print(inst.args)      # arguments stored in .args
            alpha = ones(K) / K
            it = -1
            dist = -1
            stepsize = -1

    return alpha


def quadSolveExpGrad(y, x, eps, alpha=None, XX=None):
    c1 = 10**(-4)
    c2 = 0.75
    if XX is None:
        print('making XXT')
        XX = dot(x, x.transpose())

    XY = dot(x, y)
    YY = float(dot(y, y))

    #start_time = time.time()
    #y_copy = copy(y)
    #x_copy = copy(x)

    (K, n) = x.shape
    if alpha is None:
        alpha = ones(K) / K

    old_alpha = copy(alpha)
    log_alpha = log(alpha)
    old_log_alpha = copy(log_alpha)

    it = 1
    aXX = dot(alpha, XX)
    aXY = float(dot(alpha, XY))
    aXXa = float(dot(aXX, alpha.transpose()))

    grad = 2 * (aXX - XY)
    new_obj = aXXa - 2 * aXY + YY

    old_grad = copy(grad)

    stepsize = 1
    repeat = False
    decreased = False
    gap = float('inf')
    while 1:
        eta = stepsize
        old_obj = new_obj
        old_alpha = copy(alpha)
        old_log_alpha = copy(log_alpha)
        if new_obj == 0:
            break
        if stepsize == 0:
            break

        it += 1

        # update
        log_alpha -= eta * grad
        # normalize
        log_alpha -= logsum_exp(log_alpha)
        # compute new objective
        alpha = exp(log_alpha)

        aXX = dot(alpha, XX)
        aXY = float(dot(alpha, XY))
        aXXa = float(dot(aXX, alpha.transpose()))

        old_obj = new_obj
        new_obj = aXXa - 2 * aXY + YY
        # sufficient decrease
        changeVal = c1 * stepsize * dot(grad, alpha - old_alpha)
        if not new_obj <= old_obj + changeVal:
            stepsize /= 2.0  # reduce stepsize
            alpha = old_alpha
            log_alpha = old_log_alpha
            new_obj = old_obj
            repeat = True
            decreased = True
            continue

        # compute the new gradient
        old_grad = copy(grad)
        grad = 2 * (aXX - XY)

        # curvature
        cval = c2 * dot(old_grad, alpha - old_alpha)
        if (not dot(grad, alpha - old_alpha) >= cval) and (not decreased):
            stepsize *= 2.0  # increase stepsize
            alpha = old_alpha
            log_alpha = old_log_alpha
            grad = old_grad
            new_obj = old_obj
            repeat = True
            continue

        decreased = False

        lam = copy(grad)
        lam -= lam.min()

        gap = dot(alpha, lam)
        convergence = gap
        if (convergence < eps):
            break

    return alpha, it, new_obj, stepsize, gap


def logsum_exp(y):
    m = y.max()
    return m + log((exp(y - m)).sum())
