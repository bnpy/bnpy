import numpy as np

from scipy.special import gammaln, digamma
from bnpy.util import as2D, as1D
from bnpy.viz import PlotUtil

pylab = PlotUtil.pylab
PlotUtil.ConfigPylabDefaults(
    pylab, 
    **{'figure.subplot.left':0.23,
       'figure.subplot.bottom':0.23})

def phi2mu(phi_grid, N):
    ''' Compute cumulant function at given phi value.

    Args
    ----
    phiG : 2D array, shape G x W-1
        if 1D, will be cast to 1 x W-1

    Returns
    -------
    muG : 2D array, shape G x W
        Each row of muG must sum to N and be non-negative

    Examples
    --------
    >>> np.set_printoptions(precision=4, suppress=False)
    >>> print phi2mu([0], 1)
    [[ 0.5  0.5]]
    >>> print phi2mu([0, 0], 1)
    [[ 0.3333  0.3333  0.3333]]
    >>> print phi2mu([5], 1)
    [[ 0.9933  0.0067]]
    '''
    phiG = as2D(np.asarray(phi_grid))
    muG = np.ones((phiG.shape[0], phiG.shape[1]+1))
    muG[:, :-1] = np.exp(phiG)
    muG /= np.sum(muG, axis=1)[:,np.newaxis]
    np.minimum(muG, 1-1e-10, out=muG)
    np.maximum(muG, 1e-10, out=muG)
    muG *= N
    assert np.allclose(np.sum(muG, axis=1), N)
    return muG

def mu2phi(mu_grid, N):
    muG = as2D(np.asarray(mu_grid))
    phiG = np.log(muG[:, :-1])
    phiG -= np.log(muG[:, -1])[:,np.newaxis]
    return phiG

def c_Phi(phi_grid, N):
    ''' Compute cumulant function at given phi value.

    Args
    ----
    phi_grid : 2D array, shape G x W-1
        if 1D, will be cast to 1 x W-1

    Returns
    -------
    c_grid : 1D array, shape G
    '''
    phi_grid = as2D(np.asarray(phi_grid))
    c_grid = N * np.log(1.0 + np.sum(np.exp(phi_grid), axis=1))
    return c_grid

def c_Mu(mu_grid, N):
    ''' Compute mean-cumulant function at given mu value.

    Args
    ----
    mu_grid : 2D array, shape G x W
        if 1D, will be cast to 1 x W

    Returns
    -------
    c_grid : 1D array, shape G
    '''
    muG = as2D(np.asarray(mu_grid))
    assert np.allclose(N, muG.sum(axis=1))
    c_grid = np.sum(muG * np.log(muG + 1e-100), axis=1)
    c_grid -= N * np.log(N)
    return c_grid

def bregmanDiv(muA, muB, N, W=2):
    ''' Compute Bregman divergence between two mean parameters.

    Args
    ----
    muA : 2D array, size GA x W
    muB : 2D array, size GB x W

    Returns
    -------
    Div : 2D array, size GA x GB
        Div[a,b] = divergence between muA[a] and muB[b]
    '''
    muA = as2D(np.asarray(muA))
    muB = as2D(np.asarray(muB))
    assert np.allclose(np.sum(muA,axis=1), N)
    assert np.allclose(np.sum(muB,axis=1), N)

    cA = c_Mu(muA, N)
    cB = c_Mu(muB, N)
    Div = np.zeros((muA.shape[0], muB.shape[0]))
    Div += cA[:,np.newaxis]
    Div -= cB[np.newaxis,:]
    for b in range(muB.shape[0]):
        phi_b = mu2phi(muB[b], N)
        for a in range(muA.shape[0]):
            muDiff = muA[a, :-1] - muB[b, :-1]
            Div[a,b] -= np.inner(muDiff, phi_b)
            # Double check
            d_ab = np.inner(muA[a,:], np.log(muA[a,:]+1e-100)) - \
                np.inner(muA[a,:], np.log(muB[b,:]+1e-100))
            assert np.allclose(Div[a,b], d_ab)

    #    Div[a,:] -= np.sum((muA[a, :-1] - muB[:,:-1]) * mu2phi(muB, N), axis=0)
    #for a in range(muA.shape[0]):
    #    Div[a,:] -= np.sum((muA[a, :-1] - muB[:,:-1]) * mu2phi(muB, N), axis=0)
    return Div

def makePlot_pdf_Phi(
        nu=0, tau=0, phi_grid=None,
        ngrid=1000, min_phi=-100, max_phi=100):
    label = 'nu=%7.2f' % (nu)
    cPrior = - gammaln(nu) + gammaln(nu-tau) + gammaln(tau)
    if phi_grid is None:
        phi_grid = np.linspace(min_phi, max_phi, ngrid)
    logpdf_grid = tau * phi_grid - nu * c_Phi(phi_grid) - cPrior
    pdf_grid = np.exp(logpdf_grid)
    IntegralVal = np.trapz(pdf_grid, phi_grid)
    mu_grid = phi2mu(phi_grid)
    ExpectedPhiVal = np.trapz(pdf_grid * phi_grid, phi_grid)
    ExpectedMuVal = np.trapz(pdf_grid * mu_grid, phi_grid)
    print('%s Integral=%.4f E[phi]=%6.3f E[mu]=%.4f' % (
        label, IntegralVal, ExpectedPhiVal, ExpectedMuVal))
    pylab.plot(phi_grid, pdf_grid, '-', label=label)
    pylab.xlabel('phi (log odds ratio)')
    pylab.ylabel('density p(phi)')

def makePlot_pdf_Mu(
        nu=0, tau=0, phi_grid=None,
        ngrid=1000, min_phi=-100, max_phi=100):
    label = 'nu=%7.2f' % (nu,)
    cPrior = - gammaln(nu) + gammaln(nu-tau) + gammaln(tau)

    mu_grid = np.linspace(1e-15, 1-1e-15, ngrid)
    phi_grid = mu2phi(mu_grid)
    logpdf_grid = tau * phi_grid - nu * c_Phi(phi_grid) - cPrior
    logJacobian_grid = logJacobian_mu(mu_grid)
    pdf_grid = np.exp(logpdf_grid + logJacobian_grid)
    IntegralVal = np.trapz(pdf_grid, mu_grid)
    ExpectedMuVal = np.trapz(pdf_grid * mu_grid, mu_grid)
    ExpectedPhiVal = np.trapz(pdf_grid * phi_grid, mu_grid)
    print('%s Integral=%.4f E[phi]=%6.3f E[mu]=%.4f' % (
        label, IntegralVal, ExpectedPhiVal, ExpectedMuVal))
    pylab.plot(mu_grid, pdf_grid, '-', label=label)
    pylab.xlabel('mu')
    pylab.ylabel('density p(mu)')


def makePlot_cumulant_Phi(
        phi_grid=None,
        ngrid=100, min_phi=-10, max_phi=10):
    if phi_grid is None:
        phi_grid = np.linspace(min_phi, max_phi, ngrid)
    c_grid = c_Phi(phi_grid)
    pylab.plot(phi_grid, c_grid, 'k-')
    pylab.xlabel('phi (log odds ratio)')
    pylab.ylabel('cumulant c(phi)')

def makePlot_cumulant_Mu(
        mu_grid=None, ngrid=100):
    if mu_grid is None:
        mu_grid = np.linspace(1e-15, 1-1e-15, ngrid)
    c_grid = c_Mu(mu_grid)
    pylab.plot(mu_grid, c_grid, 'r-')
    pylab.xlabel('mu')
    pylab.ylabel('gamma(mu)')


def makePlot_bregmanDiv_Mu(
        mu_grid=None, mu=0.5, ngrid=100):
    if isinstance(mu, list) and len(mu) > 1:
        pylab.figure()
        for m in mu:
            makePlot_bregmanDiv_Mu(mu=m, ngrid=5000)
        pylab.legend(loc='upper right', fontsize=13)
        pylab.ylim([-.01, 1.6])
        pylab.xlim([0, 1.0])
        return
    label = 'mu=%.2f' % (mu)
    if mu_grid is None:
        mu_grid = np.linspace(1e-15, 1-1e-15, ngrid)
    div_grid = bregmanDiv(mu_grid, mu)
    pylab.plot(mu_grid, div_grid, '-', linewidth=1, label=label)
    pylab.xlabel('x')
    pylab.ylabel('Bregman div.  D(x, mu)')


def makePlot_pdf_Mu_range(nuRange):
    pylab.figure()
    for nu in nuRange[::-1]:
        tau = mu_Phi * nu
        makePlot_pdf_Mu(nu=nu, tau=tau, 
            ngrid=500000)
    pylab.legend(loc='upper left', fontsize=13)
    pylab.ylim([0, 10])
    pylab.savefig('BetaBern_densityMu.eps')

def makePlot_pdf_Phi_range(nuRange):
    pylab.figure()
    for nu in nuRange[::-1]:
        tau = mu_Phi * nu
        makePlot_pdf_Phi(nu=nu, tau=tau, 
            ngrid=500000)
    pylab.legend(loc='upper left', fontsize=13)
    pylab.ylim([0, 2.0]); pylab.yticks([0, 1, 2])
    pylab.xlim([-10, 6])
    pylab.savefig('BetaBern_densityPhi.eps')

def h(xVec, N):
    assert np.sum(xVec) == N
    return gammaln(N+1) - np.sum(gammaln(xVec+1))

def pdf_Phi(xVec, phiVec, N):
    ''' Evaluate the pdf at a scalar phi
    '''
    assert np.allclose(xVec.sum(), N)
    logpdf = np.inner(xVec[:-1], phiVec) - c_Phi(phiVec, N) + h(xVec, N) 
    return np.exp(logpdf)

def pdf_Mu(xVec, muVec, N):
    ''' Evaluate the pdf at a scalar phi
    '''
    assert np.allclose(xVec.sum(), N)
    logpdf = -1*bregmanDiv(xVec, muVec,N) + c_Mu(xVec, N) + h(xVec, N) 
    return np.exp(logpdf)

def generateAllXForFixedN(N, W=2):
    for n in range(N+1):
        yield np.asarray([n, N-n])

if __name__ == '__main__':
    N = 7
    muG = np.asarray([0.02, 0.1, 0.5, 0.9, 0.98])
    muG = N * np.vstack([muG, 1.0-muG]).T.copy()
    print(muG)

    phiG = mu2phi(muG, N)
    print(phiG)
    print(phi2mu(mu2phi(muG, N), N))

    print('BREGMAN Dist Mat:')
    print(bregmanDiv(muG, muG, N))

    for row in range(phiG.shape[0]):
        phi = as1D(phiG[row])
        mu = phi2mu(phi,N)

        print('phi = %.3f' % (phi))
        print('mu = ', mu)
        print('----------')
        cdf = 0.0
        for xVec in generateAllXForFixedN(N):
            pdf = pdf_Phi(xVec, phi, N)
            cdf += pdf
            print('%.3f %.3f %.3f %s' % (
                cdf,
                pdf,
                pdf_Mu(xVec, mu, N),
                xVec))
            #print pdf_Phi(xVec, mu2phi(mu,N), N),
        print('')

    '''
    makePlot_pdf_Mu_range(nuRange)
    makePlot_pdf_Phi_range(nuRange)


    pylab.figure()
    makePlot_cumulant_Phi(ngrid=1000)
    pylab.savefig('BetaBern_cumulantPhi.eps')

    pylab.figure()
    makePlot_cumulant_Mu(ngrid=1000)
    pylab.savefig('BetaBern_cumulantMu.eps')
    '''
    pylab.show()
