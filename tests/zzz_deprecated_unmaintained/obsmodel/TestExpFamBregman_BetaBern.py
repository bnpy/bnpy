import numpy as np

from scipy.special import gammaln, digamma
from bnpy.viz import PlotUtil

pylab = PlotUtil.pylab
PlotUtil.ConfigPylabDefaults(
    pylab, 
    **{'figure.subplot.left':0.23,
       'figure.subplot.bottom':0.23})

def phi2mu(phi_grid):
    mu_grid = np.exp(phi_grid) / (np.exp(phi_grid) + 1.0)
    np.minimum(mu_grid, 1-1e-10, out=mu_grid)
    np.maximum(mu_grid, 1e-10, out=mu_grid)
    return mu_grid

def mu2phi(mu_grid):
    phi_grid = np.log(mu_grid) - np.log(1-mu_grid)
    return phi_grid

def c_Phi(phi_grid):
    c_grid = np.log(np.exp(phi_grid) + 1.0)
    return c_grid

def c_Mu(mu_grid):
    grid_1mu = (1-mu_grid) 
    c_grid = mu_grid * np.log(mu_grid) + grid_1mu * np.log(grid_1mu)
    return c_grid

def logJacobian_mu(mu_grid):
    ''' Compute log of Jacobian of transformation function mu2phi

    J : 1D array, size ngrid
        J[n] = log Jacobian(mu[n])
    '''
    return - np.log(mu_grid) - np.log(1-mu_grid)

def bregmanDiv(mu_grid, mu):
    '''

    Returns
    -------
    div : 1D array, size ngrid
    '''
    div = c_Mu(mu_grid) - c_Mu(mu) - (mu_grid - mu) * mu2phi(mu)
    return div

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
        pylab.savefig('BetaBern_bregmanDiv.eps')
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
    


if __name__ == '__main__':
    muRange = [0.02, 0.1, 0.5, 0.9, 0.98]
    nuRange = [1/2.0, 1, 2.0, 8, 32, 128]
    mu_Phi = 0.7
    print("mu(Mode[phi]): ", mu_Phi)
    print("Mode[phi]: ", mu2phi(mu_Phi))

    makePlot_bregmanDiv_Mu(mu=muRange, ngrid=10000)
    
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
