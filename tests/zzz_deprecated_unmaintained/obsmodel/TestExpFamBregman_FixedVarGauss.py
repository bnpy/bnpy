import numpy as np

from scipy.special import gammaln, digamma
from bnpy.viz import PlotUtil

pylab = PlotUtil.pylab
PlotUtil.ConfigPylabDefaults(
    pylab, 
    **{'figure.subplot.left':0.23,
       'figure.subplot.bottom':0.23})

def make_phi_grid(ngrid=1000, min_phi=-20, max_phi=20, **kwargs):
    phi_grid = np.linspace(min_phi, max_phi, ngrid)
    return phi_grid

def make_mu_grid(ngrid=1000, min_mu=-20, max_mu=20, **kwargs):
    mu_grid = np.linspace(min_mu, max_mu, ngrid)
    return mu_grid

def phi2mu(phi_grid, fixedVar=1.0, **kwargs):
    mu_grid = fixedVar * phi_grid
    return mu_grid

def mu2phi(mu_grid, fixedVar=1.0, **kwargs):
    phi_grid = mu_grid / fixedVar
    return phi_grid

def c_Phi(phi_grid, fixedVar=1.0, **kwargs):
    c_grid = 0.5 * fixedVar * np.square(phi_grid)
    return c_grid

def c_Mu(mu_grid, fixedVar=1.0, **kwargs):
    c_grid = 0.5 * np.square(mu_grid) / fixedVar
    return c_grid

def E_phi(nu=0, tau=0, fixedVar=1.0, **kwargs):
    return tau / nu / fixedVar

def bregmanDiv(mu_grid, mu, fixedVar=1.0, **kwargs):
    '''

    Returns
    -------
    div : 1D array, size ngrid
    '''
    div = c_Mu(mu_grid, fixedVar) - c_Mu(mu, fixedVar) \
        - (mu_grid - mu) * mu2phi(mu, fixedVar)
    return div

def c_Prior(nu=0, tau=0, fixedVar=1.0, **kwargs):
    return \
        0.5 / fixedVar * tau * tau /nu \
        - 0.5 * np.log(fixedVar*nu) \
        + 0.5 * np.log(2*np.pi)

def logJacobian_mu(mu_grid, fixedVar=1.0, **kwargs):
    '''
    Returns
    -------
    J : 1D array, size ngrid
        J[n] = log Jacobian(mu[n], mu2phi)
        Jacobian(mu, mu2phi) = d/dmu phi(mu)
    '''
    return - np.log(fixedVar)

def checkFacts_pdf_Phi(
        nu=0, tau=0, phi_grid=None, **kwargs):
    pdf_grid, phi_grid = make_pdfgrid_phi(nu, tau, phi_grid, **kwargs)
    mu_grid = phi2mu(phi_grid, **kwargs)

    IntegralVal = np.trapz(pdf_grid, phi_grid)
    E_phi_numeric = np.trapz(pdf_grid * phi_grid, phi_grid)
    E_phi_formula = E_phi(nu=nu, tau=tau, **kwargs)
    E_mu_numeric = np.trapz(pdf_grid * mu_grid, phi_grid)
    E_mu_formula = tau/nu
    mode_phi_numeric = phi_grid[np.argmax(pdf_grid)]
    mode_phi_formula = mu2phi(tau/nu, **kwargs)

    print("nu=%7.3f tau=%7.3f" % (nu, tau))
    print("     Integral=% 7.3f   should be % 7.3f" % (IntegralVal, 1.0))
    print("        E[mu]=% 7.3f   should be % 7.3f" % (E_mu_numeric, E_mu_formula))
    print("       E[phi]=% 7.3f   should be % 7.3f" % (E_phi_numeric, E_phi_formula))
    print("    mode[phi]=% 7.3f   should be % 7.3f" % (
        mode_phi_numeric, mode_phi_formula))

def makePlot_pdf_Phi(
        nu=0, tau=0, phi_grid=None, **kwargs):
    pdf_grid, phi_grid = make_pdfgrid_phi(nu, tau, phi_grid, **kwargs)
    # Plot stuff 
    label = 'nu=%7.2f' % (nu)
    pylab.plot(phi_grid, pdf_grid, '-', label=label)
    pylab.xlabel('phi')
    pylab.ylabel('density p(phi)')

def make_pdfgrid_phi(nu=0, tau=0, phi_grid=None, **kwargs):
    cPrior = c_Prior(nu=nu, tau=tau, **kwargs)
    if phi_grid is None:
        phi_grid = make_phi_grid(**kwargs)
    logpdf_grid = tau * phi_grid - nu * c_Phi(phi_grid, **kwargs) - cPrior
    pdf_grid = np.exp(logpdf_grid)
    return pdf_grid, phi_grid

def make_pdfgrid_Mu(nu=0, tau=0, mu_grid=None, **kwargs):
    cPrior = c_Prior(nu=nu, tau=tau, **kwargs)
    if mu_grid is None:
        mu_grid = make_mu_grid(**kwargs)
    phi_grid = mu2phi(mu_grid, **kwargs)
    logpdf_grid = tau * phi_grid - nu * c_Phi(phi_grid, **kwargs) - cPrior
    logJacobian_grid = logJacobian_mu(mu_grid, **kwargs)
    pdf_grid = np.exp(logpdf_grid + logJacobian_grid)
    return pdf_grid, mu_grid, phi_grid

def makePlot_pdf_Mu(
        nu=0, tau=0, mu_grid=None, **kwargs):
    pdf_grid, mu_grid, _ = make_pdfgrid_Mu(nu, tau, mu_grid, **kwargs)
    # Plot stuff here
    label = 'nu=%7.2f' % (nu,)
    pylab.plot(mu_grid, pdf_grid, '-', label=label)
    pylab.xlabel('mu')
    pylab.ylabel('density p(mu)')

def checkFacts_pdf_Mu(nu=0, tau=0, mu_grid=None, **kwargs):
    pdf_grid, mu_grid, phi_grid = make_pdfgrid_Mu(nu, tau, mu_grid, **kwargs)
    Integral = np.trapz(pdf_grid, mu_grid)
    E_mu_numeric = np.trapz(pdf_grid * mu_grid, mu_grid)
    E_mu_formula = tau/nu
    E_phi_numeric = np.trapz(pdf_grid * phi_grid, mu_grid)
    E_phi_formula = E_phi(nu, tau, **kwargs)
    print("nu=%7.3f tau=%7.3f" % (nu, tau))
    print("     Integral=% 7.3f   should be % 7.3f" % (Integral, 1.0))
    print("        E[mu]=% 7.3f   should be % 7.3f" % (E_mu_numeric, E_mu_formula))
    print("       E[phi]=% 7.3f   should be % 7.3f" % (E_phi_numeric, E_phi_formula))

def makePlot_cumulant_Phi(
        phi_grid=None, **kwargs):
    if phi_grid is None:
        phi_grid = make_phi_grid(**kwargs)
    c_grid = c_Phi(phi_grid)
    pylab.plot(phi_grid, c_grid, 'k-')
    pylab.xlabel('phi')
    pylab.ylabel('c(phi)')

def makePlot_cumulant_Mu(
        mu_grid=None, **kwargs):
    if mu_grid is None:
        mu_grid = make_mu_grid(**kwargs)
    c_grid = c_Mu(mu_grid)
    pylab.plot(mu_grid, c_grid, 'r-')
    pylab.xlabel('mu')
    pylab.ylabel('gamma(mu)')


def makePlot_bregmanDiv_Mu(
        mu_grid=None, mu=0.5, **kwargs):
    label = 'mu= % .1f' % (mu)
    if mu_grid is None:
        mu_grid = make_mu_grid(**kwargs)
    div_grid = bregmanDiv(mu_grid, mu, **kwargs)
    pylab.plot(mu_grid, div_grid, '--', linewidth=2, label=label)
    pylab.xlabel('x')
    pylab.ylabel('Bregman div.  D(x, mu)')
    return div_grid

if __name__ == '__main__':
    fixedVar = 2.0
    nuRange = [0.5, 1, 2.0, 8, 32, 128]
    mu_chosen = 1.00

    print('Mu Density')
    print('----------')
    pylab.figure()
    for nu in nuRange[::-1]:
        tau = mu_chosen * nu
        checkFacts_pdf_Mu(
            nu=nu, tau=tau,
            min_mu=mu_chosen - 10,
            max_mu=mu_chosen + 10,
            ngrid=2e6,
            fixedVar=fixedVar)
        makePlot_pdf_Mu(
            nu=nu, tau=tau,
            min_mu=mu_chosen - 5, 
            max_mu=mu_chosen + 5,
            ngrid=10000,
            fixedVar=fixedVar)
    pylab.xlim([
        mu_chosen - 2.5 * np.sqrt(fixedVar),
        mu_chosen + 5 * np.sqrt(fixedVar) # larger right gap, for legend
        ])
    pylab.legend(loc='upper right', fontsize=13)
    pylab.savefig('FVG_densityMu.eps')
    

    print('Phi Density')
    print('-----------')
    pylab.figure()
    for nu in nuRange[::-1]:
        tau = mu_chosen * nu
        mean_phi = E_phi(nu=nu, tau=tau, fixedVar=fixedVar)
        checkFacts_pdf_Phi(
            nu=nu, tau=tau,
            min_phi=mean_phi - 9 / np.sqrt(fixedVar),
            max_phi=mean_phi + 9 / np.sqrt(fixedVar),
            ngrid=2e6,
            fixedVar=fixedVar)
        makePlot_pdf_Phi(
            nu=nu, tau=tau,
            min_phi=mean_phi - 5 / np.sqrt(fixedVar),
            max_phi=mean_phi + 5 / np.sqrt(fixedVar),
            ngrid=1000,
            fixedVar=fixedVar)
    pylab.xlim([
        mean_phi - 2.5 / np.sqrt(fixedVar),
        mean_phi + 5 / np.sqrt(fixedVar) # larger right gap, for legend
        ])
    pylab.legend(loc='upper right', fontsize=13)
    pylab.savefig('FVG_densityPhi.eps')
    

    pylab.figure()
    makePlot_cumulant_Phi(ngrid=1000)
    pylab.savefig('FVG_cumulantPhi.eps')

    pylab.figure()
    makePlot_cumulant_Mu(ngrid=1000)
    pylab.savefig('FVG_cumulantMu.eps')

    pylab.figure()
    for mu in [-4, -1, 1, 4]:
        d1 = makePlot_bregmanDiv_Mu(mu=mu, fixedVar=0.25, ngrid=5000)
    pylab.legend(loc='upper right', fontsize=13)
    pylab.ylim([0, d1[-1]])
    pylab.savefig('FVG_bregmanDiv_var=0.25.eps')


    pylab.figure()
    for mu in [-4, -1, 1, 4]:
        d2 = makePlot_bregmanDiv_Mu(mu=mu, fixedVar=1.0, ngrid=5000)
    pylab.legend(loc='upper right', fontsize=13)
    pylab.ylim([0, d1[-1]])
    pylab.savefig('FVG_bregmanDiv_var=1.eps')
    from IPython import embed; embed()
    pylab.show()
