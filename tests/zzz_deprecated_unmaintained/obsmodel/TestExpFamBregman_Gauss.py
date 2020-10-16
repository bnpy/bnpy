import numpy as np

from scipy.special import gammaln, digamma
from bnpy.viz import PlotUtil

pylab = PlotUtil.pylab
PlotUtil.ConfigPylabDefaults(
    pylab, 
    **{'figure.subplot.left':0.23,
       'figure.subplot.bottom':0.23})

def make_phi1_grid(ngrid=1000, min_phi=-20, max_phi=-1e-10):
    phi1_grid = np.linspace(min_phi, max_phi, ngrid)
    return phi1_grid

def make_phi2_grid(ngrid=1000, min_phi2=-20, max_phi2=20):
    phi2_grid = np.linspace(min_phi2, max_phi2, ngrid)
    return phi2_grid

def make_mu1_grid(ngrid=1000, min_mu=1e-10, max_mu=30):
    mu1_grid = np.linspace(min_mu, max_mu, ngrid)
    return mu1_grid

def make_mu2_grid(ngrid=1000, min_mu=-30, max_mu=30):
    mu2_grid = np.linspace(min_mu, max_mu, ngrid)
    return mu2_grid

def phi2mu(phi1_grid, phi2_grid):
    inv_phi1 = 1.0 / phi2_grid
    mu1_grid = -0.5 * inv_phi1 + \
        0.25 * np.square(inv_phi1) * np.square(phi2_grid)
    mu2_grid = -0.5 * inv_phi1 * phi2_grid
    return mu1_grid, mu2_grid

def mu2phi(mu1_grid, mu2grid):
    inv_mu = 1.0/(mu1_grid - np.square(mu2_grid))
    phi1_grid = -0.5 * inv_mu
    phi2_grid = mu2_grid * inv_mu
    return phi1_grid, phi2_grid

def c_Phi(phi1_grid, phi2_grid):
    c_grid = - 0.5 * np.log(- phi1_grid) - 0.25 * np.square(phi2_grid)/phi1_grid
    return c_grid

def c_Mu(mu1_grid, mu2_grid):
    c_grid = - 0.5 * np.log(mu1_grid - np.square(mu2_grid))
    return c_grid

def E_phi1(nu=0, tau=0):
    return - (0.5 * nu + 1.0) / tau

def bregmanDiv(mu1_grid, mu2_grid, mu1, mu2):
    '''

    Returns
    -------
    div : 1D array, size ngrid
    '''
    phi1, phi2 = mu2phi(mu)
    div = c_Mu(mu1_grid, mu2_grid) - c_Mu(mu1, mu2) - \
        (mu1_grid - mu1) * phi1 - \
        (mu2_grid - mu2) * phi2
    return div

def c_Prior(nu=0, tau=0):
    scaled_nu = 0.5 * nu + 1
    return gammaln(scaled_nu) - scaled_nu * np.log(tau)

def logJacobian_mu(mu_grid):
    '''
    Returns
    -------
    J : 1D array, size ngrid
        J[n] = log Jacobian(mu[n], mu2phi)
        Jacobian(mu, mu2phi) = d/dmu phi(mu)
    '''
    pass

def checkFacts_pdf_Phi(
        nu=0, tau=0, phi_grid=None, **kwargs):
    cPrior = c_Prior(nu=nu, tau=tau)
    if phi_grid is None:
        phi_grid = make_phi_grid(**kwargs)
    logpdf_grid = tau * phi_grid - nu * c_Phi(phi_grid) - cPrior
    pdf_grid = np.exp(logpdf_grid)
    mu_grid = phi2mu(phi_grid)

    IntegralVal = np.trapz(pdf_grid, phi_grid)
    E_phi_numeric = np.trapz(pdf_grid * phi_grid, phi_grid)
    E_phi_formula = -(0.5 * nu + 1) / tau
    E_mu_numeric = np.trapz(pdf_grid * mu_grid, phi_grid)
    E_mu_formula = tau/nu
    mode_phi_numeric = phi_grid[np.argmax(pdf_grid)]
    mode_phi_formula = mu2phi(tau/nu)

    E_c_numeric = np.trapz(pdf_grid * c_Phi(phi_grid), phi_grid)
    E_c_formula = - 0.5 * digamma(0.5 * nu + 1) + 0.5 * np.log(tau)

    print("nu=%7.3f tau=%7.3f" % (nu, tau))
    print("     Integral=% 7.3f   should be % 7.3f" % (IntegralVal, 1.0))
    print("        E[mu]=% 7.3f   should be % 7.3f" % (E_mu_numeric, E_mu_formula))
    print("       E[phi]=% 7.3f   should be % 7.3f" % (E_phi_numeric, E_phi_formula))
    print("    E[c(phi)]=% 7.3f   should be % 7.3f" % (E_c_numeric, E_c_formula))
    print("    mode[phi]=% 7.3f   should be % 7.3f" % (
        mode_phi_numeric, mode_phi_formula))

def makePlot_pdf_Phi(
        nu=0, tau=0, phi_grid=None, **kwargs):
    label = 'nu=%7.2f' % (nu)
    cPrior = c_Prior(nu=nu, tau=tau)
    if phi_grid is None:
        phi_grid = make_phi_grid(**kwargs)
    logpdf_grid = tau * phi_grid - nu * c_Phi(phi_grid) - cPrior
    pdf_grid = np.exp(logpdf_grid)
    pylab.plot(phi_grid, pdf_grid, '-', label=label)
    pylab.xlabel('phi')
    pylab.ylabel('density p(phi)')

def makePlot_pdf_Mu(
        nu=0, tau=0, mu_grid=None, **kwargs):
    label = 'nu=%7.2f' % (nu,)
    cPrior = c_Prior(nu=nu, tau=tau)
    if mu_grid is None:
        mu_grid = make_mu_grid(**kwargs)
    phi_grid = mu2phi(mu_grid)
    logpdf_grid = tau * phi_grid - nu * c_Phi(phi_grid) - cPrior
    logJacobian_grid = logJacobian_mu(mu_grid)
    pdf_grid = np.exp(logpdf_grid + logJacobian_grid)
    pylab.plot(mu_grid, pdf_grid, '-', label=label)
    pylab.xlabel('mu')
    pylab.ylabel('density p(mu)')

def checkFacts_pdf_Mu(nu=0, tau=0, mu_grid=None, **kwargs):
    cPrior = c_Prior(nu=nu, tau=tau)
    if mu_grid is None:
        mu_grid = make_mu_grid(**kwargs)
    phi_grid = mu2phi(mu_grid)
    logpdf_grid = tau * phi_grid - nu * c_Phi(phi_grid) - cPrior
    logJacobian_grid = logJacobian_mu(mu_grid)
    pdf_grid = np.exp(logpdf_grid + logJacobian_grid)
    Integral = np.trapz(pdf_grid, mu_grid)
    E_mu_numeric = np.trapz(pdf_grid * mu_grid, mu_grid)
    E_mu_formula = tau/nu
    E_phi_numeric = np.trapz(pdf_grid * phi_grid, mu_grid)
    E_phi_formula = E_phi(nu,tau)
    print("nu=%7.3f tau=%7.3f" % (nu, tau))
    print("     Integral=% 7.3f   should be % 7.3f" % (Integral, 1.0))
    print("        E[mu]=% 7.3f   should be % 7.3f" % (E_mu_numeric, E_mu_formula))
    print("       E[phi]=% 7.3f   should be % 7.3f" % (E_phi_numeric, E_phi_formula))

def makePlot_cumulant_Phi(
        phi1=None, phi2=None, **kwargs):
    if phi1 is None:
        phi1 = make_phi1_grid(**kwargs)
        grid = phi1
        label = 'phi2 = %7.3f' % (phi2)
        xlabel = 'phi_1'
    if phi2 is None:
        phi2 = make_phi2_grid(**kwargs)
        grid = phi2
        label = 'phi1 = %7.3f' % (phi1)
        xlabel = 'phi_2'
    c_grid = c_Phi(phi1, phi2)
    pylab.plot(grid, c_grid, '-', label=label)
    pylab.xlabel(xlabel)
    pylab.ylabel('c(phi)')

def makePlot_cumulant_Mu(
        mu1=None, mu2=None, **kwargs):
    if mu1 is None:
        mu1 = make_mu1_grid(**kwargs)
        grid = mu1
        label = 'mu2 = %7.3f' % (mu2)
        xlabel = 'mu1'
    if mu2 is None:
        mu2 = make_mu2_grid(**kwargs)
        grid = mu2
        label = 'mu1 = %7.3f' % (mu1)
        xlabel = 'mu2'
    c_grid = c_Mu(mu1, mu2)
    pylab.plot(grid, c_grid, '-', label=label)
    pylab.xlabel(xlabel)
    pylab.ylabel('gamma(mu)')

def makePlot_cumulant_Prior(
        nu_grid=None, tau=5.0, ngrid=10000, **kwargs):
    if nu_grid is None:
        nu_grid = np.linspace(1e-10, 100, ngrid)
    c_grid = c_Prior(nu_grid, tau)
    pylab.plot(nu_grid, c_grid, 'k:')
    pylab.xlabel('nu')
    pylab.ylabel('prior cumulant(nu, tau)')

def makePlot_bregmanDiv_Mu(
        mu_grid=None, mu=0.5, **kwargs):
    label = 'mu=%.2f' % (mu)
    if mu_grid is None:
        mu_grid = make_mu_grid(**kwargs)
    div_grid = bregmanDiv(mu_grid, mu)
    pylab.plot(mu_grid, div_grid, '--', linewidth=2, label=label)
    pylab.xlabel('x')
    pylab.ylabel('Bregman div.  D(x, mu)')

def bregman_FVG(xgrid, muMean, muVar=1.0):
    ''' Compute zero-mean gaussian divergence.
    '''
    return 0.5 / muVar * (xgrid - muMean)**2

def bregman_ZMG(xgrid, muVar):
    ''' Compute zero-mean gaussian divergence.
    '''
    return - 0.5 \
        - 0.5 * np.log(xgrid) \
        + 0.5 * np.log(muVar) \
        + 0.5 * (1.0/muVar) * xgrid \

def Mstep(X=None, nu=2.0, priorVar=1.0, kappa=5.0, priorMean=0.5):
    X = np.asarray(X)
    sum_w = X.size
    sum_wx = np.sum(X)
    sum_wxxT = np.sum(np.square(X))
    var_wx = sum_wxxT/sum_w - np.square(sum_wx/sum_w)

    # Analytic optima
    muVar_opt_ZMG = (sum_wxxT + priorVar*nu) / (sum_w + nu)
    muMean_opt_FVG = (sum_wx + priorMean * kappa) / (sum_w + kappa)

    prior_kmm = priorMean * priorMean * kappa
    post_kmm = muMean_opt_FVG * muMean_opt_FVG * (kappa + sum_w)
    muVar_opt_G = (sum_wxxT + priorVar * nu + prior_kmm - post_kmm) \
        / (sum_w + nu)

    min_muMean = muMean_opt_FVG - 5 * np.maximum(np.sqrt(muVar_opt_G), 0.1)
    max_muMean = muMean_opt_FVG + 5 * np.maximum(np.sqrt(muVar_opt_G), 0.1)

    muVar_grid = np.linspace(0.001, 2 * muVar_opt_ZMG, 100000)
    muMean_grid = np.linspace(min_muMean, max_muMean, 100000)


    print('ZeroMeanGauss')
    print('=============')
    DistMat_ZMG = bregman_ZMG(np.square(X), muVar_grid[:,np.newaxis])
    Ldist_ZMG = DistMat_ZMG.sum(axis=1) \
        + nu * bregman_ZMG(priorVar, muVar_grid)
    print("mode[muVar]:  numeric % .3f  formula % .3f" % (
        muVar_grid[Ldist_ZMG.argmin()], muVar_opt_ZMG))


    print('')
    print('FixedVarGauss')
    print('=============')
    for muVar in [1.0, 2.3, 5.7]:
        DistMat_FVG = bregman_FVG(X, muMean_grid[:,np.newaxis], 
                                  muVar=muVar)
        Ldist_FVG = DistMat_FVG.sum(axis=1) \
            + kappa * bregman_FVG(priorMean,
                                  muMean_grid, 
                                  muVar=muVar)
        if Ldist_FVG.argmin() == 0:
            print('WARNING: hit lower boundary in numeric search for max')
        if Ldist_FVG.argmin() == Ldist_FVG.size - 1:
            print('WARNING: hit upper boundary in numeric search for max')
        print("muVar %.3f" % (muVar), end=' ')
        print("mode[muMean]:  numeric % .3f  formula % .3f" % (
            muMean_grid[Ldist_FVG.argmin()], muMean_opt_FVG))


    print('')
    print('Unk Mean and Variance')
    print('=====================')
    DistMat_FVG = bregman_FVG(X, 
                              muMean_opt_FVG,
                              muVar=muVar_grid[:,np.newaxis])
    # epsvec = 1e-13 * np.square(X)
    epsvec = 1e-10 * np.ones_like(X)
    DistMat_ZMG = bregman_ZMG(epsvec, muVar_grid[:,np.newaxis])
    Ldist_G = DistMat_ZMG.sum(axis=1) + DistMat_FVG.sum(axis=1) \
        + nu * bregman_ZMG(priorVar, muVar_grid) \
        + kappa * bregman_FVG(priorMean, muMean_opt_FVG, muVar=muVar_grid)

    print("mode[muVar]:  numeric % .3f  formula % .3f" % (
        muVar_grid[Ldist_G.argmin()], muVar_opt_G))

    print('')
    print('Unk Mean and Variance (prior only)')
    print('=====================')
    muVar_opt_prioronly = priorVar
    Ldist_prioronly = \
        + nu * bregman_ZMG(priorVar, muVar_grid) \
        + kappa * bregman_FVG(priorMean, priorMean, muVar=muVar_grid)
    print("mode[muVar]:  numeric % .3f formula % .3f" % (
        muVar_grid[Ldist_prioronly.argmin()],
        muVar_opt_prioronly))

    print('')
    print('Unk Mean and Variance (optimize muMean)')
    print('=====================')
    DistMat_FVG = bregman_FVG(X, 
                              muMean_grid[:,np.newaxis],
                              muVar=muVar_opt_G)
    # epsvec = 1e-13 * np.square(X)
    epsvec = 1e-12 * np.ones_like(X)
    DistMat_ZMG = bregman_ZMG(epsvec, muVar_opt_G)[np.newaxis, :]

    Ldist_G_bymuMean = DistMat_ZMG.sum(axis=1) + DistMat_FVG.sum(axis=1) \
        + nu * bregman_ZMG(priorVar, muVar_opt_G) \
        + kappa * bregman_FVG(priorMean, muMean_grid, muVar=muVar_opt_G)

    print("mode[muVar]:  numeric % .3f formula % .3f" % (
        muMean_grid[Ldist_G_bymuMean.argmin()],
        muMean_opt_FVG))

    '''

    pylab.plot(muVar_grid, Ldist_ZMG, 'r-', label='ZMG')
    pylab.plot(muVar_grid, Ldist_G, 'b-', label='G')
    pylab.legend(loc='upper right')
    pylab.show(block=0)
    from IPython import embed; embed()
    '''

if __name__ == '__main__':
    pylab.ion()

    PRNG = np.random.RandomState()
    Mstep(X= PRNG.randn(50) * -10  + 3)
    '''
    pylab.figure()
    for mu2 in [-2.1, -1.1, 0, 1, 2, 4][::-1]:
        makePlot_cumulant_Mu(mu2=mu2, min_mu=np.square(mu2)+1e-10, max_mu=100)
    pylab.legend(loc='upper right', fontsize=13)
    pylab.xlim([-2, 100])

    pylab.show()
    '''
