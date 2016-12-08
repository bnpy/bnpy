=========================================================================
Zero-mean full-covariance Gaussian observation model: Variational Methods
=========================================================================

TODO update this page

Generative model
================

The diagonal Gaussian observation model generates each data vector :math:`x_n` of length D from a multivariate Gaussian with mean :math:`\mu_k \in \mathbb{R}^D` and a diagonal covariance matrix:

.. math::
    \begin{array}{c}
    x_{n1} \\
    x_{n2} \\
    \vdots \\
    x_{nD}
    \end{array}
     \sim \mathcal{N} \left(
        \begin{array}{c c c c c}
        \mu_{k1} \\
        \mu_{k2} \\
        \vdots \\
        \mu_{kD}
        \end{array}
        , 
        \begin{array}{c c c c c}
        \lambda_{k1}^{-1} \\
        & \lambda_{k2}^{-1} \\
        & & \ddots
        \\ 
        & & & & \lambda_{kD}^{-1}
        \end{array}
        \right)


Global Random Variables
-----------------------

The global random variables are the cluster-specific means and precisions (inverse variances).

For each cluster *k*, we have the following global random variables:

.. math::
    \mu_{k1}, \mu_{k2}, \ldots \mu_{kD} &\qquad \mu_{kd} \in \mathbb{R}
    \\
    \lambda_{k1}, \lambda_{k2}, \ldots \lambda_{kD} &\qquad \lambda_{kd} \in (0, +\infty)


Local Random Variables
----------------------

Each dataset observation at index *n* has its own cluster assignment:

.. math::
    z_n \in \{1, 2, \ldots K \}

The generative model and approximate posterior for :math:`z_n` is determined by an allocation model. For all computations needed by our current observation model, we'll assume either a point estimate or an approximate posterior for :math:`z_n` is known.

Normal Wishart prior
====================

Each dimension *d* has a mean :math:`\mu_{kd}` and variance :math:`\lambda_{kd}` which have a joint univariate Normal-Wishart prior with scalar hyperparameters :math:`\bar{\nu}, \bar{\beta}_d` for the Wishart prior and then :math:`\bar{m}_d, \bar{\kappa}` for the Normal prior:

.. math ::
    \lambda_{kd} &\sim \mathcal{W}_1(\bar{\nu}, \bar{\beta}_d)
    \\
    \mu_{kd} &\sim \mathcal{N}_1(\bar{m}_d, \bar{\kappa}^{-1} \lambda_{kd}^{-1})

These are represented by the following numpy array attributes of the ``Prior`` parameter bag:

* ``nu`` : float
    degrees of freedom
* ``beta`` : 1D array, size D
    scale parameters that set mean of lambda
* ``m`` : 1D array, size D
    mean of the parameter mu
* ``kappa`` : float
        scalar precision of mu

Several keyword arguments can be used to determine the values of the prior hyperparameters when calling bnpy.run

* ``--nu`` : float
    Sets value of :math:`\bar{\nu}`.
    Defaults to D + 2.

    
* ``--kappa`` : float
    Sets value of :math:`\bar{\kappa}`.
    Defaults to ???.

* ``--ECovMat`` : str
    Determines the expected value of data covariance under the prior.
    Possible values include 'eye' and 'diagcovdata'.
    TODO

* ``--sF`` : float
   These two options set the value of :math:`\bar{\beta}`. TODO.

* TODO set m??

Approximate posterior
=====================

We assume the following factorized approximate posterior family for variational optimization:

.. math ::
    q(z, \mu, \lambda) = \prod_{n=1}^N q(z_n) \cdot \prod_{k=1}^K (\mu_k, \lambda_k )

The specific forms of the global and local factors are given below.

Posterior for local assignments
-------------------------------

For each observation vector at index *n*, we assume an independent approximate posterior over the assigned cluster indicator :math:`z_n \in \{1, 2, \ldots K \}`.

.. math ::
    q( z ) &= \prod_{n=1}^N q(z_n | \hat{r}_n )
    \\
        &= \prod_{n=1}^N \mbox{Discrete}(
            z_n | \hat{r}_{n1}, \hat{r}_{n2}, \ldots \hat{r}_{nK})

Thus, for this observation model the only local variational parameter is the assignment responsibility array :math:`\hat{r} = \{ \{ \hat{r}_{nk} \}_{k=1}^K \}_{n=1}^N`. 

Inside the `LP` dict, this is represented by the `resp` numpy array:

* ``resp`` : 2D array, size N x K
    Parameters of approximate posterior q(z) over cluster assignments.
    resp[n,k] = probability observation n is assigned to component k.

Remember, all computations required by our observation model assume that the ``resp`` array is given. The actual values of ``resp`` are updated by an allocation model.

Posterior for global parameters
-------------------------------

The goal of variational optimization is to find the best approximate posterior distribution for the mean and precision parameters of each cluster *k*:

.. math::
    q( \mu, \lambda ) &= \prod_{k=1}^K \prod_{d=1}^D q( \mu_{kd}, \lambda_{kd} )
    \\
    &= \prod_{k=1}^K \prod_{d=1}^D
        \mathcal{W}_1( \lambda_{kd} | \hat{\nu}_k, \hat{\beta}_{kd} )
        \mathcal{N}_1( \mu_{kd} | \hat{m}_{kd}, \hat{\kappa}_k^{-1} \lambda_{kd}^{-1} )

This approximate posterior is represented by the `Post` attribute of the `DiagGaussObsModel`. This is a ParamBag with the following attributes:

* ``K`` : int
    number of active clusters
* ``nu`` : 1D array, size K
    Defines :math:`\hat{\nu}_k` for each cluster
* ``beta`` : 2D array, size K x D
    Defines :math:`\hat{\beta}_{kd}` for each cluster and dimension
* ``m`` : 2D array, size K x D
    Defines :math:`\hat{m}_{kd}` for each cluster and dimension
* ``kappa`` : 2D array, size K
    Defines :math:`\hat{\kappa}_{k}` for each cluster


Objective function
------------------

Variational optimization will find the approximate posterior parameters that maximize the following objective function, given a fixed observed dataset :math:`x = \{x_1, \ldots x_N \}` and fixed prior hyparparameters :math:`\bar{\nu}, \bar{\beta}, \bar{m}, \bar{\kappa}`.

.. math::
    \mathcal{L}^{\smalltext{DiagGauss}}(
        \hat{\nu}, \hat{\beta}, \hat{m}, \hat{\kappa} )
    &= \sum_{k=1}^K \sum_{d=1}^D
            c^{\smalltext{NW}}_{1,1}(
                \hat{\nu}_k, \hat{\beta}_{kd}, \hat{m}_{kd}, \hat{\kappa})_k
            - c^{\smalltext{NW}}_{1,1}(
                \bar{\nu}, \bar{\beta}_d, \bar{m}_d, \bar{\kappa})
    \\ & \quad + \frac{1}{2} \sum_{k=1}^K \sum_{d=1}^D
        \left(
            N_k(\hat{r}) +  \bar{\nu} - \hat{\nu}_k
        \right)
        \E_q[ \log \lambda_{kd} ]
    \\ & \quad - \frac{1}{2} \sum_{k=1}^K \sum_{d=1}^D
        \left(
            N_{k}(\hat{r}) +  \bar{\kappa} - \hat{\kappa}_{k}
        \right)
        \E_q[ \lambda_{kd} ]
    \\ & \quad + \sum_{k=1}^K \sum_{d=1}^D 
        \left(
            S_{kd}^{x}(x, \hat{r})
            + \bar{\kappa} \bar{m}_d
            - \hat{\kappa}_k \hat{m}_{kd}
        \right)
        \E_q[ \lambda_{kd} \mu_{kd} ]
    \\ & \quad - \frac{1}{2} \sum_{k=1}^K \sum_{d=1}^D 
        \left(
            S_{kd}^{x^2}(x, \hat{r})
            + \bar{\beta}_d + \bar{\kappa} \bar{m}_{d}^2 
            - \hat{\beta}_{kd} - \hat{\kappa}_{k} \hat{m}_{kd}^2
        \right)
        \E_q[ \lambda_{kd} \mu_{kd}^2 ]

This objective function is computed by calling the Python function ``calc_evidence``.

Sufficient statistics
---------------------

The sufficient statistics of this observation model are functions of the local parameters :math:`\hat{r}` and the observed data :math:`x`.

.. math::
    N_{k}(\hat{r}) &= \sum_{n=1}^N \hat{r}_{nk}
    \\
    S^{x}_{kd}(x, \hat{r}) &= \sum_{n=1}^N \hat{r}_{nk} x_{nd}^2
    \\
    S^{x^2}_{kd}(x, \hat{r}) &= \sum_{n=1}^N \hat{r}_{nk} x_{nd}^2

These fields are stored within the sufficient statistics parameter bag ``SS`` as the following fields:

* ``SS.N`` : 1D array, size K
    SS.N[k] = :math:`N_k`
* ``SS.x`` : 2D array, size K x D
    SS.x[k,d] = :math:`S^{x}_{kd}(x, \hat{r})`
* ``SS.xx`` : 2D array, size K x D
    SS.xx[k,d] = :math:`S^{x^2}_{kd}(x, \hat{r})`


Cumulant function
-----------------

The cumulant function of the univariate Normal-Wishart is evaluated for each dimension *d* separately. The function takes 4 scalar input arguments and produces a scalar output.

.. math::
    c^{\smalltext{NW}}_{1,1}(\nu, \beta_d, m_d, \kappa) 
        &=  
        - \frac{1}{2} \log 2\pi
        + \frac{1}{2} \log \kappa
        + \frac{\nu}{2} \log \frac{\beta_d}{2}
        - \log \Gamma \left( \frac{\nu}{2} \right)


Coordinate Ascent Updates
=========================

Local step update
-----------------

As with all observation models, the local step computes the *expected* log conditional probability of assigning each observation to each cluster:

.. math ::
    \E[ \log p( x_n | \mu_k, \lambda_k ) ] =
        - \frac{D}{2} \log 2 \pi
        + \frac{1}{2} \sum_{d=1}^D \E[ \log \lambda_{kd} ]
        - \frac{1}{2} \sum_{d=1}^D \E[ \lambda_{kd} (x_{nd} - \mu_{kd})^2 ]

where the elementary expectations required are:

.. math ::
    \E[ \log \lambda_{kd} ] &=
        \psi \left( \frac{\hat{\nu}_k}{2} \right)
        - \log \frac{\hat{\beta}_{kd}}{2}
    \\
    \E_q \left[  \lambda_{kd} (x_{nd} - \mu_{kd})^2 \right] &= 
        \frac{1}{\hat{\kappa}_{k}} 
        + \frac{ \hat{\nu}_k }{ \hat{\beta}_{kd} } (x_{nd} - \hat{m}_{kd})^2

In our implementation, this is done via the function ``calc_local_params``, which computes the following arrays and places them inside the local parameter dict ``LP``.

* ``E_log_soft_ev`` : 2D array, N x K
    log probability of assigning each observation n to each cluster k
    
Global step update
------------------

The global step update produces an updated approximate posterior over the global random variables. Concretely, this means updated values for each field of the ``Post`` ParamBag attribute of the DiagGaussObsModel.

.. math ::
    \hat{\nu}_k &\gets N_k(\hat{r}) + \bar{\nu}
    \\
    \hat{\kappa}_k &\gets N_k(\hat{r}) + \bar{\kappa}
    \\
    \hat{m}_{kd} &\gets 
        \frac{1}{\hat{\kappa}_k}
        \left( S_k^{x}(x, \hat{r}) + \bar{\kappa} \bar{m}_d \right)
    \\
    \hat{\beta}_{kd} &\gets 
        S_{kd}^{x^2}(x, \hat{r})
        + \bar{\beta}_d
        + \bar{\kappa} \bar{m}_d^2
        - \hat{\kappa}_k \hat{m}_{kd}^2

Our implementation performs this update when calling the function ``update_global_params``.

Initialization
==============

Initialization creates valid values of the parameters which define the approximate posterior over the global random variables. Concretely, this means it creates a valid setting of the ``Post`` attribute of the DiagGaussObsModel object.

TODO


