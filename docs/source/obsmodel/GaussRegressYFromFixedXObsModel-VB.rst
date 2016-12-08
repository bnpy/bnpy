===================================================================
Gaussian Regression observation model: Variational Bayesian Methods
===================================================================

Generative model
================

The Gaussian regression observation model explains a observed collection of input/ouptut data pairs :math:`\{x_n, y_n\}_{n=1}^N`. Each input observation :math:`x_n` is a vector of length *D*, while each output observation :math:`y_n` is a scalar.

In this document, we assume that the observed input data :math:`x` are fixed and focus on a generative model for the output data :math:`y` which depends on :math:`x`. Various generative models, such as the diagonal-covariance Gaussian, are possible for the observed data :math:`x`. These can be straight-forwardly combined with the model here to produce a joint model for both :math:`x` and :math:`y`.

Each cluster *k* produces the output values according to the following standard Bayesian linear regression model:

.. math:: 
    y_{n} \sim \mathcal{N} \left(
        b_k + \sum_{d=1}^D w_{kd} x_{nd},
        \delta_{k}^{-1}         
        \right)

Here, the cluster-specific parameters are a weight vector :math:`w_k`, an intercept weight :math:`b_k`, and a precision scalar :math:`\delta_k`.  These are the global random variables of this observation model.

Alternatively, if we define an *expanded* input data vector :math:`\tilde{x}_n = [x_{n1} x_{n2} \ldots x_{nD} ~ 1]`, we can write the generative model more simply as:

.. math::
    y_{n} \sim \mathcal{N} \left(
        \sum_{d=1}^{D+1} w_{kd} \tilde{x}_{nd},
        \delta_{k}^{-1}         
        \right)

Global Random Variables
-----------------------

The global random variables are the cluster-specific weights and precisions. For each cluster *k*, we have 

.. math::
    w_{k} &\in \mathbb{R}^D, \qquad w_k = [w_{k1}, w_{k2}, \ldots w_{kD} w_{kD+1} ]
    \\
    \delta_k &\in (0, +\infty)

For convenience, let :math:`E` denote the size of this expanded representation, where :math:`E = D+1`. 

Local Random Variables
----------------------

Each dataset observation at index *n* has its own cluster assignment:

.. math::
    z_n \in \{1, 2, \ldots K \}

The generative model and approximate posterior for :math:`z_n` is determined by an allocation model. For all computations needed by our current observation model, we'll assume either a point estimate or an approximate posterior for :math:`z_n` is known.

Normal-Wishart prior
====================

We assume that the weights :math:`w_k` and the precision :math:`\delta_k` have a joint Normal-Wishart prior with hyperparameters:

* :math:`\bar{\nu}` : positive scalar
    count parameter of the Wishart prior on precision
* :math:`\bar{\tau}` : positive scalar
    location parameter of the Wishart prior on precision
* :math:`\bar{w}` : vector of size E
    mean value of the Normal prior on cluster weights
* :math:`\bar{P}` : positive definite E x E matrix
    precision matrix for the Normal prior on cluster weights

Mathematically, we have:

.. math ::
    \delta_{k} &\sim \mathcal{W}_1(\bar{\nu}, \bar{\tau})
    \\
    w_{k} &\sim \mathcal{N}_E( \bar{w}, \delta_k^{-1} \bar{P}^{-1} )

Under this prior, here are some useful expectations for the precision random variable:

.. math::
    \E_{\mbox{prior}}[ \delta_k ] &= \frac{\bar{\nu}}{\bar{\tau}}
    \\
    \E_{\mbox{prior}}[ \delta_k^{-1} ] &= \frac{\bar{\tau}}{\bar{\nu} - 2}
    \\
    \mbox{Var}_{\mbox{prior}}[ \delta_k ] &= 
        \frac{\bar{\nu}}{\bar{\tau}^2}

Likewise, here are some useful prior expectations for the weight vector random variable:

.. math::
    \E_{\mbox{prior}}[w_k] &= \bar{w}
    \\
    \mbox{Cov}_{\mbox{prior}}[w_k] &=
        \frac{\bar{\tau}}{\bar{\nu} - 2} 
        \bar{P}^{-1}

And some useful joint expectations:

.. math::
    \E_{\mbox{prior}}[\delta_k w_k ] &=  
        \frac{\bar{\nu}}{\bar{\tau}}\bar{w}
    \\
    \E_{\mbox{prior}}[\delta_k w_k w_k^T ] &=
        \bar{P}^{-1} + 
        \frac{\bar{\nu}}{\bar{\tau}} \bar{w} \bar{w}^{T}


In our Python implementation of the ``GaussRegressYFromFixedXObsModel`` class, these quantities are represented by the following numpy array attributes of the ``Prior`` parameter bag:

* ``pnu`` : float
    value of :math:`\bar{\nu}`
* ``ptau`` : float
    value of :math:`\bar{\tau}`
* ``w_E`` : 1D array, size E
    value of :math:`\bar{w}`
* ``P_EE`` : 2D array, size E x E
    value of :math:`\bar{P}`

Several keyword arguments can be used to determine the values of the prior hyperparameters when calling bnpy.run

* ``--pnu`` : float
    Sets value of :math:`\bar{\nu}`.
    Defaults to 1.
* ``--ptau`` : float
    Sets value of :math:`\bar{\tau}`.
    Defaults to 1.
* ``--w_E`` : float or 1D array
    Sets value of the vector :math:`\bar{w}`.
    If float is provided, the whole vector is filled with that value.
    Defaults to 0.
* ``--P_diag_val`` : float or 1D array
    Sets :math:`\bar{P}` to diagonal matrix with specified values.
    Defaults to 1e-6.


Approximate posterior
=====================

We assume the following factorized approximate posterior family for variational optimization:

.. math ::
    q(z, w, \delta) =
        \prod_{n=1}^N q(z_n)
        \cdot \prod_{k=1}^K (w_k, \delta_k )

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
    q( w, \delta ) &= \prod_{k=1}^K 
        \mathcal{W}_1( \delta_{k} | \hat{\nu}_k, \hat{\tau}_{k} )
        \mathcal{N}_E( w_{k} | \hat{w}_{k},
            \delta_k^{-1} \hat{P}_k^{-1}
            )

Within our Python implementation in the class ``GaussRegressYFromFixedXObsModel``, this approximate posterior is represented within the `Post` attribute. This attribute is a ParamBag object containing the following numpy arrays:

* ``K`` : int
    number of active clusters
* ``pnu_K`` : 1D array, size K
    Defines :math:`\hat{\nu}_k` for each cluster
* ``ptau_K`` : 1D array, size K
    Defines :math:`\hat{\tau}_{k}` for each cluster
* ``w_KE`` : 2D array, size K x E
    Defines :math:`\hat{w}_{ke}` for each cluster and expanded dimension
* ``P_KEE`` : 2D array, size K x E x E
    Defines precision matrix :math:`\hat{P}_{k}` for each cluster


Objective function
------------------

Variational optimization will find the approximate posterior parameters that maximize the following objective function, given a fixed observed dataset :math:`x = \{x_1, \ldots x_N \}` and fixed prior hyparparameters :math:`\bar{\nu}, \bar{\tau}, \bar{w}, \bar{P}`.

.. math::
    \mathcal{L}^{\smalltext{Gaussian Regression}}(y, x, 
        \hat{\nu}, \hat{\tau}, \hat{w}, \hat{P} )
    &= -\frac{N}{2} \log 2\pi
    \\ & \quad + \sum_{k=1}^K \sum_{d=1}^D
        c^{\smalltext{NW}}_{1,E}(
            \hat{\nu}_k, \hat{\tau}_{k}, \hat{w}_{k}, \hat{P}_k)
        - c^{\smalltext{NW}}_{1,E}(
            \bar{\nu}, \bar{\tau}, \bar{w}, \bar{P})
      \\
      & \quad -\frac{1}{2} \sum_{k=1}^K
        \left(
            N_k(\hat{r}) +  \bar{\nu} - \hat{\nu}_k
        \right)
        \E_q[ \log \delta_k ]
      \\
      & \quad -\frac{1}{2} \sum_{k=1}^K 
        \left(
            S_{k}^{yy}(y, \hat{r})
            + \bar{\tau} + \bar{w}\bar{P}\bar{w}
            - \hat{\tau}_k - \hat{w}_k \hat{P}_k \hat{w}_k
        \right)
        \E_q[ \delta_k ]
      \\
      & \quad + \sum_{k=1}^K
        \left(
            S_k^{yx}(x, y, \hat{r}) 
            + \bar{P} \bar{w}
            - \hat{P}_k \hat{w}_k
        \right)^T
        \E_q[ \delta_k w_k ]
      \\
      & \quad - \frac{1}{2} \sum_{k=1}^K 
        \mbox{trace}
        \left(
            \left(
                S_k^{xx^T}(x, \hat{r})
                + \bar{P}
                - \hat{P}_k
            \right)
            \E_q[ \delta_k w_k w_k^T]
        \right)

This objective function is computed by calling the Python function ``calc_evidence``.

We can directly interpret this function as a lower bound on the marginal evidence:

.. math ::
    \log p(y | x, \bar{\nu}, \bar{\tau}, \bar{w}, \bar{P})
    \geq 
    \mathcal{L}^{\smalltext{Gaussian Regression}}
        (y, x, \hat{\nu}, \hat{\tau}, \hat{w}, \hat{P} )


Sufficient statistics
---------------------

The sufficient statistics of this observation model are functions of the local parameters :math:`\hat{r}`, the observed input data :math:`x`, and the observed output data :math:`y`. 

.. math::
    N_{k}(\hat{r}) &= \sum_{n=1}^N \hat{r}_{nk}
    \\
    S^{y^2}_{k}(y, \hat{r}) &= \sum_{n=1}^N \hat{r}_{nk} y_n^2
    \\
    S^{yx}_{k}(x, y, \hat{r}) &= \sum_{n=1}^N \hat{r}_{nk} y_n x_{n}
    \\
    S^{xx^T}_{k}(x, \hat{r}) &= \sum_{n=1}^N \hat{r}_{nk} x_{n} x_{n}^T
    

These fields are stored within the sufficient statistics parameter bag ``SS`` as the following fields:

* ``SS.N`` : 1D array, size K
    SS.N[k] = :math:`N_k`
* ``SS.yy_K`` : 1D array, size K
    SS.yy[k] = :math:`S^{y^2}_{k}(y, \hat{r})`
* ``SS.yx`` : 2D array, size K x E
    SS.yx[k] = :math:`S^{yx}_{k}(x, y, \hat{r})`
* ``SS.xxT`` : 3D array, size K x E x E
    SS.xxT[k] = :math:`S^{xx^T}_{k}(x, \hat{r})`


Cumulant function
-----------------

The cumulant function of the Normal-Wishart produces a scalar output from 4 input arguments:

.. math::
    c^{\smalltext{NW}}_{1,E}(\nu, \tau, w, P) 
        &=  
        \frac{E}{2} \log 2 \pi
        - \frac{1}{2} \log |P|
        - \frac{\nu}{2} \log \frac{\tau}{2}
        + \log \Gamma \left( \frac{\nu}{2} \right)

where :math:`\Gamma(\cdot)` is the gamma function, and :math:`\log |P|` is the log determinant of the E x E matrix :math:`P`.

Coordinate Ascent Updates
=========================

Local step update
-----------------

As with all observation models, the local step computes the *expected* log conditional probability of assigning each observation to each cluster:

.. math ::
    \E_q[ \log p( y_n | x_n, w_k, \delta_k) ] =
        - \frac{1}{2} \log 2 \pi
        + \frac{1}{2} \E[ \log \delta_{k} ]
        - \frac{1}{2} \E[ \delta_{k} (y_{n} - w_k ^T \tilde{x}_n)^2 ]

where the elementary expectations required are:

.. math ::
    \E_q [ \log \delta_{k} ] &=
        - \log \frac{\hat{\tau}_k}{2}
        + \psi \left( \frac{\hat{\nu}_k}{2} \right)
    \\
    \E_q \left[  \delta_{k} \left( y_{n} - w_k^T \tilde{x}_n \right)^2 \right] &= 
        \tilde{x}_n^T \hat{P}_k^{-1} \tilde{x}_n
        + \frac{ \hat{\nu}_k }{ \hat{\tau}_{k} }
            (y_{n} - \bar{w}_{k}^T \tilde{x}_n)^2

The above operations can be efficiently computed via smart vectorized calculations on modern cpus.

In our implementation, this is done via the function ``calc_local_params``, which computes the following arrays and places them inside the local parameter dict ``LP``.

* ``E_log_soft_ev`` : 2D array, N x K
    log probability of assigning each observation n to each cluster k


Global step update
------------------

The global step update produces an updated approximate posterior over the global random variables. Concretely, this means updated values for each of the four parameters which define each cluster-specific Normal-Wishart:

.. math ::
    \hat{\nu}_k &\gets N_k(\hat{r}) + \bar{\nu}
    \\
    \hat{P}_{k} &\gets 
        \bar{P}_k + S^{xx^T}_k(x, \hat{r})
    \\
    \hat{w}_{k} &\gets 
        \hat{P}_k^{-1}
            \left( 
                \bar{P} \bar{w} + S^{yx}_k(x, y, \hat{r})
            \right)
    \\
    \hat{\tau}_k &\gets 
        \bar{\tau} + S^{y^2}_k(y, \hat{r})
        + \bar{w}^T \bar{P} \bar{w}
        - \hat{w}_k^T \hat{P}_k \hat{w}_k

Our implementation performs this update when calling the function ``update_global_params``.

Initialization
==============

Initialization creates valid values of the parameters which define the approximate posterior over the global random variables. Concretely, this means it creates a valid setting of the ``Post`` attribute of the ``GaussRegressYFromFixedXObsModel`` object.

TODO


