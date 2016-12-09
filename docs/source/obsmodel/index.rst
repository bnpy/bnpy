==================
Observation Models
==================

All observation models define a *likelihood* for producing data :math:`x_n` from some cluster-specific density with parameter :math:`\phi_k`: 

.. math ::

    p(x | \phi, z) = \prod_{n=1}^N p( x_n | \phi_k )^{\delta_k(z_{n})}

Supported Bayesian methods require specifying a (conjugate) prior:

.. math ::

    p(\phi) =  \prod_{k=1}^K p(\phi_k)


Variational methods for observation models
------------------------------------------

The links below describe the mathematical and computational details for performing standard variational optimization for supported observation models:

.. toctree::
    :maxdepth: 1

    ZeroMeanGaussObsModel-VB
    DiagGaussObsModel-VB
    GaussObsModel-VB    
    GaussRegressYFromFixedXObsModel-VB
