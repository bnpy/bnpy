=========================================================================
Supervised Latent Dirichlet allocation model: Variational Methods
=========================================================================

TODO update this page

Generative model
================

The model generates documents as follows: 


Draw topic proportions :math:`\theta \sim Dir(\alpha)`

For each word:
  a) Draw topic assignment :math:`z_n \sim Mult(\theta)`
  b) Draw word :math:`w_n \sim Mult(\beta_{z_n})`

Draw response :math:`y \sim Normal(\bar{z} \eta,\delta)`

where :math:`\bar{z} = \frac{1}{N_d} \sum_{n=1}^{N_d} z_n`



Variational Parameters
-----------------------

There are variational parameters on the topic proportions and for the cluster assignments for each observation.

For each document *d*, we have:

.. math::
     \pi_d \sim Dir(\alpha)

For each observation, we have:

.. math::
     r_{d,n} \sim Mult(\pi_d)

   

Global Random Variables
------------------------

The approximate posteriors for :math:`\beta`, :math:`\eta` and :math:`\delta` are determined by an observation model. For all computations needed by our current observation model, we'll assume a point estimate or an approximate posterior for :math:`\beta`. We will update :math:`\eta` according to the MAP estimate (see below) and update :math:`\delta` or keep :math:`\delta` fixed, depending on the user specification. Alternately, we put a full Normal-Wishart prior on :math:`\eta` and :math:`\delta`


Approximate posterior
=====================

Posterior for local assignments
-------------------------------

For each observation vector at index *n*, we assume an independent approximate posterior over the assigned cluster indicator :math:`z_n \in \{1, 2, \ldots K \}`.

.. math ::
    q(z | r) = \prod_{d=1}^D \prod_{n=1}^{N_d} q(z_{dn} | r_{dn})

.. math ::
    q(\theta | \pi) = \prod_{d=1}^D q(\theta_d | \pi_d) 



Posterior for global parameters
-------------------------------



Objective function
------------------

Variational optimization will find the approximate posterior parameters that maximize the following objective function, given a fixed observed dataset :math:`x = \{x_1, \ldots x_N \}` and fixed prior hyparparameters :math:`\beta, \eta, \delta`.

.. math::
    \mathcal{L}^{{supervised}} = 
        \sum_{d=1}^D \log{\frac{1}{\sqrt{2\pi\delta}}}
        - \frac{y_d^2}{2\delta}
        + \frac{y_d}{2\delta}\eta^T (\sum_{n=1}^{N_d} w_n r_{dn})
        - \frac{1}{2\delta} \mathbb{E}[\eta^T \bar{z_d}^T \bar{z_d}\eta]

where

.. math::
\mathbb{E}[\eta^T \bar{z_d}^T \bar{z_d}\eta]_d = 

This objective function 

.. math::
    \mathcal{L}^{unsupervised} + \mathcal{L}^{{supervised}}

is computed by calling the Python function ``calc_evidence``.
