=========================================================================
Supervised Latent Dirichlet allocation model: Variational Methods
=========================================================================

Generative model
================

The model generates documents as follows: 


Draw topic proportions :math:`\theta \sim Dir(\alpha)`

For each word:
  a) Draw topic assignment :math:`z_n \sim Mult(\theta)`
  b) Draw word :math:`v_n \sim Mult(\beta_{z_n})`

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
     r_{dn} \sim Mult(\pi_d)

   

Global Random Variables
------------------------

The approximate posterior for :math:`\beta` is determined by an observation model. The approximate posteriors for :math:`\eta` and :math:`\delta` are determined in the \texttt{update\_global\_params} function in the allocation model.  For all local computations needed by our current allocation model, we'll assume a point estimate or an approximate posterior for :math:`\beta`. We will update :math:`\eta` according to the MAP estimate (see below) and update :math:`\delta` or keep :math:`\delta` fixed, depending on the user specification. Alternately, we put a full Normal-Wishart prior on :math:`\eta` and :math:`\delta`.


Approximate posterior
=====================

Posterior for local assignments
-------------------------------

For each observation vector at index *n* (indexing word types, not word tokens), we assume an independent approximate posterior over the assigned cluster indicator :math:`z_n \in \{1, 2, \ldots K \}`. 

.. math ::
    q(z | r) = \prod_{d=1}^D \prod_{n=1}^{N_d} q(z_{dn} | r_{dn})

.. math ::
    q(\pi | \theta) = \prod_{d=1}^D q(\pi_d | \theta_d ) 



Posterior for global parameters
-------------------------------



Objective function
------------------

Variational optimization will find the approximate posterior parameters that maximize the following objective function, given a fixed observed dataset :math:`x = \{x_1, \ldots x_N \}` and learned prior parameters :math:`\beta, \eta, \delta, \alpha`. We let :math:`rw_{dn} = r_{dn} w_n`, (a weighted responsibility since we are doing per word type (not per token) updates) and :math:`rw_{-t} = \sum_{n \neq t} w_n r_{dn}`.

.. math::
    \mathcal{L}^{{supervised}} = 
        \sum_{d=1}^D \log{\frac{1}{\sqrt{2\pi\delta}}}
        - \frac{y_d^2}{2\delta}
        + \frac{y_d}{2\delta}\eta^T (\sum_{n=1}^{N_d}  rw_{dn})
        - \frac{1}{2\delta} \mathbb{E}[\eta^T \bar{z_d}^T \bar{z_d}\eta]

where

.. math::
    \mathbb{E}[\eta^T \bar{z_d}^T \bar{z_d}\eta]_d = 
    \frac{1}{N_d^2}[(\eta (rw_{-t}\eta^T rw_t + (\eta \circ \eta)^T rw_t]

.. math::
    \mathcal{L}^{{unsupervised}} = 
    \log \Gamma (\sum_{k=1}^K \alpha_k) - \sum_{k=1}^K \alpha_k + \sum_{k=1}^K (\alpha_k - 1)[\Psi(\pi_{dk}) - \Psi(\sum_{j=1}^K \pi_{dj})] \newline
    + \sum_{n=1}^{N_d} \sum_{k=1}^K rw_{dnk}[\Psi(\pi_{dk}) - \Psi(\sum_{j=1}^K \pi_{dj})] \newline
    - \sum_{n=1}^{N_d} \sum_{k=1}^K rw_{dnk} \log r_{dnk} \newline
    - \log \Gamma(\sum_{k=1}^K \theta_k) - \sum_{k=1}^K \log \Gamma(\theta_k) + \sum_{k=1}^K (\theta_k - 1) [\Psi(\theta_{dk}  - \Psi(\sum_{j=1}^K \theta_{dk})]


This objective function 

.. math::
    \mathcal{L}^{unsupervised} + \mathcal{L}^{{supervised}}

is computed by calling the Python function ``calc_evidence``.


Update equations
----------------

.. math::
    r_{dn} \propto \exp\{\mathbb{E}[\log \beta_v] + \Psi(\theta_{d}) - \Psi(\sum_{j=1}^K \theta_{dj}} - \frac{y_d}{\delta N_d}\eta - \frac{1}{\delta N_d^2}(2\eta^T rw_{-j} \eta + \eta \circ \eta))\}

where :math:`k` is the cluster assignment of :math:`r_{dn}` and :math:`v` is the vocab index for word :math:`n` in document :math:`d` and :math:`rw_{-j} = \sum_{n \neq j} w_n r_{dn}`. The variable :math:`w_n` is the weight (number of occurrences of token :math:`n`) in doc d.


.. math::
    \theta_d \propto \alpha + \sum_{n=1}^{N_d} rw_{dn} 



.. math::
    \eta \propto \mathbb{E}[Z^T Z]^{-1} \mathbb{E}[Z]^T y 


.. math::
    \delta \propto \frac{1}{D} [y^Ty - y^T \mathbb{E}[Z] (\mathbb{E}[Z^T Z])]^{-1}\mathbb{E}[Z]^Ty]


where :math:`\mathbb{E}[Z] = \sum_{n=1}^{N_d} rw_{dn}`. The parameters :math:`\eta` and :math:`\delta` (when applicable) are updated in the :math:`\texttt{update\_global\_params}` function.

General
-------
https://arxiv.org/pdf/1003.0783.pdf
Supervised lda with response y drawn from a Normal distribution

Parameters:

alpha: float 1.0 (default)
parameter for Dirichlet distribution

delta: float 0.1 (default)
variance of response (:math:`y \sim N(\bar{z} \eta, \delta)`)

update_delta: boolean 

0 (default) to fix delta to fixed value
1 to update variance with MAP estimate

Example code:

.. math::
    \texttt{import bnpy} \newline
     \texttt{import grid3x3\_nD400\_nW100} \newline
     \texttt{Data = grid3x3\_nD400\_nW100.get\_data()} \newline
     \texttt{Data.name = 'grid3x3\_nD400\_nW100'} \newline
     \texttt{h, r = 
 bnpy.run(TrainData,'SupervisedFiniteTopicModel2','Mult','VB',K=9,nLap=50,jobname='test',delta=0.01,update_delta=1)}

