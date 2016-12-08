Welcome to bnpy
===============

BNPy (or bnpy) is Bayesian Nonparametric clustering for Python.

Our goal is to make it easy for Python programmers to train state-of-the-art clustering models on large datasets. We focus on nonparametric models based on the Dirichlet process, especially extensions that handle hierarchical and sequential datasets. Traditional parametric counterparts (like finite mixture models) are also supported. 

Training a model with **bnpy** requires the user to specify the dataset, the model, and the algorithm to use. Flexible keyword options allow advanced users lots of control, but smart defaults make it simple for beginners. 
**bnpy**'s modular implementation makes it possible to try many variants of models and algorithms, to find the best fit for the data at hand.

Example Gallery
---------------
You can find many examples of **bnpy** in action in our curated `Example Gallery <examples/>`_.

These same demos are also directly available as Python scrips inside the `project Github repository <https://github.com/bnpy/bnpy/tree/master/examples>`_.

Quick Start
-----------

You can use **bnpy** to train a model in two ways: (1) from a command line/terminal, or (2) from within a Python script (of course). Both options require specifying a dataset, an allocation model, an observation model (likelihood), and an algorithm. Optional keyword arguments with reasonable defaults allow control of specific model hyperparameters, algorithm parameters, etc.

Below, we show how to call bnpy to train a 8 component Gaussian mixture model on a default toy dataset stored in a .csv file on disk. In both cases, log information is printed to stdout, and all learned model parameters are saved to disk.

Training from a terminal
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python -m bnpy.Run /path/to/my_dataset.csv FiniteMixtureModel Gauss EM --K 8 --output_path /tmp/my_dataset/results/

Training via Python
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import bnpy
    bnpy.run('/path/to/dataset.csv',
             'FiniteMixtureModel', 'Gauss', 'EM',
             K=8, output_path='/tmp/my_dataset/results/')

Featured algorithms
~~~~~~~~~~~~~~~~~~~

Train a Dirichlet-process Gaussian mixture model (DP-GMM) via full-dataset variational coordinate ascent. This algorithm is often called "VB" for variational Bayes.

.. code-block:: bash

    python -m bnpy.Run /path/to/dataset.csv DPMixtureModel Gauss VB --K 8

Train DP-GMM via scalable incremental or "memoized" variational coordinate ascent, with birth and merge moves, with data divided into 10 batches.

.. code-block:: bash

    python -m bnpy.Run /path/to/dataset.csv DPMixtureModel Gauss memoVB --K 8 --nBatch 10 --moves birth,merge


Train HDP-HMM model to capture sequential structure in the dataset

.. code-block:: bash
    python -m bnpy.Run /path/to/dataset.csv HDPHMM DiagGauss memoVB --K 8


Getting Help
~~~~~~~~~~~~

.. code-block:: bash

    # print help message for required arguments
    python -m bnpy.Run --help 

.. code-block:: bash

    # print help message for specific keyword options for Gaussian mixture models
    python -m bnpy.Run /path/to/dataset.csv FiniteMixtureModel Gauss EM --kwhelp


Supported allocation models
---------------------------

The following are possible *allocation* models, which is **bnpy**-terminology for a generative model which assigns clusters to structured datasets.

* Mixture models
    * `FiniteMixtureModel` : fixed number of clusters
    * `DPMixtureModel` : infinite number of clusters, via the Dirichlet process

* Topic models (aka admixtures models)
    * `FiniteTopicModel` : fixed number of topics. This is Latent Dirichlet allocation.
    * `HDPTopicModel` : infinite number of topics, via the hierarchical Dirichlet process
    
* Hidden Markov models (HMMs)
    * `FiniteHMM` : Markov sequence model with a fixture number of states
    *  `HDPHMM` : Markov sequence models with an infinite number of states

* COMING SOON
    * relational models (like the IRM, MMSB, etc.)
    * grammar models

Supported observations models
-----------------------------

Any of the above allocation models can be combined with one of these *observation* models, which describe how to produce data assigned to a specific cluster.

* Real-valued vector observations (1-dim, 2-dim, ... D-dim)
    * `Gauss` : Full-covariance Gaussian
    * `DiagGauss` : Diagonal-covariance Gaussian
    * `ZeroMeanGauss` : Zero-mean, full-covariance
    * `AutoRegGauss` : first-order auto-regressive Gaussian 
* Binary vector observations (1-dim, 2-dim, ... D-dim)
    * `Bern` : Bernoulli 
* Discrete, bag-of-words data (each observation is one of V symbols)
    * `Mult` : Multinomial


Supported algorithms
--------------------

* Variational methods
    * `EM` : Expectation-maximization
    * `VB` : variational Bayes
    * `soVB` : stochastic variational (online)
    * `moVB` : memoized variational (online)

* COMING SOON
    * Gibbs sampling



.. toctree::
    :maxdepth: 1
    :hidden:

    installation
    examples/index
    allocmodel/index
    obsmodel/index

