## **bnpy** : Bayesian nonparametric machine learning for python.

![bnpy-headline.png](https://bitbucket.org/repo/87qLXb/images/1585298866-bnpy-headline.png)

* [About](#markdown-header-about)
* [Demos](#markdown-header-demos)
* [Quick Start](#markdown-header-quick-start)
* [Academic References](#markdown-header-academic-references)
* * [NIPS 2015: HDP-HMM paper](#markdown-header-nips-2015-hdp-hmm-paper)
* * [AISTATS 2015: HDP topic models](#markdown-header-aistats-2015-hdp-topic-model-paper)
* * [NIPS 2013: DP mixture models](#markdown-header-nips-2013-dp-mixtures-paper)

# About
This python module provides code for training popular clustering models on large datasets. We focus on Bayesian nonparametric models based on the Dirichlet process, but also provide parametric counterparts. 

**bnpy** supports the latest online learning algorithms as well as standard offline methods. Our aim is to provide an inference platform that makes it easy for researchers and practitioners to compare models and algorithms.

### Supported probabilistic models (aka allocation models)

* Mixture models
    * `FiniteMixtureModel` : fixed number of clusters
    * `DPMixtureModel` : infinite number of clusters, via the Dirichlet process

* Topic models (aka admixtures models)
    * `FiniteTopicModel` : fixed number of topics. This is Latent Dirichlet allocation.
    * `HDPTopicModel` : infinite number of topics, via the hierarchical Dirichlet process
    
* Hidden Markov models (HMMs)
    * `FiniteHMM` : Markov sequence model with a fixture number of states
    *  `HDPHMM` : Markov sequence models with an infinite number of states

* **COMING SOON**
    * grammar models
    * relational models

### Supported data observation models (aka likelihoods)

* Multinomial for bag-of-words data
    * `Mult`
* Gaussian for real-valued vector data
    * `Gauss` : Full-covariance 
    * `DiagGauss` : Diagonal-covariance
    * `ZeroMeanGauss` : Zero-mean, full-covariance
* Auto-regressive Gaussian
    * `AutoRegGauss`

### Supported learning algorithms:

* Expectation-maximization (offline)
    * `EM`
* Full-dataset variational Bayes (offline)
    * `VB`
* Memoized variational (online)
    * `moVB`
* Stochastic variational (online)
    * `soVB`

These are all variants of *variational inference*, a family of optimization algorithms. We plan to eventually support sampling methods (Markov chain Monte Carlo) too.

# Demos

You can find many examples of **bnpy** in action in our curated set of  [IPython notebooks](http://nbviewer.ipython.org/urls/bitbucket.org/michaelchughes/bnpy-dev/raw/master/demos/DemoIndex.ipynb).

These same demos are also directly available on our [wiki](http://bitbucket.org/michaelchughes/bnpy-dev/wiki/demos/DemoIndex.rst).

# Quick Start

You can use **bnpy** from the terminal, or from within Python. Both options require specifying a dataset, an allocation model, an observation model (likelihood), and an algorithm. Optional keyword arguments with reasonable defaults allow control of specific model hyperparameters, algorithm parameters, etc.

Below, we show how to call bnpy to train a 8 component Gaussian mixture model on the default AsteriskK8 toy dataset (shown below).
In both cases, log information is printed to stdout, and all learned model parameters are saved to disk.

## Calling from the terminal/command-line

```
$ python -m bnpy.Run AsteriskK8 FiniteMixtureModel Gauss EM --K 8
```

## Calling directly from Python

```
import bnpy
bnpy.run('AsteriskK8', 'FiniteMixtureModel', 'Gauss', 'EM', K=8)
```

## Other examples
Train Dirichlet-process Gaussian mixture model (DP-GMM) via full-dataset variational algorithm (aka "VB" for variational Bayes).

```
python -m bnpy.Run AsteriskK8 DPMixtureModel Gauss VB --K 8
```

Train DP-GMM via memoized variational, with birth and merge moves, with data divided into 10 batches.

```
python -m bnpy.Run AsteriskK8 DPMixtureModel Gauss moVB --K 8 --nBatch 10 --moves birth,merge
```

## Quick help
```
# print help message for required arguments
python -m bnpy.Run --help 

# print help message for specific keyword options for Gaussian mixture models
python -m bnpy.Run AsteriskK8 FiniteMixtureModel Gauss EM --kwhelp
```

# Installation and Configuration

To use **bnpy** for the first time, follow the [installation instructions](http://bitbucket.org/michaelchughes/bnpy-dev/wiki/Installation.md) on our project wiki.

Once installed, please visit the [Configuration](http://bitbucket.org/michaelchughes/bnpy-dev/wiki/Configuration.md) wiki page to learn how to configure where data is saved and loaded from on disk.

All documentation can be found on the  [project wiki](http://bitbucket.org/michaelchughes/bnpy-dev/wiki/Home.md).

# Team

### Primary contact
Mike Hughes  
PhD candidate  
Brown University, Dept. of Computer Science  
Website: [www.michaelchughes.com](http://www.michaelchughes.com)

### Faculty adviser

Erik Sudderth  
Assistant Professor  
Brown University, Dept. of Computer Science  
Website: [http://cs.brown.edu/people/sudderth/](http://cs.brown.edu/people/sudderth/)

### Contributors 

* Soumya Ghosh
* Dae Il Kim
* Geng Ji
* William Stephenson
* Sonia Phene
* Mert Terzihan
* Mengrui Ni
* Jincheng Li

# Academic References

## Conference publications based on BNPy

#### NIPS 2015 HDP-HMM paper

> Our NIPS 2015 paper describes inference algorithms that can add or remove clusters for the sticky HDP-HMM.

* "Scalable adaptation of state complexity for nonparametric hidden Markov models." Michael C. Hughes, William Stephenson, and Erik B. Sudderth. NIPS 2015.
[[paper]](http://michaelchughes.com/papers/HughesStephensonSudderth_NIPS_2015.pdf)
[[supplement]](http://michaelchughes.com/papers/HughesStephensonSudderth_NIPS_2015_supplement.pdf)
[[scripts to reproduce experiments]](http://bitbucket.org/michaelchughes/x-hdphmm-nips2015/)

#### AISTATS 2015 HDP topic model paper

> Our AISTATS 2015 paper describes our algorithms for HDP topic models.

* "Reliable and scalable variational inference for the hierarchical Dirichlet process." Michael C. Hughes, Dae Il Kim, and Erik B. Sudderth. AISTATS 2015.
[[paper]](http://michaelchughes.com/papers/HughesKimSudderth_AISTATS_2015.pdf)
[[supplement]](http://michaelchughes.com/papers/HughesKimSudderth_AISTATS_2015_supplement.pdf)
[[bibtex]](http://cs.brown.edu/people/mhughes/papers/HughesKimSudderth-AISTATS2015-MemoizedHDP-bibtex.txt)

#### NIPS 2013 DP mixtures paper

> Our NIPS 2013 paper introduced memoized variational inference algorithm, and applied it to Dirichlet process mixture models.

* "Memoized online variational inference for Dirichlet process mixture models." Michael C. Hughes and Erik B. Sudderth. NIPS 2013.
[[paper]](http://michaelchughes.com/papers/HughesSudderth_NIPS_2013.pdf)
[[supplement]](http://michaelchughes.com/papers/HughesSudderth_NIPS_2013_supplement.pdf)
[[bibtex]](http://cs.brown.edu/people/mhughes/papers/HughesSudderth-NIPS2013-MemoizedDP-bibtex.txt)

## Workshop papers

> Our short paper from a workshop at NIPS 2014 describes the vision for **bnpy** as a general purpose inference engine.

* "bnpy: Reliable and scalable variational inference for Bayesian nonparametric models."
Michael C. Hughes and Erik B. Sudderth. Probabilistic Programming Workshop at NIPS 2014.
[[paper]](http://michaelchughes.com/papers/HughesSudderth_NIPSProbabilisticProgrammingWorkshop_2014.pdf)

## Background reading
For background reading to understand the broader context of this field, see our [Resources wiki page](../wiki/Resources.md).

# Target Audience

Primarly, we intend **bnpy** to be a platform for researchers. 
By gathering many learning algorithms and popular models in one convenient, modular repository, we hope to make it easier to compare and contrast approaches.
We also how that the modular organization of **bnpy** enables researchers to try out new modeling ideas without reinventing the wheel.
