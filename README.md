## **bnpy** : Bayesian nonparametric machine learning for python.

[Project Website](https://bnpy.readthedocs.io/en/latest/)
&#8226;
[Example Gallery](https://bnpy.readthedocs.io/en/latest/examples/)
&#8226;
[Installation](https://bnpy.readthedocs.io/en/latest/installation.html)
&#8226;
[Team](#team)
&#8226;
[Academic Papers](#academic-papers)
&#8226;
[Report an Issue](https://github.com/bnpy/bnpy/issues)

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

* Expectation-maximization
    * `EM`
* Full-dataset variational Bayes
    * `VB`
* Memoized variational (can process small minibatches per update step)
    * `moVB`
* Stochastic variational (can process small minibatches per update step)
    * `soVB`

These are all variants of *variational inference*, a family of optimization algorithms. 


# Example Gallery

You can find many examples of **bnpy** in action in our curated [Example Gallery](https://bnpy.readthedocs.io/en/latest/examples/).

These same demos are also directly available as Python scrips inside the [examples/ folder of the project Github repository](https://github.com/bnpy/bnpy/tree/master/examples).

# Quick Start

You can use **bnpy** from a command line/terminal, or from within Python. Both options require specifying a dataset, an allocation model, an observation model (likelihood), and an algorithm. Optional keyword arguments with reasonable defaults allow control of specific model hyperparameters, algorithm parameters, etc.

Below, we show how to call bnpy to train a 8 component Gaussian mixture model on a default toy dataset stored in a .csv file on disk. In both cases, log information is printed to stdout, and all learned model parameters are saved to disk.

## Calling from the terminal/command-line

```
$ python -m bnpy.Run /path/to/my_dataset.csv FiniteMixtureModel Gauss EM --K 8 --output_path /tmp/my_dataset/results/
```

## Calling directly from Python

```
import bnpy
bnpy.run('/path/to/dataset.csv',
         'FiniteMixtureModel', 'Gauss', 'EM',
         K=8, output_path='/tmp/my_dataset/results/')

```

## Advanced examples

Train Dirichlet-process Gaussian mixture model (DP-GMM) via full-dataset variational algorithm (aka "VB" for variational Bayes).

```
python -m bnpy.Run /path/to/dataset.csv DPMixtureModel Gauss VB --K 8
```

Train DP-GMM via memoized variational, with birth and merge moves, with data divided into 10 batches.

```
python -m bnpy.Run /path/to/dataset.csv DPMixtureModel Gauss memoVB --K 8 --nBatch 10 --moves birth,merge
```

## Quick help
```
# print help message for required arguments
python -m bnpy.Run --help 

# print help message for specific keyword options for Gaussian mixture models
python -m bnpy.Run /path/to/dataset.csv FiniteMixtureModel Gauss EM --kwhelp
```

# Installation

To use **bnpy** for the first time, follow the documentation's [Installation Instructions](https://bnpy.readthedocs.io/en/latest/installation.html).

# Team

### Primary investigators

Mike Hughes  
Assistant Professor (Aug. 2018 - present)  
Tufts University, Dept. of Computer Science  
Website: <https://www.michaelchughes.com>  

Erik Sudderth
Professor  
University of California, Irvine  
Website: <https://www.ics.uci.edu/~sudderth/>  

### Contributors

* Soumya Ghosh
* Dae Il Kim
* Geng Ji
* William Stephenson
* Sonia Phene
* Gabe Hope
* Leah Weiner
* Alexis Cook
* Mert Terzihan
* Mengrui Ni
* Jincheng Li
* Xi Chen (Tufts)

# Academic Papers

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


# Target Audience

Primarly, we intend **bnpy** to be a platform for researchers. 
By gathering many learning algorithms and popular models in one convenient, modular repository, we hope to make it easier to compare and contrast approaches. We also hope that the modular organization of **bnpy** enables researchers to try out new modeling ideas without reinventing the wheel.
