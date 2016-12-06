This a Python implementation of the paper 
"A Practical Algorithm for Topic Modeling with Provable Guarantees"

Sanjeev Arora (arora@cs.princeton.edu)
Rong Ge (rongge@cs.princeton.edu)
Yoni Halpern (halpern@cs.nyu.edu)
David Mimno (mimno@cs.princeton.edu)
Ankur Moitra (moitra@ias.edu)
David Sontag (dsontag@cs.nyu.edu)
Yichen Wu (yichenwu@princeton.edu)
Michael Zhu (mhzhu@princeton.edu)

ICML 2013.

####################################################

anchor-word-recover (AWR) is free software; you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by
the Free Software Foundation

AWR is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with AWR; see the file gpl-3.0.txt.  If not, write to the Free
Software Foundation, 59 Temple Place - Suite 330, Boston, MA
02111-1307, USA.

####################################################

This is a work in progress. Any questions/comments/suggestions can be directed to:
Yoni Halpern: halpern@cs.nyu.edu
 
----------------------------------------------------
Running Environment:
    python 2.x
    numpy (for mathematical calculations)
    scipy (for sparse matrix manipulation)

a tutorial for installing python and the rest on macOS is here:
http://www.lowindata.com/2013/installing-scientific-python-on-mac-os-x/

----------------------------------------------------
DEMO:

a demo which downloads an example file from uci machine learning database and learns different models can be found in demo.sh
Some notes about the demo:
    20 topics is likely too few, so it is possible that the topics will be mixed together
    50+ topics should be good.
    Using KL as a loss is slower than L2 loss.

----------------------------------------------------

Simple Usage:

    input:  1-indexed word-document matrix in the UCI format 
            vocabulary file
            (see http://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/ for examples of matrix and vocab files.) 
            list of stopwords named stopwords.txt
            
    1. transform document to sparse matrix format:
        python uci_to_scipy.py input_file output_file
    
    2. remove rare words and stopwords (script removes words that do not appear in at least N documents).
        python truncate_vocabulary.py sparse_matrix_file vocab_file N
        note: this implementation is fairly sensitive to rare words. In the demo we filter words that appear in fewer than 50 documents and only allow anchors if they appear in over 100 documents.

    3. learn K topics
        python learn_topics.py sparse_matrix_file settings_file vocab_file K loss output_prefix
    
        an example settings file can be found in settings.example
        loss can be L2 or KL
    
    outputs:
        Recovered A matrix
        Top words
        topic likelihoods 
        

