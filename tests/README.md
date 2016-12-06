# Tests for bnpy

We support three primary ways to verify code correctness:

* **doctests** for basic functionality of code

```
$ cd $BNPYREPO/tests/
$ python run_doctests.py
```

* Notebook reproducibility tests

Verify that the notebooks used for documentation have output that matches what the existing code would produce.

```
$ cd $BNPYREPO/tests/
$ python verify_notebook_docs.py
```

* **nosetests** for custom unit tests

```
$ cd $BNPYREPO/tests/
$ python run_nosetests.py
```
