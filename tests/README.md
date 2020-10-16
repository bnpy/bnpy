# Tests for bnpy

We support several ways to verify code correctness:

### 1) **doctests** for basic functionality of each unit of code

```
$ cd $BNPYREPO/tests/
$ python run_doctests.py
```

### 2) **nosetests** for custom unit tests

```
$ cd $BNPYREPO/tests/
$ python run_nosetests.py
```

### 3) Example Gallery tests for overall end-to-end functionality

Verify that the example gallery used for documentation runs without errors.

```
$ cd $BNPYREPO/tests/
$ python run_all_example_gallery_tests.py
```

TODO verify it matches what the existing code would produce.
