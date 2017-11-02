'''
run_doctests.py

Simple executable script to run doctests on every file within bnpy/ package.

Usage
-----
# To run doctests on EVERY .py file within the bnpy package
$ python run_doctests.py

# To run doctests on a specific folder (relative to bnpy/ root dir)
$ python run_doctests.py --path suffstats/
'''
import six
import os
import argparse
import doctest
import subprocess

def test_all_py_files_in_dir(dir_path):
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in files:
            full_name = os.path.join(root, name)
            if full_name.count('zzz') > 0:
                continue
            if full_name.endswith('.py'):
                run_doctest_for_py_file(full_name)

def run_doctest_for_py_file(python_file_path):
    ''' Perform doc tests on specific file

    Args
    ----
    python_file_path : str, valid filesystem path to Python file

    Post Condition
    --------------
    File path will always be printed to stdout.
    If any tests fail, error info printed to stdout.
    Otherwise, nothing else will be printed. No news is good news.
    '''
    print(python_file_path)
    root_path, filename = os.path.split(python_file_path)
    with cd(root_path):
        if six.PY2:
            v = "2"
        else:
            v = "3"
        CMD = "python{version} -m doctest {filename}".format(version=v, filename=filename)
        print(CMD)
        proc = subprocess.Popen(
                CMD.split(),
                shell=False)
        proc.wait()
        print("")

class cd:
    """ Context manager for changing the current working directory

    Source
    ------
    http://stackoverflow.com/questions/431684/how-do-i-cd-in-python#13197763
    """

    def __init__(self, newPath):
        self.newPath = os.path.expandvars(
            os.path.expanduser(newPath))

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

if __name__ == '__main__':
    bnpy_root_path = os.path.sep.join(
        os.path.abspath(__file__).split(os.path.sep)[:-2])
    bnpy_module_path = os.path.join(bnpy_root_path, 'bnpy/')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path',
        type=str,
        default=bnpy_module_path,
        help="Path to directory or file to test. May be absolute or relative to module directory bnpy/",
        )
    args = parser.parse_args()
    path = args.path

    if not path.startswith(os.path.sep):
        path = os.path.join(bnpy_module_path, path)

    if path.endswith('.py'):
        run_doctest_for_py_file(path)
    else:
        test_all_py_files_in_dir(path)
