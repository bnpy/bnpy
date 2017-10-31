import glob
import os
import sys

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError


def verify_notebook_output(root_path):
    for notebook in glob.glob(os.path.join(root_path, "*.ipynb")):
        print(notebook)
        nb, errors = execute_notebook(notebook)
        assert errors == []

def execute_notebook(
        path,
        timeout_sec=60):
    """ Execute a notebook via nbconvert and collect output.

    Returns
    -------
    parsed_nb : parsed nb object
    err_list : list of execution errors
    """
    kernel_name = 'python%d' % sys.version_info[0]
    this_file_directory = os.path.dirname(__file__)
    errors = []

    with open(path) as f:
        nb = nbformat.read(f, as_version=4)
        nb.metadata.get('kernelspec', {})['name'] = kernel_name
        ep = ExecutePreprocessor(
            kernel_name=kernel_name,
            timeout=timeout_sec,
            allow_errors=False)

        try:
            ep.preprocess(nb, {'metadata': {'path': this_file_directory}})
        except CellExecutionError as e:
            if "SKIP" in e.traceback:
                print((str(e.traceback).split("\n")[-2]))
            else:
                raise e
    return nb, errors

if __name__ == '__main__':
    bnpy_root_path = os.path.sep.join(
        os.path.abspath(__file__).split(os.path.sep)[:-2])
    bnpy_demo_path = os.path.join(bnpy_root_path, 'demos/')

    verify_notebook_output(bnpy_demo_path)
