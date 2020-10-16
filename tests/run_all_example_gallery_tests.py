import os
import sys
import subprocess
import time

from tempfile import NamedTemporaryFile

def run_script_file(
        script_path,
        timeout_sec=60):
    """ Execute a given python script and collect stdout and stderr.

    Returns
    -------
    stdout_lines
    stderr_lines
    """
    start_time_sec = time.time()
    s = script_path
    with NamedTemporaryFile(mode='r+w', prefix=s, suffix='stdout') as f_out:
        with NamedTemporaryFile(mode='r+w', prefix=s, suffix='stderr') as f_err:
            script_process = subprocess.call(
                ["python", script_path],
                stdout=f_out,
                stderr=f_err)

            f_out.flush()
            f_out.seek(0)
            f_err.flush()
            f_err.seek(0)
            stdout_lines = f_out.readlines()
            stderr_lines = f_err.readlines()

    elapsed_time_sec = time.time() - start_time_sec
    return stdout_lines, stderr_lines, elapsed_time_sec

if __name__ == '__main__':
    bnpy_root_path = os.path.sep.join(
        os.path.abspath(__file__).split(os.path.sep)[:-2])
    example_gallery_path = os.path.join(bnpy_root_path, 'examples/')

    for cur_root, dirs, files in os.walk(example_gallery_path):
        dirs.sort()  # Process dirs  in sorted order
        files.sort() # Process files in sorted order

        print("==================")
        print("Current directory:")
        print(cur_root)

        os.chdir(cur_root)
        for file in files:
            if file.startswith("plot-") and file.endswith(".py"):
                print("-- Running file:")
                print("-- " + file)
                
                stdout, stderr, elapsed_time_sec = run_script_file(file)
                if len(stderr) > 0:
                    for line in stderr:
                        print(line.strip())
                    raise ValueError("Error raised. See traceback above.")

                print("%9.1f seconds elapsed. Done." % elapsed_time_sec)
