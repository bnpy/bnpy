'''
Run all tests using nose

Usage
-----
$ python run_nosetests.py
'''

import os
import nose
import glob
import subprocess

def add_trailing_sep(path):
    if not path.endswith(os.path.sep):
        path = path + os.path.sep
    return path

testroot = add_trailing_sep(
    os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-1]))

test_folder_list = glob.glob(os.path.join(testroot, '*'))
for folder in test_folder_list:
    folder = add_trailing_sep(folder)
    if not os.path.isdir(folder):
        continue
    if folder.count('zzz') or folder.count('endtoend'):
        continue
    CMD = "nosetests %s -v --nocapture" % (folder)
    print(CMD)
    print("------------------------------------------------------------")
    proc = subprocess.Popen(
            CMD.split(),
            shell=False)
    proc.wait()

    #result = nose.run(argv=CMD.split())
