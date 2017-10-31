from builtins import *
import ctypes
from numpy.ctypeslib import ndpointer

def LoadFuncFromCPPLib(libpath, srcpath, funcName):
    ''' Load function from compiled C++ library and make available to Python.
    '''
    try:
        # Load the compiled C++ library from disk
        lib = ctypes.cdll.LoadLibrary(libpath)
        with open(srcpath, 'r') as f:
            DO_READ = 0
            argTypeList = list()
            for line in f.readlines():
                line = line.strip()
                if DO_READ and line.count('}'):
                    break
                if line.count('extern "C"'):
                    DO_READ = 1
                elif DO_READ > 0:
                    if len(line) > 3:
                        if line.count(funcName):
                            DO_READ = 2
                        elif DO_READ == 2:
                            if line.count('int'):
                                curType = ctypes.c_int
                            elif line.count('double'):
                                curType = ctypes.c_double
                            else:
                                raise ValueError("MISSING TYPE")
                            if line.count('*'):
                                argTypeList.append(ndpointer(curType))
                            else:
                                argTypeList.append(curType)
            assert len(argTypeList) > 0
            setattr(getattr(lib, funcName), 'restype', None)
            setattr(getattr(lib, funcName), 'argtypes', argTypeList)
        return getattr(lib, funcName)
    except OSError as e:
        def errorFunc(*args, **kwargs):
            raise ImportError("Could not import C++ func: %s" % (funcName))
        return errorFunc
