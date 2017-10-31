from builtins import *
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules=[
    Extension("EntropyUtilX",
        ["EntropyUtilX.pyx"],
        libraries=["m"],
        extra_compile_args = ["-O3", "-ffast-math"]),
    Extension("SparseRespUtilX",
        ["SparseRespUtilX.pyx"],
        libraries=["m"],
        extra_compile_args = ["-O3", "-ffast-math"]),
    Extension("TextFileReaderX",
        ["TextFileReaderX.pyx"],
        libraries=["m"],
        extra_compile_args = [])
    ]

# CYTHON DIRECTIVES
# http://docs.cython.org/src/reference/compilation.html#compiler-directives
for e in ext_modules:
    e.cython_directives = {
		'embedsignature':True,
		'boundscheck':False,
		'nonecheck':False,
		'wraparound':False}

setup(
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules)
