import os
from setuptools import setup, Extension
from Cython.Distutils import build_ext
from distutils.sysconfig import customize_compiler

def get_path_to_eigen():
    try:
        eigen_path = os.environ['EIGENPATH']
    except KeyError as e:
        eigen_path = ''
    return eigen_path

def get_path_to_boost():
    try:
        eigen_path = os.environ['BOOSTMATHPATH']
    except KeyError as e:
        eigen_path = ''
    return eigen_path

def make_extensions():
    ''' Assemble C++/Cython extension objects for compilation

    Warns user if required environment variables are not specified.

    Returns
    -------
    ext_list : list of extension objects
    '''
    ext_list = [
        make_cython_extension_SparseRespUtilX(),
        make_cython_extension_EntropyUtilX(),
        ]
    if get_path_to_eigen():
        ext_list.append(make_cpp_extension_libfwdbwd())
        ext_list.append(make_cpp_extension_libsparsemix())
    else:
        print "Warning: Environment variable EIGENPATH not found."
        print "Will not compile the following C++ extensions"
        print " - libsparsemix (for L-sparse mixtures)"
        print " - libfwdbwd (for fast local steps for HMMs)"
    if get_path_to_eigen() and get_path_to_boost():
        ext_list.append(make_cpp_extension_libsparsetopics())
        ext_list.append(make_cpp_extension_libsparsetopicsmanydocs())
    else:
        print "Warning: Environment variable BOOSTMATHPATH not found."
        print "Will not compile the following C++ extensions"
        print " - libsparsetopics (for L-sparse topic models)"
    return ext_list

def make_cpp_extension_libfwdbwd():
    ext = Extension(
        'bnpy.allocmodel.hmm.lib.libfwdbwdcpp',
        sources=['bnpy/allocmodel/hmm/lib/FwdBwdRowMajor.cpp'],
        include_dirs=[get_path_to_eigen()],
    )
    return ext

def make_cpp_extension_libsparsemix():
    ext = Extension(
        'bnpy.util.lib.sparseResp.libsparsemix',
        sources=['bnpy/util/lib/sparseResp/SparsifyRespCPPX.cpp'],
        include_dirs=[get_path_to_eigen()],
    )
    return ext

def make_cpp_extension_libsparsetopics():
    ext = Extension(
        'bnpy.util.lib.sparseResp.libsparsetopics',
        sources=['bnpy/util/lib/sparseResp/TopicModelLocalStepCPPX.cpp'],
        include_dirs=[get_path_to_eigen(), get_path_to_boost()],
    )
    return ext

def make_cpp_extension_libsparsetopicsmanydocs():
    ext = Extension(
        'bnpy.util.lib.sparseResp.libsparsetopicsmanydocs',
        sources=[
            'bnpy/util/lib/sparseResp/TopicModelLocalStepManyDocsCPPX.cpp'],
        include_dirs=[get_path_to_eigen(), get_path_to_boost()],
    )
    return ext

def make_cython_extension_EntropyUtilX():
    ext = Extension(
        "bnpy.util.EntropyUtilX",
        sources=["bnpy/util/EntropyUtilX.pyx"],
        libraries=["m"],
        extra_compile_args = ["-O3", "-ffast-math"],
        )
    return add_directives_to_cython_ext(ext)

def make_cython_extension_SparseRespUtilX():
    ext = Extension(
        "bnpy.util.SparseRespUtilX",
        sources=["bnpy/util/SparseRespUtilX.pyx"],
        libraries=["m"],
        extra_compile_args = ["-O3", "-ffast-math"],
        )
    return add_directives_to_cython_ext(ext)

def add_directives_to_cython_ext(ext):
    ''' Improve speed of cython code extensions

    References
    ----------
    http://docs.cython.org/src/reference/compilation.html#compiler-directives
    '''
    ext.cython_directives = {
        'embedsignature':True,
        'boundscheck':False,
        'nonecheck':False,
        'wraparound':False}
    return ext

def read_txt_file_as_string(fname='README.md'):
    s = ""
    with open(os.path.join(os.path.dirname(__file__), fname), 'r') as f:
        s = f.read()
    return s

class CustomizedBuildExt(build_ext):
    ''' Custom override of some default build options

    Basically, avoids some crappy options.

    References
    ----------
    http://stackoverflow.com/questions/8106258/cc1plus-warning-command-line-option-wstrict-prototypes-is-valid-for-ada-c-o
    '''
    def build_extensions(self):
        customize_compiler(self.compiler)
        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
        except (AttributeError, ValueError):
            pass
        build_ext.build_extensions(self)

def make_list_of_subpackages():
    ''' Traverse subdirectories recursively and add to list
    '''
    package_list = []
    # Traverse root directory, and list directories as dirs and files as files
    for root, dirpath_list, fpath_list in os.walk('bnpy/'):
        subpkg_path = root.strip(os.path.sep).replace(os.path.sep, '.')
        for fpath in fpath_list:
            if fpath == "__init__.py":
                package_list.append(subpkg_path)
    return package_list

def make_list_of_datasets_specs():
    ''' Traverse subdirectories recursively and add to list
    '''
    data_spec_list = []
    data_spec_list.append(
        ('', ['requirements.txt']),
        )
    data_spec_list.append(
        ('', ['LICENSE.txt']),
        )
    # Traverse root directory, and list directories as dirs and files as files
    for root, dirpath_list, fpath_list in os.walk('datasets/'):
        for fpath in fpath_list:
            if (fpath.endswith('.npz')
                    or fpath.endswith('.csv')
                    or fpath.endswith('.txt')):
                full_fpath = os.path.join(root, fpath)
                full_dirpath = os.path.split(full_fpath)[0] + os.path.sep
                data_spec = (
                    full_dirpath,
                    [full_fpath])
                print data_spec
                data_spec_list.append(data_spec)
    return data_spec_list


########################################################################
# Main function
setup(
    name="bnpy",
    version="0.1.5",
    author="Michael C. Hughes",
    author_email="mike@michaelchughes.com",
    description=(
        "Bayesian nonparametric machine learning for Python"),
    license="BSD",
    keywords=[
        "Dirichlet process",
        "mixture model",
        "topic model",
        "hidden Markov model"],
    url="https://github.com/bnpy/bnpy",
    packages=make_list_of_subpackages(),
    package_data = {
        # If any subpackage contains these files, include them:
        '': ['*.conf', '*.txt', '*.md', '*.npz', '*.csv'],
        '': ['*.cpp', '*.c', '*.h', '*.pyx'],
    },
    include_package_data=True,
    long_description='',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: BSD License"],
    install_requires=[
        "matplotlib>=1.5",
        "numpy>=1.11",
        "pandas>=0.18",
        "scipy>=0.18",
        "Cython>=0.25",
        "ipython>=5.1",
        "joblib>=0.10",
        "memory_profiler>=0.41",
        "munkres>=1.0",
        "numexpr>=2.6",
        "psutil>=5.0",
        "scikit_learn>=0.18",
        "sphinx_gallery>=0.1",
        ],
    zip_safe=False,
    ext_modules=make_extensions(),
    cmdclass=dict(build_ext=CustomizedBuildExt),
)
