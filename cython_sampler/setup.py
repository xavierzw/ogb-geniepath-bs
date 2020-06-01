# cython: language_level=3
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

import os
os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

setup(ext_modules = cythonize(["cython_sampler/cython_sampler.pyx","cython_sampler/cython_utils.pyx"]), include_dirs = [numpy.get_include()])
