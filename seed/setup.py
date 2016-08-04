from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

setup(
    extensions =[Extension("*", ["*.pyx"])],
    cmdclass={'build_ext': Cython.Build.build_ext},
)