from distutils.core import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np

def main():
    np_incl = np.get_include()
    extensions = [
        Extension("AES_helper", ["AES_helper.pyx"],
            include_dirs=[np_incl, '.'])
        ]
    setup(
        name = 'AES helper extension',
        cmdclass = {'build_ext': build_ext},
        ext_modules = cythonize(extensions)
    )

if __name__ == "__main__":
    main()
