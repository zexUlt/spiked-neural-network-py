from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

extentions = ['cboosted/misc/_misc.pyx',
              'cboosted/ellipse_algorithms.pyx']

setup(
    ext_modules=cythonize(extentions, compiler_directives={'language_level': "3"}
                          ),
    include_dirs=[np.get_include()]
)
