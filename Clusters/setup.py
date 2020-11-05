from distutils.core import setup
from Cython.Build import cythonize

# This compiles the cython code (annotate True creates the html file)
# In the html file the more yellow means the most python it's running
#   instead of C
setup(ext_modules = cythonize("*.pyx", annotate=True))
