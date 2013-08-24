#!/usr/bin/env python
from distutils.core import setup, Extension
import distutils.sysconfig
import numpy.distutils.misc_util

# Get the arrayobject.h(numpy) and python.h(python) header file paths:
include_dirs = numpy.distutils.misc_util.get_numpy_include_dirs()
include_dirs.insert(0, distutils.sysconfig.get_python_inc())

# Get the text for the readme and license
with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(name='pymix',
      version="0.8b",
      
      description='PyMix -- Python mixture package',
      long_description=readme, 
      
      author="Benjamin Georgi",
      author_email="georgi@molgen.mpg.de",
      url ="http://www.pymix.org",
      license=license,
      
      packages = ['pymix', 'pymix.examples', 'pymix.tests'],
      
      ext_modules = [Extension('_C_mixextend',
                               ['pymix/C_mixextend.c'],
                               include_dirs = include_dirs,
                               libraries = ['gsl', 'gslcblas' ,'m'],
                               )
                     ],

      requires = [
          'numpy',
      ]

     )

# EOF: setup.py
