#!/usr/bin/env python
################################################################################
#
#       This file is part of the Python Mixture Package 
#
#       file:    setup.py
#       author: Benjamin Georgi
#
#       Copyright (C) 2004-2009 Benjamin Georgi
#       Copyright (C) 2004-2009 Max-Planck-Institut fuer Molekulare Genetik,
#                               Berlin
#
#       Contact: georgi@molgen.mpg.de
#
#       This library is free software; you can redistribute it and/or
#       modify it under the terms of the GNU Library General Public
#       License as published by the Free Software Foundation; either
#       version 2 of the License, or (at your option) any later version.
#
#       This library is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#       Library General Public License for more details.
#
#       You should have received a copy of the GNU Library General Public
#       License along with this library; if not, write to the Free
#       Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#
#
#
################################################################################
from distutils.core import setup, Extension
import distutils.sysconfig
import numpy.distutils.misc_util
import os
import sys

# Get the arrayobject.h(numpy) and python.h(python) header file paths:
include_dirs = numpy.distutils.misc_util.get_numpy_include_dirs()
include_dirs.insert(0, distutils.sysconfig.get_python_inc())


setup(name='pymix',
      description='PyMix -- Python mixture package',
      version="0.8b",
      url ="http://www.pymix.org",
      
      author="Benjamin Georgi",
      author_email="georgi@molgen.mpg.de",
      license='LICENSE.txt',
      
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
