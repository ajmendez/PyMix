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
from distutils.errors import DistutilsExecError
import os
import sys

def guess_include_dirs():
    """
    The C extension requires the paths to Python.h and the numpy interface arrayobject.h.
    It is assumed that numpy is installed in the same directory structure as the Python installation
    setup.py is run with.
    
    The paths are assembled by making use of sys.prefix and sys.version_info.
    
    ( There is probably a more canonical version of doing this ...)
    """
    prefix = sys.prefix  # prefix of the python installation
    pyvs = str(sys.version_info[0]) + '.' + str(sys.version_info[1])  # major Python verion
    
    pypath = prefix + '/include/python' +pyvs  # path to Python.h 

    numpypath =  prefix + '/lib/python' +pyvs + '/site-packages/numpy/core/include/numpy'  # path to arrayobject.h

    return [pypath, numpypath]

include_dirs = guess_include_dirs()

# print '-------------------------------------------------------------------------------'
# print 'The following include paths are used for compilation of the C extension:\n'
# print 'Python.h: '+include_dirs[0]
# print 'arrayobject.h: '+include_dirs[1],'\n'
# print 'In case the installation fails, check these paths first.'
# print '-------------------------------------------------------------------------------\n'

setup(name="pymix",
      description="PyMix -- Python mixture package",
      version="0.8b",
      url ="http://www.pymix.org",
      
      author="Benjamin Georgi",
      author_email="georgi@molgen.mpg.de",
      license='LICENSE.txt',
      
      packages = ['pymix'],
      
      ext_modules = [Extension('pymix._C_mixextend',
                               ['pymix/C_mixextend.c'],
                               include_dirs = include_dirs,
                               libraries = ['gsl', 'gslcblas' ,'m'],
                               )
                     ],


      # py_modules = ['mixture','mixtureHMM','mixtureunittests','alphabet', 'plotMixture',
      #               'bioMixture', 'AminoAcidPropertyPrior','mixtureHMMunittests','randomMixtures', 'setPartitions'],

     )

# EOF: setup.py
