pymix
=====

PyMix - The Python mixture package  
Author: Benjamin Georgi
Contact: georgi@molgen.mpg.de
Contributors: Alexander Mendez

Installation Instructions:
--------------------------

* Required Packages

        # Python (version 2.5.2 recommended)
        # Numpy
        # GSL GNU Scientific library
    
* Optional Packages

        # GHMM for mixtures of HMMs
        # pylab for plotting functions in plotMixture.py

* How to install ?

    Extract the tarball to some directory of your choice. 
    Change into the directory and run:
    
        python setup.py build
        python setup.py install --prefix=/some/where

    After the installation is completed I would recommend to run 
    mixtureunittests.py to check whether everything is in order.
    

Documentation:
--------------
    Example code for most aspects of the library can be found in 
    the /examples subdirectory and mixtureunittest.py.
    Automatically generated documentation for the module is available 
    on the Pymix home page www.pymix.org.