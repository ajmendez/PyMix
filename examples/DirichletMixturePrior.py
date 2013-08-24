################################################################################
# 
#       This file is part of the Python Mixture Package
#
#       file:    DirichletMixturePrior.py
#       author: Benjamin Georgi
#
#       Copyright (C) 2004-2007 Benjamin Georgi
#       Copyright (C) 2004-2007 Max-Planck-Institut fuer Molekulare Genetik,
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

"""
Dirichlet mixture prior example.

"""
import mixture

# Setting up a two component Bayesian mixture over four features.
# Two features are Normal distributions, two discrete.

# initializing atomar distributions for first component
n11 = mixture.NormalDistribution(1.0,1.5)
n12 = mixture.NormalDistribution(2.0,0.5)
d13 = mixture.DiscreteDistribution(4,[0.1,0.4,0.4,0.1])
d14 = mixture.DiscreteDistribution(4,[0.25,0.25,0.25,0.25])

# initializing atomar distributions for second component
n21 = mixture.NormalDistribution(4.0,0.5)
n22 = mixture.NormalDistribution(-6.0,0.5)
d23 = mixture.DiscreteDistribution(4,[0.7,0.1,0.1,0.1])
d24 = mixture.DiscreteDistribution(4,[0.1,0.1,0.2,0.6])

# creating component distributions
c1 = mixture.ProductDistribution([n11,n12,d13,d14])
c2 = mixture.ProductDistribution([n21,n22,d23,d24])

# setting up a Dirichlet mixture prior with two components
piPr = mixture.DirichletPrior(2,[1.0,1.0])  # uniform prior of mixture coefficients
dPrior= [ mixture.DirichletPrior(4,[1.3,1.6,1.1,4.0]),
          mixture.DirichletPrior(4,[3.1,1.2,1.1,1.0])  ] 
dmixPrior = mixture.DirichletMixturePrior(2,4,[0.5,0.5],dPrior)

# assembling the model prior
compPrior = [ mixture.NormalGammaPrior(1.5,0.1,3.0,1.0), mixture.NormalGammaPrior(-2.0,0.1,3.0,1.0),
            dmixPrior, dmixPrior]

# putting together the prior for the whole mixture
prior = mixture.MixtureModelPrior(0.03,0.03, piPr,compPrior)


# intializing Bayesian mixture model
pi = [0.4,0.6]
m = mixture.BayesMixtureModel(2,pi,[c1,c2],prior)
print "Initial parameters"
print m
# Now that the model is complete we can start using it.

# sampling data
data = m.sampleDataSet(600)

# randomize model parameters
m.modelInitialization(data)
print "Randomized parameters"
print m

# parameter training
m.mapEM(data,40,0.1)

print "Retrained parameters"
print m


# clustering
c = m.classify(data,silent=1)

