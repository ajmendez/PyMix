import labeledBayesMixture
import mixture
import copy

# Setting up a three component Bayesian mixture over four features.
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

# initializing atomar distributions for second component
n31 = mixture.NormalDistribution(2.0,0.5)
n32 = mixture.NormalDistribution(-3.0,0.5)
d33 = mixture.DiscreteDistribution(4,[0.1,0.1,0.1,0.7])
d34 = mixture.DiscreteDistribution(4,[0.6,0.1,0.2,0.1])

# creating component distributions
c1 = mixture.ProductDistribution([n11,n12,d13,d14]) 
c2 = mixture.ProductDistribution([n21,n22,d23,d24]) 
c3 = mixture.ProductDistribution([n31,n32,d33,d34])

# setting up the mixture prior
piPr = mixture.DirichletPrior(3,[1.0,1.0,1.0])  # uniform prior of mixture coefficients

# conjugate priors over the atomar distributions - Normal-Gamma for Normal distribution, Dirichlet for the discrete distribution
compPrior = [ mixture.NormalGammaPrior(1.5,0.01,3.0,1.0), mixture.NormalGammaPrior(-2.0,0.01,3.0,1.0), 
            mixture.DirichletPrior(4,[1.01,1.01,1.01,1.01]), mixture.DirichletPrior(4,[1.01,1.01,1.01,1.01])]  

# putting together the mixture prior
prior = mixture.MixtureModelPrior(0.03,0.03, piPr,compPrior)
N = 400
prior.structPriorHeuristic(0.01, N)

# intializing Bayesian mixture model
pi = [0.3,0.3,0.4]
m = labeledBayesMixture.labeledBayesMixtureModel(3,pi,[c1,c2,c3],prior,struct=1)
print "Initial parameters"
print m

# sampling of the data set and assignement of sample labels

# first we draw 30 labeled sample from each component
l = m.components[0].sampleSet(30)
l += m.components[1].sampleSet(30)
l += m.components[2].sampleSet(30)

# the remaining, unlabeled samples are drawn from the whole mixture
l += m.sampleSet(310)

# a ConstrainedDataSet instance is initialised
cdat = mixture.ConstrainedDataSet()
cdat.fromList(l)

# the labels are assigned to the data set. By construction the 
# first 90 samples are labeled.
cdat.setConstrainedLabels([range(30),range(30,60,1),range(60,91,1) ])

# randomize model parameters before training
m.modelInitialization(cdat)

# run parameter estimation and structure learning
m.bayesStructureEM(cdat,1,5,80,0.1,silent=0)

print m
c = m.classify(cdat)



















