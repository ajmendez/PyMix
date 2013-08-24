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

# setting up the mixture prior
piPr = mixture.DirichletPrior(2,[1.0,1.0])  # uniform prior of mixture coefficients

# conjugate priors over the atomar distributions - Normal-Gamma for Normal distribution, Dirichlet for the discrete distribution
compPrior = [ mixture.NormalGammaPrior(1.5,0.1,3.0,1.0), mixture.NormalGammaPrior(-2.0,0.1,3.0,1.0),
            mixture.DirichletPrior(4,[1.0,1.0,1.0,1.0]), mixture.DirichletPrior(4,[1.0,1.0,1.0,1.0])]

# putting together the mixture prior
prior = mixture.MixtureModelPrior(0.03,0.03, piPr,compPrior)


# intializing Bayesian mixture model
pi = [0.4,0.6]
m = mixture.BayesMixtureModel(2,pi,[c1,c2],prior,struct = 1)
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

