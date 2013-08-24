import mixture

# Setting up a two component mixture over four features.
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

# intializing mixture
pi = [0.4,0.6]
m = mixture.MixtureModel(2,pi,[c1,c2])
print "Initial parameters"
print m
# Now that the model is complete we can start using it.

# sampling data
data = m.sampleDataSet(20)

# randomize model parameters
m.modelInitialization(data)
print "Randomized parameters"
print m

# parameter training
m.EM(data,40,0.1)
print "Retrained parameters"
print m

# clustering
c = m.classify(data,silent=1)
