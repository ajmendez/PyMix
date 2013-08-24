import mixture

# Example for context-specific independence (CSI) structure learning.
# First we generate a data set from a three component mixture with a CSI like structure
# in the distribution parameters. Then a  five component CSI mixture is trained.
# The training should recover the true number of components (three), 
# the CSI structure of the generating model as well as the distribution parameters.


# Setting up the generating model. This is a benign case in the
# sense that the components are reasonably well separated and we 
# allow ourselves plenty of training data. 

# Component distributions
n11 = mixture.NormalDistribution(1.0,0.5)
n12 = mixture.NormalDistribution(2.0,1.5)
n13 = mixture.NormalDistribution(3.0,0.7)
d14 = mixture.DiscreteDistribution(4,[0.4,0.3,0.1,0.2])

c1 = mixture.ProductDistribution([n11,n12,n13,d14])

n21 = mixture.NormalDistribution(1.0,0.5)
n22 = mixture.NormalDistribution(-6.0,0.5)
n23 = mixture.NormalDistribution(3.0,0.7)
d24 = mixture.DiscreteDistribution(4,[0.1,0.1,0.4,0.4])

c2 = mixture.ProductDistribution([n21,n22,n23,d24])

n31 = mixture.NormalDistribution(2.0,0.5)
n32 = mixture.NormalDistribution(-3.0,0.5)
n33 = mixture.NormalDistribution(3.0,0.7)
d34 = mixture.DiscreteDistribution(4,[0.4,0.3,0.1,0.2])

c3 = mixture.ProductDistribution([n31,n32,n33,d34])

# creating the model
pi = [0.4,0.3,0.3]
m = mixture.MixtureModel(3,pi,[c1,c2,c3])

# sampling of the training data
data = m.sampleDataSet(800)

#---------------------------------------------------

# setting up the five component model we are going to train

tn11 = mixture.NormalDistribution(1.0,0.5)
tn12 = mixture.NormalDistribution(2.0,0.5)
tn13 = mixture.NormalDistribution(-3.0,0.5)
td14 = mixture.DiscreteDistribution(4,[0.25]*4)

tc1 = mixture.ProductDistribution([tn11,tn12,tn13,td14])

tn21 = mixture.NormalDistribution(4.0,0.5)
tn22 = mixture.NormalDistribution(-6.0,0.5)
tn23 = mixture.NormalDistribution(1.0,0.5)
td24 = mixture.DiscreteDistribution(4,[0.25]*4)

tc2 = mixture.ProductDistribution([tn21,tn22,tn23,td24])

tn31 = mixture.NormalDistribution(1.0,0.5)
tn32 = mixture.NormalDistribution(2.0,0.5)
tn33 = mixture.NormalDistribution(-3.0,0.5)
td34 = mixture.DiscreteDistribution(4,[0.25]*4)

tc3 = mixture.ProductDistribution([tn31,tn32,tn33,td34])

tn41 = mixture.NormalDistribution(4.0,0.5)
tn42 = mixture.NormalDistribution(-6.0,0.5)
tn43 = mixture.NormalDistribution(1.0,0.5)
td44 = mixture.DiscreteDistribution(4,[0.25]*4)

tc4 = mixture.ProductDistribution([tn41,tn42,tn43,td44])

tn51 = mixture.NormalDistribution(4.0,0.5)
tn52 = mixture.NormalDistribution(-6.0,0.5)
tn53 = mixture.NormalDistribution(1.0,0.5)
td54 = mixture.DiscreteDistribution(4,[0.25]*4)

tc5 = mixture.ProductDistribution([tn51,tn52,tn53,td54])

tpi = [0.3,0.2,0.2,0.2,0.1]

# the hyperparameter of the NormalGamma distributions are
# estimated heuristically in .setParams(...)
sp1 = mixture.NormalGammaPrior(1.0,1.0,1.0,1.0)
sp1.setParams(data.getInternalFeature(0),5)
sp2 = mixture.NormalGammaPrior(1.0,1.0,1.0,1.0)
sp2.setParams(data.getInternalFeature(1),5)
sp3 = mixture.NormalGammaPrior(1.0,1.0,1.0,1.0)
sp3.setParams(data.getInternalFeature(2),5)

sp4 = mixture.DirichletPrior(4,[1.02]*4)
pipr = mixture.DirichletPrior(5,[1.0]*5)

# the hyperparameter alpha is chosen based on the heuristic below
delta = 0.1
structPrior = 1.0 / (1.0+delta)**data.N

# creating the model prior
prior = mixture.MixtureModelPrior(structPrior,0.03, pipr,[sp1,sp2,sp3,sp4])

# creating the model
tm = mixture.BayesMixtureModel(5,tpi,[tc1,tc2,tc3,tc4,tc5],prior,struct=1)


# call to the learning algorithm
tm.bayesStructureEM(data,1,5,40,0.1)

# printing out the result of the training. The model should have three components and 
# parameters closely matching the generating model.
print "---------------------"
print tm
print tm.leaders
print tm.groups

