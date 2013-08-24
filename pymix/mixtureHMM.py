################################################################################
#
#       This file is part of the Python Mixture Package
#
#       file:    mixtureHMM.py
#       author: Benjamin Georgi
#               Ivan Gesteira Costa Filho
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
Mixtures of HMMs (requires the GHMM package)
"""
import mixture
import ghmm 
import numpy
import copy


class SequenceDataSet(mixture.ConstrainedDataSet):
    def __init__(self):
        mixture.ConstrainedDataSet.__init__(self)
    
        # Sequence DataSets introduce a whole lot of tedious index book-keeping into the DataSet 
        # since we cannot store the sequence objects
        # in the numaray internalData. The introduction of additional data structures for the SequenceSets means that
        # we have to map the external, global feature indices into different data structures.
        self.complex = None  # flag for complex DataSet
        self.complexFeature = None  # feature-wise list of flags for complex data

        # We store the GHMM SequenceSets in the list complexData. The map from external feature indices 
        # to indices in complexData is stored in complexDataIndexMap.
        self.complexData = None 
        self.complexDataIndexMap = {}

        # XXX for now we require that all GHMM sequence features are assigned the
        # highest sequence indices. That means indices (e.g. in suff_dataRange) don't change 


    def __copy__(self):        
        if self.complexData:
            raise NotImplementedError, "No copying of complex DataSets yet."


    def fromGHMM(self,List,sequences, IDs = None, col_header=None):
        """
        Construct a complex DataSet. The non-sequence data is given as a Python list 'List', the
        sequence data as GHMM SequenceSet objects in the 'sequences' list. There has to be one 
        SequenceSet per HMM feature in a corresponding MixtureModel.
        """

        self.complex = 1 # DataSet is complex
        self.seq_p = len(sequences)

        # same number of samples ?
        for seq in sequences:
            assert len(sequences[0]) == len(seq)

        
        # consistency checks between non-sequence and sequence data
        self.N = len(sequences[0])
        self.p = len(sequences)
        if len(List) > 0:
            assert self.N == len(List)
            self.p += len(List[0])
         
        if not IDs:
            self.sampleIDs = range(self.N)
        else:
            assert len(IDs) == self.N
            self.sampleIDs = IDs

        if not col_header:
             l = range(self.p)
             self.headers = []
             for k in l:
                self.headers.append(str(k))
        else:
            assert len(col_header) == self.p
            self.headers = col_header   

        # set non-sequence data
        self.dataMatrix = List

        # set complex data
        self.complexData = sequences
        
        #print "\n",self.complexFeature
        #print self.dataMatrix
        #print self.complexData

    def internalInit(self,m):
        """
        Initializes the internal representation of the data
        used by the EM algorithm. 
        
        @param m: MixtureModel object
        

        """
        assert m.p == self.p,"Invalid dimensions in data and model."+str(m.p)+' '+str(self.p)

        if self.complex:
            # set complexFeature flags
            self.complexFeature = []
            
            if self.p == 1:
                if isinstance(m.components[0], mixture.ProductDistribution):
                    assert m.components[0].dist_nr == 1
                    assert isinstance(m.components[0].distList[0],HMM)
                else:
                    assert isinstance(m.components[0],HMM)
                    
                self.complexFeature = [1]
                self.complexDataIndexMap[0] = 0
                
            else:
                ind = 0
                for i in range( m.components[0].dist_nr ):
                
                
                    if isinstance(m.components[0].distList[i], HMM):
                        self.complexFeature.append(1)
                        self.complexDataIndexMap[i] = ind
                        ind += 1
                    else:
                        self.complexFeature.append(0)
                    
       
        templist = []
        for i in range(len(self.dataMatrix)):
            
            [t,dat] = m.components[0].formatData(self.dataMatrix[i])

            #print self.dataMatrix[i] ,"->",dat

            templist.append(dat)

        sequence = numpy.array(templist)
       
        self.internalData = sequence
        
        if m.dist_nr > 1:
            self.suff_dataRange = copy.copy(m.components[0].suff_dataRange) 
        else:    
            self.suff_dataRange = [m.suff_p]
            
        self.suff_p = m.components[0].suff_p
        
        #print 'p',self.p
        #print 'seq_p',self.seq_p


    def removeFeatures(self, ids):
        raise NotImplementedError,"Needs implementation"
        
    def removeSamples(self,fid,min_value,max_value):
        raise NotImplementedError,"Needs implementation"
        
    def maskDataSet(self, valueToMask, maskValue):
        raise NotImplementedError,"Needs implementation"
        
    def maskFeatures(self, headerList, valueToMask, maskValue):
        raise NotImplementedError,"Needs implementation"
        
    def getExternalFeature(self, fid):
        raise NotImplementedError,"Needs implementation"
        
    def getInternalFeature(self, i):
        """
        Returns the columns of self.internalData containing the data of the feature with index 'i'
        """
                
        #print "***",i
        #print "***",self.complexFeature

        #internal_index = i
         
        assert self.suff_dataRange is not None  
        if i < 0 or i >= len(self.suff_dataRange):
            raise IndexError, "Invalid index " + str(i)

        if i == 0:
            prev_index = 0
        else:    
            prev_index =  self.suff_dataRange[i-1]  
        
        this_index =  self.suff_dataRange[i]

        #print "suff_dataRange",self.suff_dataRange
        #print "i",i, "->",self.complexFeature[i]
        #print "this",this_index
        
        if self.complex == 1:   
            # if feature 'internal_index' is complex we return the appropriate SequenceSet
            if self.complexFeature[i] == 1:
        
                #print "COMPLEX!"
        
                internal_index = self.complexDataIndexMap[i]
                return self.complexData[internal_index ]
       
        if self.p == 1:   # only a single feature
            return self.internalData[:]
        
        elif (this_index - prev_index) == 1:   # multiple features, feature 'i' has single dimension 
            return numpy.take(self.internalData,(this_index-1,),axis=1)
        else:
            return self.internalData[:,prev_index:this_index ]  # multiple features, feature 'i' has multiple dimensions
        
    def extractSubset(self,ids):
        raise NotImplementedError,"Needs implementation"
        



def getHMM(emissionDomain, distribution, A, B, pi,name=None):
    """
    Takes HMM-style parameter matrices and returns a mixture.HMM object which was 
    intialised with a GHMM object using these parameters.
    """
    hmm = ghmm.HMMFromMatrices(emissionDomain, distribution, A, B, pi,hmmName=name)
    return HMM(hmm,1)


class HMM(mixture.ProbDistribution): 
    """
    Wrapper class for GHMM Hidden Markov Models.
    """
    
    def __init__(self,hmm,iterations):
        """
        Interface to ghmm.HMMFromMatrices
        """
        
        self.hmm = hmm

        self.p = 1   # we consider each sequence set to be a single features, so p is one.
        
        #  getting the free parameters of a GHMM object requires iteration over all states
        self.freeParams = self.hmm.N-1 # pi
        self.freeParams += (self.hmm.N-1) * self.hmm.N  # transitions
        for i in range(self.hmm.N):
            if isinstance(self.hmm.emissionDomain,ghmm.Alphabet):
                # discrete emissions
                self.freeParams += (self.hmm.M-1)  # XXX for now only first order HMMS ** self.hmm.order[i]
            elif isinstance(self.hmm.emissionDomain, ghmm.Float):
                # gaussian emissions
                self.freeParams += (self.hmm.M * 2) + (self.hmm.M -1)
            else:
                raise TypeError, "Unknown EmissionDomain "+str(self.hmm.emissionDomain.__class__)
        
        
        
        self.suff_p = 1  # since we save the whole sequence set under on index suff_p is 1 
        self.dist_nr = 1
        self.iterations=iterations
        

    def __eq__(self,other):
        raise NotImplementedError, "Needs implementation"

    def __str__(self):
        #return str(self.hmm)
        return "< GHMM object >"

    def __copy__(self):
        raise NotImplementedError, "Needs implementation"
    
    def pdf(self,data):

        if isinstance(data, SequenceDataSet ):
            assert len(data.complexData) == 1
            data = data.complexData[0]
        elif isinstance(data, ghmm.SequenceSet):
            pass
        else:
            raise TypeError,"Unknown/Invalid input type:"+str(data)

        
        #print "pdf "+str(self.hmm.cmodel.name) +":"
        #print self.hmm.loglikelihoods(data)
        return numpy.array(self.hmm.loglikelihoods(data), dtype='Float64')
        

    def MStep(self,posterior,data,mix_pi=None):       
        
        if isinstance(data,SequenceDataSet):            
            assert len(data.complexData) == 1            
            data = data.complexData[0]
        elif isinstance(data,ghmm.SequenceSet):
            pass
        else:
            raise TypeError, "Unknown/Invalid input to MStep."    
        
        # set sequence weights according to posterior
        for i in range(len(data)):
            data.setWeight(i,posterior[i])

        # run BaumWelch for parameter update
        self.hmm.baumWelch(data,self.iterations, 0.0001)
                
        
    def sample(self):
        raise NotImplementedError
        
    def sampleSet(self,nr):
        raise NotImplementedError
        

    def isValid(self,x):
        if not isinstance(x,ghmm.SequenceSet):
            raise InvalidDistributionInput, "\nInvalid data in HMM."
        
    def sufficientStatistics(self,x):
       pass
       #raise NotImplementedError, "Needs implementation"
    
    
    def flatStr(self,offset):
        raise NotImplementedError, "Needs implementation"                
        
    def posteriorTraceback(self,x):
        raise NotImplementedError, "Kawoom !"

    def update_suff_p(self):
        return self.suff_p
        
    def merge(self,dlist,weights):
       raise NotImplementedError, "Kawoom !"
        
    def sortStr(self):
        raise NotImplementedError, "Kawoom !"



