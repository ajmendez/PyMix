#-------------------------------------------------------------------------------
#- EmissionDomain and derived  -------------------------------------------------
class EmissionDomain:
    """ Abstract base class for emissions produced by an HMM.

        There can be two representations for emissions:
        1) An internal, used in ghmm.py and the ghmm C-library
        2) An external, used in your particular application

        Example:
        The underlying library represents symbols from a finite,
        discrete domain as integers (see Alphabet).

        EmissionDomain is the identity mapping
    """

    def internal(self, emission):
        """ Given a emission return the internal representation
        """
        return emission


    def internalSequence(self, emissionSequence):
        """ Given a emissionSequence return the internal representation
        """
        return emissionSequence


    def external(self, internal):
        """ Given an internal representation return the
            external representation
        """
        return internal

    def externalSequence(self, internalSequence):
        """ Given a sequence with the internal representation return the
            external representation
        """
        return internalSequence


    def isAdmissable(self, emission):
        """ Check whether emission is admissable (contained in) the domain
            raises GHMMOutOfDomain else
        """
        return None


class Alphabet(EmissionDomain):
    """ Discrete, finite alphabet

    """
    def __init__(self, listOfCharacters):
        """ Creates an alphabet out of a listOfCharacters """
        self.listOfCharacters = listOfCharacters
        self.index = {} # Which index belongs to which character
        i = 0
        for c in self.listOfCharacters:
            self.index[c] = i
            i += 1
        self.CDataType = "int" # flag indicating which C data type should be used

    def __str__(self):
        strout = "GHMM Alphabet:\n"
        strout += "Number of symbols: " + str(len(self)) + "\n"
        strout += "External: " + str(self.listOfCharacters) + "\n"
        strout += "Internal: " + str(range(len(self))) + "\n"
        return strout
    

    def __len__(self):
        return len(self.listOfCharacters)
        

    def size(self):
        """ Deprecated """
        print "Warning: The use of .size() is deprecated. Use len() instead."
        return len(self.listOfCharacters)

        
    def internal(self, emission):
        """ Given a emission return the internal representation
        """
        return self.index[emission]


    def internalSequence(self, emissionSequence):
        """ Given a emission_sequence return the internal representation

            Raises KeyError
        """
        
        result = copy.deepcopy(emissionSequence)
        try:
            result = map(lambda i: self.index[i], result)
        except IndexError:
            raise KeyError
        return result


    def external(self, internal):
        """ Given an internal representation return the
            external representation

            Raises KeyError
        """
        if internal < 0 or len(self.listOfCharacters) <= internal:
            raise KeyError, "Internal symbol "+str(internal)+" not recognized."
        return self.listOfCharacters[internal]

    def externalSequence(self, internalSequence):
        """ Given a sequence with the internal representation return the
            external representation
        """
        result = copy.deepcopy(internalSequence)
        try:
            result = map(lambda i: self.listOfCharacters[i], result)
        except IndexError:
            raise KeyError
        return result

    def isAdmissable(self, emission):
        """ Check whether emission is admissable (contained in) the domain
        """
        return emission in self.listOfCharacters



DNA = Alphabet(['a','c','g','t'])
AminoAcids = Alphabet(['A','C','D','E','F','G','H','I','K','L',
                       'M','N','P','Q','R','S','T','V','W','Y'])
def IntegerRange(a,b):
    l = range(a,b)
    for i,s in enumerate(l):
        l[i] = str(s)
    return Alphabet(l)


# To be used for labelled HMMs. We could use an Alphabet directly but this way it is more explicit.
class LabelDomain(Alphabet):    
    def __init__(self, listOfLabels):
        Alphabet.__init__(self, listOfLabels)


class Float(EmissionDomain):

    def __init__(self):
        self.CDataType = "double" # flag indicating which C data type should be used

    def isAdmissable(self, emission):
        """ Check whether emission is admissable (contained in) the domain
            raises GHMMOutOfDomain else
        """
        return isinstance(emission,float)

