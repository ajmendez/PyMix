################################################################################
# 
#       This file is part of the Python Mixture Package
#
#       file:    AminoAcidPropertyPrior.py
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

Dirichlet mixture priors for amino acid data.

"""

import mixture
import numarray

def getAATable():
    """
    Returns the amino acid property table used to derive the paramters of the Dirichlet mixture prior
    """
    #    'A'  'R'  'N'  'D'  'C'  'Q'  'E'  'G'  'H'  'I'  'L'  'K'  'M'  'F'  'P'  'S'  'T'  'W'  'Y'  'V'  '-'
    M= [['X', '-', '-', '-', 'X', '-', '-', 'X', 'X', 'X', 'X', 'X', 'X', 'X', '-', '-', 'X', 'X', 'X', 'X', '-'],   # Hydrophobic
        ['-', 'X', 'X', 'X', '-', 'X', 'X', '-', 'X', '-', '-', 'X', '-', '-', '-', 'X', 'X', 'X', 'X', '-', '-'],   #  Polar
        ['X', '-', 'X', 'X', 'X', '-', '-', 'X', '-', '-', '-', '-', '-', '-', 'X', 'X', 'X', '-', '-', 'X', '-'],   #  Small
        ['X', '-', '-', '-', '-', '-', '-', 'X', '-', '-', '-', '-', '-', '-', '-', 'X', '-', '-', '-', '-', '-'],   #  Tiny
        ['-', '-', '-', '-', '-', '-', '-', '-', '-', 'X', 'X', '-', '-', '-', '-', '-', '-', '-', '-', 'X', '-'],   #  Aliphatic
        ['-', '-', '-', '-', '-', '-', '-', '-', 'X', '-', '-', '-', '-', 'X', '-', '-', '-', 'X', 'X', '-', '-'],   #  Aromatic
        ['-', 'X', '-', '-', '-', '-', '-', '-', 'X', '-', '-', 'X', '-', '-', '-', '-', '-', '-', '-', '-', '-'],   #  Positive
        ['-', '-', '-', 'X', '-', '-', 'X', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],   #  Negative
        ['-', 'X', '-', 'X', '-', '-', 'X', '-', 'X', '-', '-', 'X', '-', '-', '-', '-', '-', '-', '-', '-', '-']]   #  Charged  

    return M


def constructPropertyPrior(base,psum):
    """
    Constructs a Dirichlet mixture prior for amino acid alphabet based on nine
    chemical properties.
    
    @param base: value of alpha parameters when a property is not present
    @param psum: sum of all entries in each alpha parameter vector when a property is present

    @return: DirichletMixturePrior object
    """
    annot = ['Hydrophobic','Polar','Small','Tiny','Alphatic','Aromatic','Positive','Negative','Charged'] 

    M = getAATable()

    alph = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','-'] 
    alpha_M = []  # list of alpha parameter vectors for the Dirichlet Mixture
    pos_count = [ s.count('X') for s in M ] 
    for i in range(len(M)):
        r = [0.0] * 21
        for j,aa in enumerate(alph):
            if M[i][j] == 'X':
                r[j] = base + (float(psum) / pos_count[i])
            elif M[i][j] == '-':
                r[j] = base       
            else:
                raise RuntimeError
        alpha_M.append(r)

    pi_AA = [float(pos_count[i])/sum(pos_count) for i in range(len(M))] 

    dComp = []
    for i in range(len(M)):
        dComp.append( mixture.DirichletPrior(21, alpha_M[i]) )
    dirMixprior = mixture.DirichletMixturePrior(len(dComp),21,pi_AA,dComp)
    
    return dirMixprior



def printPropertyTable():
    """
    Pretty print of the amino acid property table.
    """

    annot = ['Hydrophobic','Polar','Small','Tiny','Alphatic','Aromatic','Positive','Negative','Charged'] # 'Proline'

    M = getAATable()

    alph = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','-'] 

    pos_count = [ s.count('X') for s in M ] 
    print ' '*20,
    
    for tt in alph:
        print tt,' ',
    print
    
    for i in range(len(M)):
        print "%-20s" % annot[i],
        r = [' '] * 21
        for j,aa in enumerate(alph):
            if M[i][j] == 'X':
                r[ j ] = 'X'
            elif M[i][j] == '-':
                r[ j ] = '-'
            else:
                raise RuntimeError
        for rr in r:
            print rr,' ',
        print


def readUCSCPrior(filename):
    """
    Reads files in the UCSC Dirichlet Mixture prior (DMP) format (http://www.soe.ucsc.edu/compbio/dirichlets/)
    and converts them into PyMix DirichletMixturePrior objects.
    
    Note that the alphabet in the UCSC priors does not contain the gap symbol. For the output DirichletMixturePrior
    the gap symbol is introduced with a parameter value of 0.01 in all components.
    
    @param filename: file in UCSC DMP format
    
    @return: DirichletMixturePrior object
    
    """
    f = open(filename,'r')

    ex1 = re.compile('Mixture=\s(\d+.\d+)')    
    ex2 = re.compile('Order\s*=\s+([A-Z\s]+)')
    ex3 = re.compile('Alpha=\s+([\d+.\d+\s+,\d+e-]+)')

    pi = []
    sigma = None
    dComp = []
    alpha_mat = []
    
    for l in f:
        l = mixture.chomp(l)
        m1 = ex1.match(l)
        if m1:
            pi.append( float(m1.groups(1)[0]))
        m2 = ex2.match(l)
        if m2:
            s = m2.groups(1)[0]
            sigma = s.split(' ')
            
        m3 = ex3.match(l)
        if m3:
            s = m3.groups(1)[0]
            alpha = s.split(' ')
            alpha = map(float,alpha)
            as =  alpha.pop(0) # first entry is the sum of the others -> remove
            alpha_mat.append(alpha)

    # intergrate gap character '-' into the alphabet
    sigma.append('-')
    alphabet = mixture.Alphabet(sigma)        
   
    for i in range(len(alpha_mat)):
        alpha_mat[i].append(0.01) # add hyper paramerter for '-'
        dComp.append( mixture.DirichletPrior(21,alpha_mat[i]) )

    prior = mixture.DirichletMixturePrior(len(dComp),21,pi,dComp)
    return alphabet,prior



