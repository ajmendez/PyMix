################################################################################
# 
#       This file is part of the Python Mixture Package
#
#       file:   setPartitions.py 
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

""" 

 The following functions enumerate all possible partitions of a set
 Necessary for the structure learning with exhaustive enumeration.
 
 Based on 'Efficient Generation of Set Partitions' by Michael Orlov
 www.cs.bgu.ac.il/~orlovm/papers/partitions.pdf 

"""

import numpy

def init_first(l):
    kap = numpy.zeros(l)
    maxkap = numpy.zeros(l)


    return kap, maxkap
    
    
def init_last(l):
    kap = numpy.arange(l)
    maxkap = numpy.arange(l)


    return kap, maxkap    


def next_partition(kappa, M):
    #print kappa
    #print M

    for i in range(len(kappa)-1,0,-1):
        #print 'i=',i
        if kappa[i] <= M[i-1]:
            #print i,'!'
            
            kappa[i] += 1
            M[i] = numpy.max([M[i],kappa[i]])  # XXX slow
            for j in range(i+1,len(kappa)):
                kappa[j] = kappa[0]
                M[j] = M[i]
            return kappa,M                

    #print 'Finish !'
    return None

def prev_partition(kappa, M):
    for i in range(len(kappa)-1,0,-1):
        if kappa[i] > kappa[0]:
            kappa[i] -= 1
            M[i] = M[i-1]
            for j in range(i+1,len(kappa)):
                kappa[j] = M[i] + j - i
                M[j] = kappa[j]
            return kappa,M

    return None

def decode_partition(set, kappa, M):
    """
    Given the index vector representation of a partition, return the tuple representation
    """

    nr_part = M[-1]
    
    #print nr_part,'different partitions'
    
    part = []
    for i in range(int(nr_part+1)):
        i_ind = numpy.where(kappa == i)[0]
        #print i, i_ind
        
        part.append( tuple(set[i_ind]) )

    return part        

def encode_partition(leaders, groups, G):
    """
    Given a CSI model structure, return the index vector representation of the corresponding partition
    """
    v = [0] * G
    
    for i,l in enumerate(leaders):
        v[l] = i
        for g in groups[l]:
            v[g] = i

    return v


def generate_all_partitions(G,order='forward'):
    """
    Returns a list of all possible partitions for a set of cardinality G.
    
    Warning: for large G output gets huge
    """

    if order == 'forward':
        set = numpy.arange(G+1)
        P = []

        ind = 0
        while 1:
            if ind == 0:
                kappa,M = init_first(G)

            else:
                ret = next_partition(kappa, M)
                if ret == None:
                    break


                kappa,M  = ret


            #if M[-1]+1 == nr:
            #print 'kappa=',kappa,'->' ,decode_partition(set,kappa,M)
            P.append(decode_partition(set,kappa,M))

            ind += 1

        return P     
        
        
    elif order == 'reverse':
        set = numpy.arange(G+1)

        P = []

        ind = 0
        while 1:
            if ind == 0:
                kappa,M = init_last(G)

            else:
                ret = prev_partition(kappa, M)
                if ret == None:
                    break


                kappa,M  = ret


            #if M[-1]+1 == nr:
            #print 'kappa=',kappa,'->' ,decode_partition(set,kappa,M)
            P.append(decode_partition(set,kappa,M))

            ind += 1

        return P     
        
    else:
        raise TypeError           


def get_partitions_w_cardinality(G,R):
    """
    Returns all partitions of a G-element set with cardinality R.
    """
    # XXX inefficent for large G ...
    P_all = generate_all_partitions(G)
    res =[]

    for p in P_all:
        if len(p) == R:
            res.append(p)

    return res    
    

def get_random_partition(G):
    #nr = random.randint(1,G+1)  # number of subgroups
    set = numpy.arange(G+1)
    kap = numpy.zeros(G)
    M = numpy.zeros(G)

    for i in range(G):
        kap[i] = random.randint(0,G)
    
    #print kap    

    # post-process to remove empty subsets
    d = {}
    ind = 0
    for e in kap:
        if not d.has_key(e):
            d[e] = ind
            ind += 1
    
    #print d
    
    for i,e in enumerate(kap):
        kap[i] = d[e]

        if kap[i] > M[i]:
            for i2 in range(i,G):
                M[i2] = kap[i]

    
    #print kap
    #print M
    
    return decode_partition(set,kap,M)


            
