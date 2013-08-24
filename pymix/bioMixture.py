################################################################################
#
#       This file is part of the Python Mixture Package
#
#       file:    mixPWM.py
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
This file contains auxiliary functions for the analysis of biological sequences.

e.g. searching for transcription factor binding sites using mixtures of PWMs (positional weight matrices).
"""

import mixture
import numarray
import re


# mapping from amino acid one to three letter code
AAlong = {      'A' : 'Ala', 
                'R' : 'Arg', 
                'N' : 'Asn', 
                'D' : 'Asp', 
                'C' : 'Cys', 
                'E' : 'Glu',   
                'Q' : 'Gln',   
                'G' : 'Gly', 
                'H' : 'His', 
                'I' : 'Ile', 
                'L' : 'Leu', 
                'K' : 'Lys', 
                'M' : 'Met', 
                'F' : 'Phe', 
                'P' : 'Pro', 	
                'S' : 'Ser', 
                'T' : 'Thr', 
                'W' : 'Trp', 
                'Y' : 'Tyr', 
                'V' : 'Val'  }

def readJASPAR(fileName):
    """
    Reads a flat file of JASPAR binding sites matrices. JASPAR files are
    essentially fasta, but only upper case letters are part of the binding site proper.
    Lower case letters are discarded.
    
    """
    f = open(fileName,"r")

    seq_head = re.compile("^\>(.*)")
    end = re.compile("^[A,C,G,T,a,c,g,t]\s*\[")
    seq = []

    ids = []
    count = 1
    for line in f:
        line = mixture.chomp(line)
        #line = line.upper()
        s = seq_head.search(line) 
        if s:
            ids.append('seq'+str(count))  
            count +=1
            #print s.group(1)

        elif end.search(line):
            break
        else:
            if len(line) > 0:
                line = list(line)
                               
                # remove lower case letters, only upper case letters are part of the 
                # binding site
                site = []
                for i,s in enumerate(line):
                    if s.isupper():
                        site.append(s)

                seq.append(site)     
                #print len(site)
            
    data = mixture.DataSet()          
    data.fromList(seq, IDs = ids)
      
    return data



def readFastaSequences(fileName, out_type='DataSet'):
    """
    Reads a file in fasta format and returns the sequence in a DataSet object
    
    @param fileName: Name of the input file
    @param: type of output object: DataSet or ConstrainedDataSet 
    @return: list of sequence lists
    """
    f = open(fileName,"r")
    index = -1
    seqM = []
    nameList = []
    partSeq = ""
    nameReg = re.compile("^\>(.*)")
    
    try:
        while 1==1:
            line = f.next() 
            s = nameReg.search(line)
            if s:
                if index != -1:
                    if partSeq[len(partSeq)-2:len(partSeq)] == "//":
                        partSeq = partSeq[0:len(partSeq)-2]

                    partSeq = partSeq.upper() # upper case letters by convention
                    seqM.append(list(partSeq))
                partSeq = ""
                index +=1
                nameList.append(mixture.chomp(s.group(1) ))
            else:
                partSeq += mixture.chomp(line)
    
    except StopIteration:
        if partSeq[len(partSeq)-2:len(partSeq)] == "//":
            partSeq = partSeq[0:len(partSeq)-2]
        partSeq = partSeq.upper()
        seqM.append(list(partSeq))

    if out_type == 'DataSet':
        data = mixture.DataSet()
    elif out_type == 'ConstrainedDataSet':
        data = mixture.ConstrainedDataSet()    
    else:
        raise TypeError, 'Invalid output type ' + str(out_type)
        
    data.fromList(seqM,IDs=nameList)
    
    return data

def readSites(fileName):
    """
    Flat file parser for the JASPAR .sites  format. The files are essentially fasta but
    there is a count matrix at the end of the file.
    
    @param fileName: File name of .sites file
    @return: DataSet object
    """
    f = open(fileName,"r")

    seq_head = re.compile("^\>(.*)")
    end = re.compile("^[A,C,G,T,a,c,g,t]\s*\[")
    seq = []

    ids = []
    count = 1
    for line in f:
        line = mixture.chomp(line)
        #line = line.upper()
        s = seq_head.search(line) 
        if s:
            #print s.groups(1)[0]
            tl = s.groups(1)[0].split('\t')
            
            ids.append(str(tl[1])+'_'+str(tl[2])) 
    
            
            #ids.append('seq'+str(count))  
            #count +=1
            #print s.group(1)

        elif end.search(line):
            break
        else:
            if len(line) > 0:
                line = list(line)
                               
                # remove lower case letters, only upper case letters are part of the 
                # binding site
                site = []
                for i,s in enumerate(line):
                    if s.isupper():
                        site.append(s)

                seq.append(site)     
                #print len(site)
            
    data = mixture.DataSet()          
    data.fromList(seq, IDs = ids)
      
    return data

def readAlnData(fn,reg_str=None, out_type='DataSet'):
    """
    Parses a CLUSTALW format .aln multiple alignment file and returns a mixture.DataSet object.
    
    @param reg_str: regular expression for sequence parsing
    @param: type of output object: DataSet or ConstrainedDataSet 
    @return: DataSet object
    """

    f = open(fn,'r') 
    if reg_str:
        parse = re.compile(reg_str)
    else:
        parse = re.compile("(\w+\|\w+)\s+([\w,\-,.]+)")    

    d = {}
    f.readline()  # remove first line

    for l in f:
        l = mixture.chomp(l)
        pat = parse.search(l)
        if pat:
            k =  pat.group(1)
            seq = pat.group(2)
            if k in d.keys():
                d[k] += seq
            else:
                d[k] = seq

        else:
            continue

    if out_type == 'DataSet':
        data = mixture.DataSet()
    elif out_type == 'ConstrainedDataSet':
        data = mixture.ConstrainedDataSet()    
    else:
        raise TypeError, 'Invalid output type ' + str(out_type)
    
    sIDs = d.keys()
    dMatrix = []
    for z in d.keys():
        dMatrix.append(list(d[z]))
    
    data.fromList(dMatrix,IDs = sIDs)
    
    return data


def getModel(G,p):
    """
    Constructs a PWM MixtureModel.
    
    @param G: number of components
    @param p: number of positions of the binding site
    @return: MixtureModel object
    """
    DNA = mixture.Alphabet(['A','C','G','T'])
    comps = []
    for i in range(G):
        dlist = []
        for j in range(p):
           phi = mixture.random_vector(4)
           dlist.append( mixture.DiscreteDistribution(4,phi,DNA))
        comps.append(mixture.ProductDistribution(dlist))
    pi = mixture.random_vector(G)
    m = mixture.MixtureModel(G,pi, comps)
    return m

def getBayesModel(G,p, mixPrior = None):
    """
    Constructs a PWM CSI BayesMixtureModel.
    
    @param G: number of components
    @param p: number of positions of the binding site
    @return: BayesMixtureModel object
    """

    if not mixPrior:
        piPrior = mixture.DirichletPrior(G,[1.0]*G)
        compPrior= []
        for i in range(p):
            compPrior.append( mixture.DirichletPrior(4,[1.02,1.02,1.02,1.02]) )

        # arbitrary values of struct and comp parameters. Values should be
        # reset by user using the structPriorHeuristic method.
        mixPrior = mixture.MixtureModelPrior(0.05,0.05,piPrior, compPrior)
    
    DNA = mixture.Alphabet(['A','C','G','T'])
    comps = []
    for i in range(G):
        dlist = []
        for j in range(p):
           phi = mixture.random_vector(4)
           dlist.append( mixture.DiscreteDistribution(4,phi,DNA))
        comps.append(mixture.ProductDistribution(dlist))
    pi = mixture.random_vector(G)
    m = mixture.BayesMixtureModel(G,pi, comps, mixPrior, struct =1)
    return m


def getBackgroundModel(p, dist = None):
    """
    Construct background model
    
    @param p: number of positions of the binding site
    @param dist: background nucleotide frequencies, uniform is default
    
    @return: MixtureModel representing the background
    """
    DNA = mixture.Alphabet(['A','C','G','T'])
    dlist = []
    
    if dist == None:
        phi = [0.25] * 4
    else:
        phi = dist
    
    for j in range(p):
        dlist.append( mixture.DiscreteDistribution(4,phi,DNA))
    comps = [ mixture.ProductDistribution(dlist) ]

    m = mixture.MixtureModel(1,[1.0], comps)
    return m



def scanSequence(mix, bg, seq,scoring='mix'):
    """
    Scores all positions of a sequence with the given model and background.
    
    @param mix: MixtureModel object
    @param bg: background MixtureModel object
    @param seq: sequence as list of nucleotides
    @param scoring: flag to determine the scoring scheme used for the mixtures. 
      'compmax' means maximum density over the components, 'mix' means true mixture density
    
    @return: list of position-wise log-odd scores
    """
    # convert sequence to internal representation, alphabet of seq must be DNA
    alph = mixture.Alphabet(['A','C','G','T'])
    f = lambda x: alph.internal(x)
    seq=map(f,seq)
    
    dnr = mix.components[0].dist_nr

    # init with dummy value at first position
    s = numarray.array([[-1]+ seq[0:dnr-1]])
    
    
    score = []
    for i in range(dnr-1,len(seq),1):
        # shift query sequence by one position
        s[0] = numarray.concatenate( [s[0][1:],numarray.array([seq[i]])],0)

        if scoring == 'compmax':
            # score as maximum over components 
            c_m_l = numarray.zeros(mix.G,numarray.Float)
            for i in range(mix.G):
                c_m_l[i] = mix.components[i].pdf(s)[0]
            m_l = c_m_l.max()

        elif scoring == 'mix':
            m_l =   mix.pdf(s)[0]          
            
        bg_l = bg.pdf(s)[0]


        score.append(m_l-bg_l)

    return score

