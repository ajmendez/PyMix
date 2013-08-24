################################################################################
#
#       This file is part of the Python Mixture Package
#
#       file:    plotMixture.py
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
PlotMixture implements visualization functions for mixture models and clustering.
"""

import pylab
import mixture
import numpy



def plotData(data,axis= None, title= None, newfigure=True, markersymbol='o'):
    # XXX should work on DataSet XXX 
    
    assert len(data[0]) == 2
    
    if newfigure:
        pylab.figure()

    pylab.plot(data[:,0] ,data[:,1],markersymbol,markerfacecolor='k')

    if axis is not None:
        pylab.axis(axis)  

    if title:
        pylab.title(title)
    else:
        pylab.title('Data Plot') 

def plotConstrainedData(data,axis= None, title= None, newfigure=True):

    plotData(data.internalData,axis= axis, title= title, newfigure=newfigure)

    # plot positive constraints
    for i1 in range(data.N):
        for i2 in range(i1+1):
            if data.pairwisepositive[i1,i2] != 0:
                #print i1,i2
                pylab.plot( [data.dataMatrix[i1][0], data.dataMatrix[i2][0] ],
                                [data.dataMatrix[i1][1], data.dataMatrix[i2][1]],'k-', markersize=10)
    

    # plot negative constraints
    for i1 in range(data.N):
        for i2 in range(i1+1):
            if data.pairwisenegative[i1,i2] != 0:
                #print i1,i2
                pylab.plot( [data.dataMatrix[i1][0], data.dataMatrix[i2][0] ],
                                [data.dataMatrix[i1][1], data.dataMatrix[i2][1]],'k--', markersize=10 )

    if axis is not None:
        pylab.axis(axis)  



def plotDataHistogram(data,axis=None,title=None):
    pylab.hist(data)

    if axis:
        pylab.axis(axis)  

    if title:
        pylab.title(title)
    else:
        pylab.title('Data Histogram') 


def plotClustering(data, labels, axis=None, title = None, markersize=5):
    assert len(data) == len(labels)
    assert len(data[0]) == 2
    
    # XXX colors hard coded here XXX
    color = ['b','r','g','y']    
    symbol = ['o','d','s','^']
    
    # first extract unique labels from labels, we assume labels are consistent with trueLabels (XXX?)
    d = {}
    for l in labels:
        if not d.has_key(l):
            d[l] = 1
    label_alphabet = d.keys()
    #print label_alphabet

    if len(label_alphabet) > 4:
        print "XXX Not enough colors specified to plot clustering. XXX"
        raise RuntimeError
   
   
    cind= []
    for u in label_alphabet:
        cind.append(numpy.where(labels==u)[0])
    
    
    #ind0 = numpy.where(c==0)[0]
    #ind1 = numpy.where(c==1)[0]
    
    #marker='o', markerfacecolor='g', markeredgecolor='r'

    pylab.figure()
    
    #  marker='o', markerfacecolor='g', markeredgecolor='r' 
    for i,c in enumerate(cind):
        pylab.plot(data[c,0] ,data[c,1],symbol[i],markerfacecolor=color[i], markersize=markersize )
        
           # ,data[ind1,0],data[ind1,1],'yo')
    if axis:
        pylab.axis(axis)  

    if title:
        pylab.title(title)
    else:
        pylab.title('Cluster Plot')    

def plotConstrainedClustering(data, labels, axis=None, title = None):
    """
    Plot clustering with pairwise constraints.
    """
    plotClustering(data.internalData, labels, axis=axis, title = title)


    # XXX colors hard coded here XXX
    color = ['b','r','g','y']    
    symbol = ['o','d','.','^']
    
    # plot positive constraints
    for i1 in range(data.N):
        for i2 in range(i1+1):
            if data.pairwisepositive[i1,i2] != 0:
                #print i1,i2
                pylab.plot( [data.dataMatrix[i1][0], data.dataMatrix[i2][0] ],
                                [data.dataMatrix[i1][1], data.dataMatrix[i2][1]],'k-', markersize=12)
                pylab.plot( [data.dataMatrix[i1][0]],[data.dataMatrix[i1][1]],symbol[labels[i1]],markerfacecolor=color[labels[i1]], markersize=7 )
                pylab.plot( [data.dataMatrix[i2][0]],[data.dataMatrix[i2][1]],symbol[labels[i2]],markerfacecolor=color[labels[i2]], markersize=7 )    
    

    # plot negative constraints
    for i1 in range(data.N):
        for i2 in range(i1+1):
            if data.pairwisenegative[i1,i2] != 0:
                #print i1,i2
                pylab.plot( [data.dataMatrix[i1][0], data.dataMatrix[i2][0] ],
                                [data.dataMatrix[i1][1], data.dataMatrix[i2][1]],'k--', markersize=12 )

                pylab.plot( [data.dataMatrix[i1][0]],[data.dataMatrix[i1][1]],symbol[labels[i1]],markerfacecolor=color[labels[i1]], markersize=7 )
                pylab.plot( [data.dataMatrix[i2][0]],[data.dataMatrix[i2][1]],symbol[labels[i2]],markerfacecolor=color[labels[i2]], markersize=7 )    

    if axis:
        pylab.axis(axis)  

    
def plotClusterEval(data, predLabels, trueLabels ,axis=None):
    assert len(data) == len(predLabels) == len(trueLabels), str(len(data))+"!="+ str(len(predLabels))+"!="+str( len(trueLabels))
    assert len(data[0]) == 2
    
    # XXX colors hard coded here XXX
    color = ['b','r','g','m','c','y']    
    
    # first extract unique labels from predLabels, we assume labels are consistent with trueLabels (XXX?)
    d = {}
    for l in predLabels:
        if not d.has_key(l):
            d[l] = 1
    
    
    
    label_alphabet = d.keys()
    
    print label_alphabet

    nr_labels = len(label_alphabet)
    if nr_labels > 4:
        print "XXX Not enough colors specified to plot clustering. XXX"
        raise RuntimeError
   
    # build color dictionary for true labels
    true_color_dict = {}
    for i, l in enumerate(label_alphabet):
        true_color_dict[l] = color[i]

    # compute label equivalencies between true and predicted labels    
    label_equiv = numpy.zeros((nr_labels,nr_labels) )
    indices_dict = {}
    for i,l1 in enumerate(label_alphabet):  # loop over true labels
       indices_dict[l1] = {}
       for j,l2 in enumerate(label_alphabet):  # loop over predicted labels
          ind1 = numpy.where(predLabels==l1)[0]
          ind2 = numpy.where(trueLabels==l2)[0]
          ind_inter = numpy.array(mixture.list_intersection(ind1,ind2) )
    
          indices_dict[l1][l2] = ind_inter
          label_equiv[i][j] = len(ind_inter)
    
    #print label_equiv
#    print indices_dict


    label_map = {}  # label map from pred -> true
    for i,l1 in enumerate(label_alphabet):
        m =  numpy.where( label_equiv[i] == max(label_equiv[i] ) )[0][0]
        label_map[label_alphabet[m]] =  l1
    
    
    #print label_map
    

    # build color dictionary for the predicted labels according to the established label equivalencies
    pred_color_dict = {}
    for i,l1 in enumerate(label_alphabet):
        pred_color_dict[l1] = true_color_dict[ label_map[l1] ]

    #print "true color", true_color_dict
    #print "pred color", pred_color_dict

        

    pylab.figure()
    
    # computing the different TP and FP sets and plotting
    for l1 in label_alphabet:  # loop over true labels
       for l2 in label_alphabet:  # loop over predicted labels
          
          # for the true positives the edge is a thin black, otherwise we use a thick edge in the true class color
          if true_color_dict[l1] == pred_color_dict[l2]:
            edge_color = 'k'
            face_color = pred_color_dict[l2]
            edge_width = 0.5
          else:
            edge_color = pred_color_dict[l2] # XXX
            face_color = true_color_dict[l1] # XXX
            edge_width = 1.0
          
          ind_inter = indices_dict[l1][l2]
          
          if len(ind_inter) > 0:
          
          #print l1,color_dict[l1]
          #print l2,color_dict[l2]
          
              pylab.plot(data[ind_inter,0] ,data[ind_inter,1],'o',markerfacecolor=face_color,markeredgecolor=edge_color, markeredgewidth=edge_width)


    if axis:
        pylab.axis(axis)  
    
    pylab.title('Cluster Evaluation')
    

def plotNormalMixtureDensity(mix, axis, title= None, newfigure=False, fill=True, alpha=1.0):
    """ 
    @param axis: matlab-like axis coordinates: [x_start, x_end, y_start, y_end] 
    
    """
    
    if newfigure == True:
        pylab.figure()
    
    # -5, 10.0, -5.0, 10.0

    x = pylab.arange(axis[0],axis[1]+0.1,0.1)
    y = pylab.arange(axis[2],axis[3]+0.1,0.1)
    
    #print len(x)
    #print len(y)
    
    
    #X,Y = pylab.meshgrid(x,y)
    #z = pylab.exp(-(X*X + Y*Y)) + 0.6*pylab.exp(-((X+1.8)**2 + Y**2))   
    #pylab.contour(x,y,z)

    z = numpy.zeros( (len(y),len(x)),dtype='Float64' )
    for i in range(len(y)):
        ndat = numpy.zeros((len(x),2),dtype='Float64' ) 
        ndat[:,1] = y[i]
        ndat[:,0] = x
        #print numpy.exp(mix.pdf(dat))
        
        dat = mixture.DataSet()
        dat.fromList(ndat)
        dat.internalInit(mix)
        
        # XXX pdf is log valued, we want the true value XXX
        
        #print numpy.exp(mix.pdf(dat)).tolist()

        z[i,:] = numpy.exp(mix.pdf(dat))
        
    
    #print "z", len(z),'x', len(z[0]) ,'=', len(z) * len(z[0])
    
    #print "max",z.max()
    max_val = z.max()

    step = max_val / 40.0

    #step = max_val / 200.0

    #print "step",step

    #pylab.figure(1)
    #pylab.contour(x,y,z)
    
    if fill == True:
        pylab.contourf(x,y,z,pylab.arange(0,max_val,step),alpha=alpha) 
    else:
        pylab.contour(x,y,z,pylab.arange(0,max_val,step),alpha=alpha)

    if title:
        pylab.title(title)
    else:
        pylab.title('Normal Mixture Density Plot')
    
def plotUnivariateNormalMixtureDensity(m, axis, title= None, format= '-b'):
    """ 
    
    """
    
    # -5, 10.0, -5.0, 10.0

    x = pylab.arange(axis[0],axis[1],0.02)
    dat_x = mixture.DataSet()
    dat_x.fromList(x)
    dat_x.internalInit(m)
    
    #print len(x)
    #print len(y)
    

    y = numpy.exp( m.pdf(dat_x) )


    #pylab.figure()
    pylab.plot(x,y, format) #
    pylab.axis(axis)  

    
    if title:
        pylab.title(title)
    else:
        pylab.title('Normal Mixture Density Plot')
    
    
def plotMixtureEntropy(mix, axis):
    """ 
    @param axis: matlab-like axis coordinates: [x_start, x_end, y_start, y_end] 
    
    """
    
    # -5, 10.0, -5.0, 10.0

    x = pylab.arange(axis[0],axis[1]+0.1,0.1)
    y = pylab.arange(axis[2],axis[3]+0.1,0.1)
    
    #print x
    #print y
    
    #print len(x)
    #print len(y)
    
    
    #X,Y = pylab.meshgrid(x,y)
    #z = pylab.exp(-(X*X + Y*Y)) + 0.6*pylab.exp(-((X+1.8)**2 + Y**2))   
    #pylab.contour(x,y,z)

    z = numpy.zeros( (len(y),len(x)),dtype='Float64' )
    for i in range(len(y)):
        dat = numpy.zeros((len(x),2),dtype='Float64' ) 
        dat[:,1] = y[i]
        dat[:,0] = x
        #print numpy.exp(mix.pdf(dat))


        #print "---------------------------\n",dat
        data = mixture.DataSet()
        data.fromList(dat)
        data.internalInit(mix)
        
        l = mixture.get_posterior(mix,data,logreturn=False)

        #print l

        
        #print numpy.exp(mix.pdf(dat)).tolist()
        for j in range(len(x)):
            
            z[i,j] = mixture.entropy(l[:,j])
            #print dat[j,:] ,":",l[:,j], "=",z[i,j]
        #print "---------------------------\n"        
    
    #print "z", len(z),'x', len(z[0]) ,'=', len(z) * len(z[0])
    
    print "max",z.max()
    #max_val = z.max()
    
    max_val = numpy.log(mix.G) # maximum entropy for a vector of length mix.G
    print "theor. max", max_val
    
    step = max_val / 10.0
    print "step",step

    #pylab.figure(1)
    #pylab.contour(x,y,z)
    
    #pylab.figure(2)
    #pylab.contour(x,y,z,pylab.arange(0,max_val,step))
    #pylab.legend()  

#    pylab.colorbar()
    pylab.contourf(x,y,z,) # pylab.arange(0,max_val,step)
    
    
    
    pylab.title('Posterior Entropy Plot')
        
    
    
def plotPosteriorMax(mix, axis):
    """ 
    @param axis: matlab-like axis coordinates: [x_start, x_end, y_start, y_end] 
    
    """
    
    # -5, 10.0, -5.0, 10.0

    x = pylab.arange(axis[0],axis[1],0.1)
    y = pylab.arange(axis[2],axis[3],0.1)
    
    #print len(x)
    #print len(y)

    # XXX colors hard coded here XXX
    color =  ['b','r','g','m','c','y']  
    assert mix.G <= len(color)
    
    z = numpy.zeros( (len(y),len(x)),dtype='Float64' )
    for i in range(len(y)):
        dat = numpy.zeros((len(x),2),dtype='Float64' ) 
        dat[:,1] = y[i]
        dat[:,0] = x
        #print numpy.exp(mix.pdf(dat))


        #print "---------------------------\n",dat
        l = mixture.get_posterior(mix,dat)

        #print l

        # XXX pdf is log valued, we want the true value XXX
        
        #print numpy.exp(mix.pdf(dat)).tolist()
        for j in range(len(x)):
            
            z[i,j] = numpy.argmax(l[:,j])
            #print dat[j,:] ,":",l[:,j],numpy.argmax(l[:,j])
            
            
            #print dat[j,:] ,":",l[:,j], "=",z[i,j]
        #print "---------------------------\n"        
    
    #print "z", len(z),'x', len(z[0]) ,'=', len(z) * len(z[0])
    
    print "max",z.max()
    #max_val = z.max()
    
    max_val = numpy.log(mix.G) # maximum entropy for a vector of length mix.G
    print "theor. max", max_val
    
    step = max_val / 40.0
    print "step",step

    #pylab.figure(1)
    #pylab.contour(x,y,z)
    
    #pylab.figure(2)
    #pylab.contour(x,y,z,pylab.arange(0,max_val,step))
    #pylab.legend()  

    pylab.figure()
#    pylab.colorbar()
    

    pylab.contourf(x,y,z,cmap=pylab.cm.hsv) # pylab.arange(0,max_val,step)
    
    
    
    pylab.title('Posterior Maximum Plot')
        
def plotMixtureStructure(mix,headers,transpose=1):
    plot = numpy.zeros(( mix.G,mix.dist_nr ) )

    for i in range(mix.dist_nr):
        #print "-------------\n",headers[i]
        #print mix.leaders[i]
        #print mix.groups[i]
        
        
        # check for noise variables
        if len(mix.leaders[i]) == 1:
            l = mix.leaders[i][0]
            for g in range(mix.G):
               plot[g,i] = 1
        
        else:
            for l in mix.leaders[i]:    
                
                #if len(mix.groups[i][l]) == 0:
               #     plot[l,i] = 2
               # else:
               plot[l,i] = l+3
               for g in mix.groups[i][l]:
                   plot[g,i] = l+3
  
    for j in range(mix.dist_nr):
        t = {}
        t[2] = 2
        t[1] = 1
            
        index = 3
        v_list = []
        for v in plot[:,j]:
            if v == 2:
                continue
            elif v == 1:
                break
            else:        
                if v not in v_list:
                    v_list.append(v)
    
        v_list.sort()
        #print plot[:,j]
        #print v_list
        for v in v_list:
            t[v] = index
            index +=1
        
        for i in range(mix.G):
            plot[i,j] = t[plot[i,j]]
                    
    
    #for p in plot:
    #    print p.tolist()
    #print plot
    
    #pylab.subplot(1,2,1)
    #x = numpy.array([0.0,5.0],dtype='Float64')
    #y = numpy.array([0.0,5.0],dtype='Float64')
    #pylab.plot(x,y,'o')
    #pylab.subplot(1,2,2)
    
    
    z = numpy.array(plot)
    if transpose:
        
        #print z.shape
        z = z.transpose()
        #print z.shape
        
        mat = pylab.matshow(z)
        
#        pylab.setp(mat)
#        print mat.axes
#        print mat.figure
        
        
        
        
        pylab.grid(True,linestyle='-',linewidth=0.5)
        
        # set xticks
        #pylab.xticks( numpy.arange(len(headers))+0.5,['']*len(headers), size=12)
        xtickpos = numpy.arange(0.0, mix.G+1,0.5)
        temp = zip(range(mix.G), ['']*mix.G)
        xticklabels = []
        for tt in temp:
            xticklabels += list(tt) 
        pylab.xticks( xtickpos,xticklabels) # rotation='vertical',size=12

        #print xtickpos
        #print xticklabels

        # remove grid lines for tick lines at label positions
        xgridlines = pylab.getp(pylab.gca(), 'xgridlines')
        #print  xgridlines
        
        for j in range(len(xticklabels)):
            if xticklabels[j] != '':
                xgridlines[j].set_linestyle('None')


        # set yticks
        ytickpos = numpy.arange(0.0, len(headers),0.5)
        temp = zip(headers, ['']*len(headers))
        
        yticklabels = []
        for tt in temp:
            yticklabels += list(tt) 
        
       
        pylab.yticks( ytickpos, yticklabels, size=8)

        # remove grid lines for tick lines at label positions
        ygridlines = pylab.getp(pylab.gca(), 'ygridlines')
        
        #print len(ygridlines), len(ytickpos), len(yticklabels)
        
        for j,yl in enumerate(ygridlines):
            if yticklabels[j] != '':
                yl.set_linestyle('None')
            
        
        #pylab.setp(ygridlines, 'linestyle', 'None')
        
        #pylab.yticks( numpy.arange(len(headers))+0.5)


#        loc, ll = pylab.yticks()
#        for lll in ll:
#            print lll,lll.get_dashlength()
#            #lll.set_dashlength(-1)



        #plotMixture.pylab.figtext(0.15,0.5,'T1',size=20, weight='bold')
        #plotNormalMixtureDensitture.pylab.figtext(0.925,0.5,'T1',size=20, weight='bold')

        #start = 0.15
#        start = 0.18
#        end = 0.925
#
#        space = end - start
#        step = space / mix.dist_nr
        #for i in range(0,mix.dist_nr):
        # XXX HACK for two components
        #pylab.figtext(0.12,0.63,'$C_1$',size=20, weight='bold')
        #pylab.figtext(0.12,0.26,'$C_2$',size=20, weight='bold')





    else:
        fig = pylab.matshow(z)
        pylab.grid(True,linestyle='-',linewidth=0.5)
        #pylab.yticks(numpy.arange(mix.G+1) )
        #pylab.xticks(  numpy.arange(len(plot[0])+1)+0.5,headers,rotation=90,size=12)
        #fig.set_size_inches(20,20)    

        # set xticks
        #pylab.xticks( numpy.arange(len(headers))+0.5,['']*len(headers), size=12)

        xtickpos = numpy.arange(0.0, len(headers),0.5)
        temp = zip(headers, ['']*len(headers))

        xticklabels = []
        for tt in temp:
            xticklabels += list(tt) 
        pylab.xticks( xtickpos,xticklabels) # rotation='vertical',size=12

        print len(xtickpos), xtickpos.tolist()
        print len(xticklabels), xticklabels

        # remove grid lines for tick lines at label positions
        xgridlines = pylab.getp(pylab.gca(), 'xgridlines')
        print  xgridlines
        
        for j in range(len(xticklabels)):
            if xticklabels[j] != '':
                xgridlines[j].set_linestyle('None')


        # set yticks
        ytickpos = numpy.arange(0.0, mix.G,0.5)
        temp = zip(range(mix.G), ['']*mix.G)
        
        yticklabels = []
        for tt in temp:
            yticklabels += list(tt) 
     
        pylab.yticks( ytickpos, yticklabels, size=12)

        print len(ytickpos), ytickpos.tolist()
        print len(yticklabels), yticklabels

        # remove grid lines for tick lines at label positions
        ygridlines = pylab.getp(pylab.gca(), 'ygridlines')
        
        #print len(ygridlines), len(ytickpos), len(yticklabels)
        
        for j,yl in enumerate(ygridlines):
            if yticklabels[j] != '':
                yl.set_linestyle('None')
    


        #labels = fig.get_xticklabels()
        #pylab.setp(labels,'rotation', 90 , fontsize =12)

    #pylab.show()
    
    
def plotFeatureRanks(model, comp_list, data, axis, title= None):
    
    raise NotImplementedError
    ranks = model.KLFeatureRanks(data, comp_list)
    print ranks
    # XXX 

#----------------------------------------------------------------------------------------

# Dirichlet distribution plots
# Based on code adapted from the CMPy - Computational Mechanics Python package
# https://svn.cse.ucdavis.edu/trac/cmpy/wiki

def Property( fcn ):
    return property( **fcn() )

def exp2(x):
    return 2 ** x

class Simplex2DPlotter(object):
    """This class helps plot a 2D-Simplex.  It is backend independent, and
    one must provide an axis when calling plot commands."""

    def __init__(self, simplex, axes):
        self.simplex = simplex
        self.axes = axes
        self.primary_plots = []
        
    def plot_simplex(self):
        self.axes.clear()
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.plot_outer_edges()
        #self.plot_inner_edges()
        if self.simplex.dimension <= 16:
            self.plot_labels()

    def prepare_axes(self):
        if self.simplex.dimension <= 4:
            self.axes.set_xlim([-1.05,1.05])
            self.axes.set_ylim([-1.05,1.05])            
        else:
            self.axes.set_xlim([-1.45,1.45])
            self.axes.set_ylim([-1.45,1.45])       

        self.axes.set_frame_on(False)

    def plot_outer_edges(self):
        x, y = self.simplex.construct_outer_edges()
        p, = self.axes.plot(x, y, 'b-')
        self.primary_plots.append(p)

    def plot_inner_edges(self):
        for x, y in self.simplex.construct_inner_edges():
            p, = self.axes.plot(x, y, 'k-')
            self.primary_plots.append(p)            

    def plot_labels(self):
        for x, y, label in self.simplex.labels:
            self.axes.text(x, y, r"%s" % label, 
                           fontsize=10, 
                           horizontalalignment='center',
                           verticalalignment='center')
        
    def clear_other_plots(self):
        for plot in copy(self.axes.lines):
            if plot not in self.primary_plots:
                self.axes.lines.remove(plot)
        
    

class Simplex2D(object):
    """This class serves as a 2-dimensional, abstract representation of an 
    n-dimensional simplex.  It contains no plotting routines, storing only the 
    information that would be helpful for plotting.
    
    >>> a = Simplex2D(keys=[('0','0'), ('0','1'), ('1','0'), ('1','1')])
    >>> a.project_distribution({('1','0'):.3, ('0','1'):.7}, use_logs=False)
    array([ 0.28284271, -0.70710678])

    """
    
    def __init__(self, keys=[], **keywords):
        self.modify_labels = keywords.pop("modify_labels", True)
        self.tex_labels = keywords.pop("tex_labels", True)
        self.sort = keywords.pop("sort", True)
        self.invert = keywords.pop("invert", False)
        self.distribution_keys = keys
        
    def construct_all(self):
        self.construct_vertices()
        self.construct_outer_edges()
        self.construct_inner_edges()
        self.construct_labels()
    
    def construct_vertices(self):
        """Constructs the vertices of the simplex."""
        vertices = range(self.dimension)
        if not self.invert:
            vertices = numpy.pi/2 - 2 * numpy.pi / self.dimension * (numpy.array(vertices) + 1.0/2)
        else:
            vertices = 3*numpy.pi/2 - 2 * numpy.pi / self.dimension * (numpy.array(vertices) + 1.0/2)
        self.vertices_x = numpy.cos(vertices)
        self.vertices_y = numpy.sin(vertices)
        # This is a 2-by-n matrix.
        self.projection_matrix = numpy.array([self.vertices_x, self.vertices_y])
        return self.vertices_x, self.vertices_y
    
    def construct_outer_edges(self):
        """Constructs the lines connecting the verticies of the simplex."""
        x = list(self.vertices_x)
        x.append(x[0])
        y = list(self.vertices_y)
        y.append(y[0])
        return x,y

    def construct_inner_edges(self):
        """Constructs the lines (necessary to make the simplex a complete
        graph) that appear on the interior of the simplex. The output is a
        list of tuples representing the x- and y-coordinates of the points
        necessary to make up each edge."""
        self.inner_edges = []
        for i in range(self.dimension):
            for j in range(i+1, self.dimension):
                if j==i+1 or (i==0 and j==self.dimension-1):
                    continue
                else:
                    self.inner_edges.append(([self.vertices_x[i], self.vertices_x[j]],
                                             [self.vertices_y[i], self.vertices_y[j]]))
        return self.inner_edges

    def construct_labels(self):
        self.labels = []
        for x, y, label in zip(self.vertices_x, self.vertices_y, self.distribution_keys):
            # The labels are only useful when the dimension of the simplex
            # is low. For this reason, I will assume that the alphabet has
            # less than 10 characters.
            if self.modify_labels:
                label = ''.join(label)
                if self.tex_labels:
                    #label = "\Prob($%s$)" % label
                    label = "P($%s$)" % label
                else:
                    label = "P(%s)" % label
               
            # Where to we place the points?
            # Let's stretch the vector a bit.
            point = numpy.array([x,y]) * 1.1  # 1.2
            self.labels.append((point[0], point[1], label))

        return self.labels
            
    def project_distribution(self, distribution, use_logs=True):
        """This function takes a distribution and projects it to a point (x,y)
        in the plane. The distribution is not validated before projecting."""
        if use_logs:
            values = [exp2(distribution.get(key, -inf)) for key in self.distribution_keys]
        else:
            values = [distribution.get(key, 0) for key in self.distribution_keys]
 
            #print '\n** input=',distribution
            #print '** values=',values
          
        #print 'proj mat=',self.projection_matrix.tolist()
            
        point = numpy.dot(self.projection_matrix, values)
        return point


    @Property
    def distribution_keys():
        doc = """A list representing the dimensionality of the simplex."""
        def fget(self):
            return self.__distribution_keys
        def fset(self, keys):
            self.__distribution_keys = keys
            if self.sort:
                self.__distribution_keys.sort()
            self.dimension = len(self.__distribution_keys)
            # Any time the dimensionality changes, we need to reconstruct.
            self.construct_all()
        fdel = None
        return locals()
    
    

def plotDirichletDensity(dirichlet_dist,title='DirichletDensity'):
    
    assert dirichlet_dist.M == 3, 'Only 3 dimensions for now.'
    
    # A 2-simplex lives in 3-space.
    dimension = 3 # XXX dimension fixed to 3 for now

    # These are the vertex labels, converted to strings.
    labels = numpy.eye(dimension, dtype=int)
    labels = map(str, map(tuple, labels))

    # Let's create the simplex.
    simplex = Simplex2D(labels, modify_labels=False)

    # construct grid
    dist = []
    x = numpy.arange(0.001,1.0,0.01)
    y = numpy.arange(0.001,1.0,0.01)
    for p1 in x:
        d_row = []
        for p2 in y:
            #for p3 in z:
            p3 = 1.0-p1-p2
            d_row.append([p1,p2,p3])
        dist.append(d_row)

    proj_x = []
    proj_y = []
    density = []
    f = lambda x: round(x,3)
    for drow in dist:
        x_row = []
        y_row = []
        d_row = []
        for d in drow:
            if (1.0 - numpy.sum(map(f,d))) < 1e-15 and d[0] > 0 and d[1] > 0 and d[2] > 0.0:
                d_row.append( numpy.exp( dirichlet_dist.pdf(mixture.DiscreteDistribution(3, d)) ))
            else:
                d_row.append( 0.0)
            
            sample_dist = dict(zip(labels, d))  
            pp = simplex.project_distribution(sample_dist, use_logs=False)
            x_row.append(pp[0])
            y_row.append(pp[1])

        proj_x.append(x_row)
        proj_y.append(y_row)
        density.append(d_row)

    proj_x = numpy.array(proj_x)
    proj_y = numpy.array(proj_y)
    density = numpy.array(density)


    # Create the figure
    fig = pylab.figure()
    fig.set_facecolor('w')
    fig.add_axes([.15,.15,.70,.70], axisbg='w', aspect='equal')
    axis = pylab.gca()
    
    # Plot the simplex
    simplex_plotter = Simplex2DPlotter(simplex, axis)
    simplex_plotter.prepare_axes()
    simplex_plotter.plot_simplex()
    axis.set_title(title)
    
    # Plot the samples
    #x = [sample[0] for sample in samples]
    #y = [sample[1] for sample in samples]

    max_val = density.max()
    step = max_val / 60.0
    
    axis.contourf(proj_x, proj_y,density,pylab.arange(0,max_val,step),norm = pylab.matplotlib.colors.Normalize(proj_x) ) 
    #pylab.show()
    
def plotKLDistance(ref_dist, objf='sym' ,title='KL Distance', show=True):
    
    assert ref_dist.M == 3, 'Only 3 dimensions for now.'
    
    # KL distance to be used, either symmetric or the two directions
    # with respect to ref_dist
    assert objf in ['sym', 'leftToRight', 'rightToLeft']  
    
    # A 2-simplex lives in 3-space.
    dimension = 3 # XXX dimension fixed to 3 for now

    # These are the vertex labels, converted to strings.
    labels = numpy.eye(dimension, dtype=int)
    labels = map(str, map(tuple, labels))

    # Let's create the simplex.
    simplex = Simplex2D(labels, modify_labels=False)

    # construct grid
    dist = []
    x = numpy.arange(0.001,1.0,0.01)
    y = numpy.arange(0.001,1.0,0.01)
    for p1 in x:
        d_row = []
        for p2 in y:
            #for p3 in z:
            p3 = 1.0-p1-p2
            d_row.append([p1,p2,p3])
        dist.append(d_row)
    
    sample_dist = dict(zip(labels, ref_dist.phi))
    proj_ref = simplex.project_distribution(sample_dist, use_logs=False)
    
    
    proj_x = []
    proj_y = []
    distance = []
    f = lambda x: round(x,3)
    for drow in dist:
        x_row = []
        y_row = []
        d_row = []
        for d in drow:
            #print d, 1.0 - numpy.sum(map(f,d))
            
            
            if (1.0 - numpy.sum(map(f,d))) < 1e-15 and d[0] > 0 and d[1] > 0 and d[2] > 0.0:
                #print ref_dist,mixture.DiscreteDistribution(3, d), mixture.sym_kl_dist(  ref_dist, mixture.DiscreteDistribution(3, d))
                if objf == 'sym':
                    d_row.append( mixture.sym_kl_dist(  ref_dist, mixture.DiscreteDistribution(3, d)))
                elif objf == 'leftToRight':    
                    d_row.append( mixture.kl_dist(  ref_dist, mixture.DiscreteDistribution(3, d)))
                elif objf == 'rightToLeft':    
                    d_row.append( mixture.kl_dist( mixture.DiscreteDistribution(3, d), ref_dist ))                
                else:
                    raise TypeError
                    
            else:
                d_row.append( 0.0)
            
            sample_dist = dict(zip(labels, d))  
            pp = simplex.project_distribution(sample_dist, use_logs=False)
            x_row.append(pp[0])
            y_row.append(pp[1])

        proj_x.append(x_row)
        proj_y.append(y_row)
        distance.append(d_row)

    proj_x = numpy.array(proj_x)
    proj_y = numpy.array(proj_y)
    distance = numpy.array(distance)

    # Create the figure
    fig = pylab.figure()
    fig.set_facecolor('w')
    fig.add_axes([.15,.15,.70,.70], axisbg='w', aspect='equal')
    axis = pylab.gca()
    
    # Plot the simplex
    simplex_plotter = Simplex2DPlotter(simplex, axis)
    simplex_plotter.prepare_axes()
    simplex_plotter.plot_simplex()
    
    #axis.set_title(title)
    axis.text(-0.5, 0.55, title, fontsize=12)
    
    # Plot the samples
    #x = [sample[0] for sample in samples]
    #y = [sample[1] for sample in samples]

    max_val = distance.max()
    step = max_val / 50.0
    
    axis.contourf(proj_x, proj_y,distance,pylab.arange(0,max_val,step)) 
    axis.plot([proj_ref[0]], [proj_ref[1]], 'or') 

    if show:
        pylab.show()
     
    
    
    
    
    
    
    
