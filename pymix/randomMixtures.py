import mixture
import numpy
import random
import copy
import time
import fullEnumerationExhaustive
import gc
import StructureLearningVariants
import setPartitions

def updateHyperparameters(model, data, delta):
    for j,p in enumerate(model.prior.compPrior):
        if isinstance(p, mixture.NormalGammaPrior):
            data_j = data.getInternalFeature(j)
            p.setParams(data_j,model.G)

    model.prior.structPriorHeuristic(delta, data.N)        



def product_distribution_sym_kl_dist(p1,p2):
    """
    Returns the symmetric KL distance of two ProductDistribution objects, defined as sum of the
    the component-wise KL distances
    """
    d = 0.0
    for j in range(p1.dist_nr):
        d += mixture.sym_kl_dist(p1[j],p2[j])
    return d

def get_random_pi(G,min_val):
    """
    Rejection sampling of pi vectors where all elements are larger than min_val
    """
    
    inval = True
    while inval:
    
        p = numpy.zeros(G)

        for i in range(G):
            p[i] = random.random()

       
        p = p / numpy.sum(p)
        
        if numpy.alltrue(p > min_val):
            inval = False 
            #print p
        
    
    return p
    
    


def getRandomMixture(G, p, KL_lower, KL_upper, dtypes='discgauss', M=4,seed = None):
    
#    if seed:
#        random.seed(seed)
#        mixture._C_mixextend.set_gsl_rng_seed(seed)
#        #print '*** seed=',seed
#        
#    else: # XXX debug
#        seed = random.randint(1,9000000)
#        mixture._C_mixextend.set_gsl_rng_seed(seed)
#        random.seed(seed)
#        #print '*** seed=',seed
            
    
    #M = 4  # Alphabet size for discrete distributions
    
    min_sigma = 0.1    # minimal std for Normal
    max_sigma = 1.0   # maximal std for Normal
    min_mu = -5.0      # minimal mean
    max_mu = 8.0       # maximal mean
    
    if dtypes == 'disc':
        featureTypes = [0] * p
    elif dtypes == 'gauss':
        featureTypes = [1] * p    
    elif dtypes == 'discgauss':    
        # discrete or Normal features for now, chosen uniformly
        # 0 discrete, 1 Normal
        featureTypes = [ random.choice( (0, 1) )  for i in range(p) ]
    else:
        raise TypeError
    
        
    #print featureTypes

    C = []
    for j in range(p):
        c_j = []
        for i in range(G):
            #print i,j
            if featureTypes[j] == 0:
                acc = 0
                while acc == 0:
                    cand = mixture.DiscreteDistribution(M, mixture.random_vector(M) )
                    
                    #print 'cand:',cand
                    
                    acc = 1
                    
                    for d in c_j:
                        KL_dist = mixture.sym_kl_dist(d,cand)
                        if KL_dist > KL_upper or KL_dist < KL_lower:
                            #print '  *', cand, 'rejected:', d , KL_dist
                            acc = 0
                            break
                
                c_j.append(cand)
            elif featureTypes[j] == 1:
                acc = 0
                while acc == 0:
                    mu = random.uniform(min_mu, max_mu)
                    sigma = random.uniform(min_sigma, max_sigma)
                    
                    cand = mixture.NormalDistribution(mu, sigma )
                    
                    #print 'cand:',cand
                    
                    acc = 1
                    
                    for d in c_j:
                        KL_dist = mixture.sym_kl_dist(d,cand)
                        if KL_dist > KL_upper or KL_dist < KL_lower:
                            #print '  *', cand, 'rejected:', d , KL_dist
                            acc = 0
                
                c_j.append(cand)

            else:
                RuntimeError
                
        C.append(c_j)                

#    print '\n'
#    for cc in C:
#        print cc
    
                
    comps = []
    for i in range(G):
        comps.append( mixture.ProductDistribution( [ C[j][i] for j in range(p) ] ) )

    pi = get_random_pi(G,0.1)

    m = mixture.MixtureModel(G,pi, comps,struct=1)            
    m.updateFreeParams()

    return m                
    
    
def getRandomCSIMixture(G, p, KL_lower, KL_upper, M=8, dtypes='discgauss', seed = None, fullstruct=False, disc_sampling_dist=None):
    
#    if seed:
#        random.seed(seed)
#        mixture._C_mixextend.set_gsl_rng_seed(seed)
#        #print '*** seed=',seed
#        
#    else: # XXX debug
#        seed = random.randint(1,9999999)
#        mixture._C_mixextend.set_gsl_rng_seed(seed)
#        random.seed(seed)
#        #print '*** seed=',seed

    

    if disc_sampling_dist == None:
        discSamp = mixture.DirichletPrior(M,[1.0] * M ) # uniform sampling
    else:
        discSamp = disc_sampling_dist   
        

    
    min_sigma = 0.3    # minimal std for Normal
    max_sigma = 5.0   # maximal std for Normal
    min_mu = -25.0      # minimal mean
    max_mu = 25.0       # maximal mean
    
    assert dtypes in ['disc','gauss','discgauss']    
        
    if dtypes == 'disc':
        featureTypes = [0] * p
    elif dtypes == 'gauss':
        featureTypes = [1] * p    
    elif dtypes == 'discgauss':    
        # discrete or Normal features for now, chosen uniformly
        # 0 discrete, 1 Normal
        featureTypes = [ random.choice( (0, 1) )  for i in range(p) ]
    else:
        raise TypeError

    #print featureTypes


    # generate random CSI structures

    if G < 15:
        P = setPartitions.generate_all_partitions(G) # XXX too slow for large G
    #print P

    C = []
    
    leaders = []
    groups = []
    for j in range(p):
        c_j = {}
        
        leaders_j = []
        groups_j = {}
    
    
        if fullstruct == True:
            struct_j = [(i,) for i in range(G)]
            
        elif G < 15:
            struct_j = random.choice(P)
        else:
            print 'WARNING: improper structure sampling !'
            struct_j = setPartitions.get_random_partition(G)
        
        #print '\nstruct',j,struct_j
        
        for i,grp in enumerate(struct_j):
            
            lg = list(grp)        
            
            #print lg
            
            lgj = lg.pop(0)
            
            #print lgj
            
            leaders_j.append(lgj)
            groups_j[lgj] = lg

            max_tries = 100000
            tries = 0


            if featureTypes[j] == 0:
                acc = 0
                
                while acc == 0:
                    cand = discSamp.sample() 
                    
                    #print 'Cand:', cand
                    
                    acc = 1
                    for d in c_j:
                        KL_dist = mixture.sym_kl_dist(c_j[d],cand)
                        
                        #print c_j[d],cand, KL_dist
                        
                        if KL_dist > KL_upper or KL_dist < KL_lower:
                            acc = 0
                            tries += 1
                            break

                    if tries >= max_tries:
                        raise RuntimeError, 'Failed to find separated parameters !'
                                                
                    
                for cind in grp:
                    c_j[cind] = cand


            elif featureTypes[j] == 1:
                acc = 0
                while acc == 0:
                    mu = random.uniform(min_mu, max_mu)
                    sigma = random.uniform(min_sigma, max_sigma)
                    cand = mixture.NormalDistribution(mu, sigma )
                    acc = 1
                    
                    for d in c_j:
                        KL_dist = mixture.sym_kl_dist(c_j[d],cand)
                        if KL_dist > KL_upper or KL_dist < KL_lower:
                            acc = 0
                            tries += 1
                            break

                    if tries >= max_tries:
                        raise RuntimeError
                            
                    
                #    print '.',
                #print
                
                for cind in grp:
                    c_j[cind] = cand

            else:
                RuntimeError
                
        leaders.append(leaders_j)
        groups.append(groups_j)
        
        C.append(c_j)                
                
    comps = []
    for i in range(G):
        comps.append( mixture.ProductDistribution( [ C[j][i] for j in range(p) ] ) )

    pi = get_random_pi(G, 0.3 / G)
    #print '** pi =',pi 
    
    
    # create prior
    piprior = mixture.DirichletPrior(G,[2.0]*G)
    
    cprior = []
    for j in range(p):
        if featureTypes[j] == 0:
            cprior.append( mixture.DirichletPrior(M,[1.02]*M)) 

        elif featureTypes[j] == 1:
            cprior.append( mixture.NormalGammaPrior(0,0,0,0))   # dummy parameters, to be set later

        else:
            RuntimeError
        
    mprior = mixture.MixtureModelPrior(0.1,0.1, piprior, cprior)
    

    m = mixture.BayesMixtureModel(G,pi, comps, mprior, struct =1)            
    m.leaders = leaders
    m.groups = groups
    
    m.identifiable()
    m.updateFreeParams()
    #print m

    return m          
#-------------------------------------------------------------------------------------------        


# XXX return a CSI model containing multivariate features
def getRandomCSIMixture_conditionalDists(G, p, KL_lower, KL_upper, M=8, dtypes='discgauss', seed = None, fullstruct=False, disc_sampling_dist=None):
    
#    if seed:
#        random.seed(seed)
#        mixture._C_mixextend.set_gsl_rng_seed(seed)
#        #print '*** seed=',seed
#        
#    else: # XXX debug
#        seed = random.randint(1,9999999)
#        mixture._C_mixextend.set_gsl_rng_seed(seed)
#        random.seed(seed)
#        #print '*** seed=',seed

    

    if disc_sampling_dist == None:
        discSamp = mixture.DirichletPrior(M,[1.0] * M ) # uniform sampling
    else:
        discSamp = disc_sampling_dist   
        

    
    min_sigma = 0.3    # minimal std for Normal
    max_sigma = 5.0   # maximal std for Normal
    min_mu = -25.0      # minimal mean
    max_mu = 25.0       # maximal mean
    
    assert dtypes in ['disc','gauss','discgauss']    
        
    if dtypes == 'disc':
        featureTypes = [0] * p
    elif dtypes == 'gauss':
        featureTypes = [1] * p    
    elif dtypes == 'discgauss':    
        # discrete or Normal features for now, chosen uniformly
        # 0 discrete, 1 Normal
        featureTypes = [ random.choice( (0, 1) )  for i in range(p) ]
    else:
        raise TypeError

    #print featureTypes


    # generate random CSI structures

    if G < 15:
        P = setPartitions.generate_all_partitions(G) # XXX too slow for large G
    #print P

    C = []
    
    leaders = []
    groups = []
    for j in range(p):
        c_j = {}
        
        leaders_j = []
        groups_j = {}
    
    
        if fullstruct == True:
            struct_j = [(i,) for i in range(G)]
            
        elif G < 15:
            struct_j = random.choice(P)
        else:
            print 'WARNING: improper structure sampling !'
            struct_j = setPartitions.get_random_partition(G)
        
        #print '\nstruct',j,struct_j
        
        for i,grp in enumerate(struct_j):
            
            lg = list(grp)        
            
            #print lg
            
            lgj = lg.pop(0)
            
            #print lgj
            
            leaders_j.append(lgj)
            groups_j[lgj] = lg

            max_tries = 100000
            tries = 0


            if featureTypes[j] == 0:
                acc = 0
                
                while acc == 0:
                    cand = discSamp.sample() 
                    
                    #print 'Cand:', cand
                    
                    acc = 1
                    for d in c_j:
                        KL_dist = mixture.sym_kl_dist(c_j[d],cand)
                        
                        #print c_j[d],cand, KL_dist
                        
                        if KL_dist > KL_upper or KL_dist < KL_lower:
                            acc = 0
                            tries += 1
                            break

                    if tries >= max_tries:
                        raise RuntimeError, 'Failed to find separated parameters !'
                                                
                    
                for cind in grp:
                    c_j[cind] = cand


            elif featureTypes[j] == 1:
                acc = 0
                while acc == 0:
                    mu = random.uniform(min_mu, max_mu)
                    sigma = random.uniform(min_sigma, max_sigma)
                    cand = mixture.NormalDistribution(mu, sigma )
                    acc = 1
                    
                    for d in c_j:
                        KL_dist = mixture.sym_kl_dist(c_j[d],cand)
                        if KL_dist > KL_upper or KL_dist < KL_lower:
                            acc = 0
                            tries += 1
                            break

                    if tries >= max_tries:
                        raise RuntimeError
                            
                    
                #    print '.',
                #print
                
                for cind in grp:
                    c_j[cind] = cand

            else:
                RuntimeError
                
        leaders.append(leaders_j)
        groups.append(groups_j)
        
        C.append(c_j)                
                
    comps = []
    for i in range(G):
        comps.append( mixture.ProductDistribution( [ C[j][i] for j in range(p) ] ) )

    pi = get_random_pi(G, 0.3 / G)
    #print '** pi =',pi 
    
    
    # create prior
    piprior = mixture.DirichletPrior(G,[2.0]*G)
    
    cprior = []
    for j in range(p):
        if featureTypes[j] == 0:
            cprior.append( mixture.DirichletPrior(M,[1.02]*M)) 

        elif featureTypes[j] == 1:
            cprior.append( mixture.NormalGammaPrior(0,0,0,0))   # dummy parameters, to be set later

        else:
            RuntimeError
        
    mprior = mixture.MixtureModelPrior(0.1,0.1, piprior, cprior)
    

    m = mixture.BayesMixtureModel(G,pi, comps, mprior, struct =1)            
    m.leaders = leaders
    m.groups = groups
    
    m.identifiable()
    m.updateFreeParams()
    #print m

    return m          


#-------------------------------------------------------------------------------------------
# XXX debug 
def printModel(m,title):
    print '\n'+title+':'
    print 'pi:', ['%.3f' % m.pi[i] for i in range(m.G)]
    for jj in range(m.dist_nr):
        print 'Feature', jj,':'
        for ll in m.leaders[jj]:
            if isinstance(m.components[ll][jj], mixture.DiscreteDistribution):
                f = lambda x: '%.3f' % x
                print '  ',[ll]+  m.groups[jj][ll],':', map(f, m.components[ll][jj].phi)
            else:
                print '  ',[ll]+  m.groups[jj][ll],':', m.components[ll][jj]
    print


def printStructure(m):
    s = []
    for j in range(m.dist_nr):
        s.append([])
        for l in m.leaders[j]:
            s[j].append( (l,)+tuple( m.groups[j][l] ) )

    print s



# XXX mostly copied from MixtureModel.minimalStructure(m)
def checkComponentRedundancy(leaders, groups):
        
        distNr = len(leaders)
        # features with only one group can be excluded
        exclude = []
        for i in range(distNr):
            if len(leaders[i]) == 1:
                exclude.append(i)
        # get first feature with more than one group
        first = -1
        for i in range(distNr):
            if i not in exclude:
                first = i
                break
        # initialising group dictionaries for first non-trivial group structure
        firstgroup_dicts = []   
        for j in range(len(leaders[first])):
            d = {}
            for k in [leaders[first][j]] + groups[first][leaders[first][j]]:
                d[k] = 1
            firstgroup_dicts.append(d)    

        # initialising group dictionaries for remaining features
        allgroups_dicts = []        
        for i in range(first+1,distNr,1):
            if i in exclude:
                continue
            gdicts_i = []
            for l in leaders[i]:
                d = {}
                for k in [l] + groups[i][l]:
                    d[k]  = 1
                gdicts_i.append(d)    
            allgroups_dicts.append(gdicts_i)    

        toMerge = []
        # for each group in first non-trivial structure
        for g, fdict in enumerate(firstgroup_dicts):
            candidate_dicts = [fdict]
            # for each of the other non-trivial features
            for i, dict_list in enumerate(allgroups_dicts):
                new_candidate_dicts = [] 
                # for each group in the i-th feature
                for adict in dict_list:
                    # for each candidate group
                    for j,cand_dict in enumerate(candidate_dicts):
                        # find intersection
                        inter_d = mixture.dict_intersection(cand_dict, adict)       
                        if len(inter_d) >= 2:
                            new_candidate_dicts.append(inter_d)
                candidate_dicts = new_candidate_dicts
                # check whether any valid candidates are left
                if len(candidate_dicts) == 0:
                    break
            if len(candidate_dicts) > 0:
                for c in candidate_dicts:
                    toMerge.append(c.keys())
        
        #print '***',toMerge
        for i in range(len(toMerge)-1,-1,-1):
            if len(toMerge[i]) == 1:
                toMerge.pop(i)
        
        
        return toMerge
        
#        print 'Components:',m.G-len(toMerge)
    

def matchModelStructures(gen, m):
    """
    Checks whether m1 and m2 are consistent in the sense that for each
    leader in m1, there is a number of distributions in m2 which take minimum
    distance to the leader as the number of distributions in the group in m1.
    
    Used to check whether a the parameteric EM has obviously captured the CSI structure of
    the generating model.
    """
    #print '**** matchModelStructures'
    
    gen_csi = []
    for j in range(gen.dist_nr):    
        gen_csi.append({})
        for l in gen.leaders[j]:
            gen_csi[j][ tuple( [l] + gen.groups[j][l] ) ] = []
    
    #print gen_csi
    
    for j in range(gen.dist_nr):
        #print 'feature:',j
        for i1 in range(m.G):
            kldists = numpy.zeros(m.G)
            for i2 in range(m.G):
                kldists[i2] = mixture.sym_kl_dist(m.components[i1][j], gen.components[i2][j])
            cg = numpy.where( kldists == kldists.min() )[0]

            gen_csi[j][tuple(cg)].append(i1)
                

    #print gen_csi       
    # check easy case: all components match in gen and m
    match = 1
    for j in range(gen.dist_nr):
        for cg in gen_csi[j]:
            if cg != tuple(gen_csi[j][cg]):
                match = 0
    if match:
        #print 'Simple match !'             
        return 1

    # check whether component indices have changed but the structures are consistent otherwise
    cmaps = []
    for j in range(gen.dist_nr):   
        cmaps.append({})
        for i1 in range(m.G):
            kldists = numpy.zeros(m.G)
            for i2 in range(m.G):
                kldists[i2] = mixture.sym_kl_dist(m.components[i1][j], gen.components[i2][j])
            cg = numpy.where( kldists == kldists.min() )[0]
            cmaps[j][i1]=cg
    
    #print cmaps   

    gen_compred = checkComponentRedundancy(gen.leaders, gen.groups)    
    
    #print 'gen_compred', gen_compred 
    
    if len(gen_compred) > 1:
        return 0  # XXX case not covered yet 
    
    match = 1
    m_to_gen = {}
    for i in range(m.G):
        m_to_gen[i] = -1
    
    for j in range(gen.dist_nr):    
        for i in cmaps[j]:
            if len(cmaps[j][i]) == 1:
                if m_to_gen[i] == -1:
                    m_to_gen[i] = cmaps[j][i][0]
                else:
                    if m_to_gen[i] == cmaps[j][i][0]:
                        continue
                    else:
                        match = 0
                        break
    #print m_to_gen   

    if len(gen_compred) == 0:
        for k in m_to_gen:
            #print m_to_gen[k]
            if m_to_gen[k] == -1:
                return 0
        return 1

    for k in m_to_gen:
        if m_to_gen[k] == -1:  # no assignment so far
            for j in range(gen.dist_nr):
                #print gen_compred
                #print k, cmaps[j][k].tolist(),gen_compred[0]
                if not cmaps[j][k].tolist() == gen_compred[0]:

                    match = 0                    
            

    #print '*** match=', match

    return match
    

def mixtureKLdistance(m1,m2):

    d = 0.0
    
    for k1 in range(m1.G):
        dd = numpy.zeros(m2.G)
        for k2 in range(m2.G):
            dd[k2] = product_distribution_sym_kl_dist(m1.components[k1], m2.components[k2])
        #d += m1.pi[k1] * dd.min()    
        d += dd.min()    
    return d   
     

def mixtureMaxKLdistance(m1,m2):
    d = numpy.zeros(m1.G)
   
    for k1 in range(m1.G):
        dd = numpy.zeros(m2.G)
        for k2 in range(m2.G):
            dd[k2] = product_distribution_sym_kl_dist(m1.components[k1], m2.components[k2])
        d[k1] = dd.min()    

    return d.max()   


#-------------------------------------------------------------------------------------------

def scoreStructureLearning(N, gen, delta, seed=None, silent=False, skipAfterRNGcalls = False):
    """
    If skipAfterRNGcalls is True, the function terminates after all calls to RNGs have been done.
    """


    #print 'start scoring'

#    if seed != None:
#        random.seed(seed)
#        mixture._C_mixextend.set_gsl_rng_seed(seed)
#        print '*** given seed=',seed
#        
#    else: # XXX debug
#        seed = random.randint(1,999999999)
#        random.seed(seed)
#        mixture._C_mixextend.set_gsl_rng_seed(seed)
#        print '*** random seed=',seed


        
    data = gen.sampleDataSet(N)                



    # XXX update NormalGammaPrior hyperparameters
    for j in range(gen.dist_nr):
        if isinstance(gen.prior.compPrior[j], mixture.NormalGammaPrior):
            gen.prior.compPrior[j].setParams(data.getInternalFeature(j), gen.G)
    
    gen.prior.structPriorHeuristic(delta, data.N)
    
    print '\nupdating generating model structure:'
    print 'vorher:'
    print gen.leaders
    print gen.groups


    fullEnumerationExhaustive.updateStructureBayesianFullEnumeration(gen, data, silent=1)

    print '\nnachher:'
    print gen.leaders
    print gen.groups


    if silent == False:
        printModel(gen,'generating model')


    m = copy.copy(gen)
    # reset structure
    m.initStructure()
    
    # training parameters
    nr_rep = 40 # XXX
    #nr_rep = 4 # XXX
    
    
    nr_steps = 400
    em_delta = 0.6
    
    print 'start training'
    print 'EM repeats:',nr_rep

    m.randMaxTraining(data,nr_rep, nr_steps,em_delta,silent=1,rtype=0)
    print 'finished training'

    if skipAfterRNGcalls == True:
        print '*** Skipping !'
        return numpy.zeros(4)



#    # check for consistency of component indices (identifiability issues)
#    bad = 0
    if silent == False:
        cmap = {}
        
        for j in range(gen.dist_nr):
            print '\nfeature:',j
    
            for i1 in range(m.G):
                kldists = numpy.zeros(m.G)
                for i2 in range(m.G):
                    kldists[i2] = mixture.sym_kl_dist(m.components[i1][j], gen.components[i2][j])
                print i1,'->', kldists.argmin(), map(lambda x:'%.2f' % float(x),kldists)     # kldists.min()
            
        
#        for i1 in range(m.G):
#            print
#            cdists = numpy.zeros(m.G)
#            for i2 in range(m.G):
#                cdists[i2] = product_distribution_sym_kl_dist(m.components[i1], gen.components[i2])
#                #print i1,i2,product_distribution_sym_kl_dist(m.components[i1], gen.components[i2])
#
#            print i1,'maps to', numpy.argmin(cdists), cdists.tolist()
#            amin = numpy.argmin(cdists)
#            if not amin == i1:     # minimal KL distance should occur at equal indices in gen and m
#                bad = 1
#                cmap[i1] = amin

#    if bad:            
#        
#       
#        
#        # XXX check whether cmap defines new unambiguous ordering
#        
#        # check whether components have switched positions
#        reorder = 0
#        order = range(m.G)
#        try:
#            
#            #print cmap
#            
#            for i1 in cmap.keys():
#                order[i1] = cmap[i1]
#            
#            #print order
#            #print set(order)
#            #print  list(set(order))
#            
#            if len(set(order)) == m.G:
#                reorder = 1        
#        except KeyError:
#            pass
#        except AssertionError:    
#            pass
#            
#        if reorder:
#            print '** new order', order
#            
#            m.reorderComponents(order)
#
#        else:
#                    
#        
#            #print cdists
#            print i1,'maps to', numpy.argmin(cdists)
#
#            print 'Failed matching.'
#
#            print 'generating model gen:'
#            print gen
#
#            print 'trained model m:'
#            print m 
#
#            raise ValueError     


#    mtest =copy.copy(gen)
#    ch = mtest.updateStructureBayesian(data,silent=1)
#    print '\nTEST:',ch
#    for j in range(m.dist_nr):
#        print j,mtest.leaders[j], mtest.groups[j]


    #print m.prior

    print '-----------------------------------------------------------------------'
    print '\n True structure:'
    print 'True model post:',mixture.get_loglikelihood(gen, data) + gen.prior.pdf(gen)
    #for j in range(m.dist_nr):
    #    print j,gen.leaders[j], gen.groups[j]
    print gen.leaders
    print gen.groups

    if silent == False:
        printModel(m,'trained model')

    m1 = copy.copy(m)
    t0 = time.time()
    #print '\n\n################# TOPDOWN #####################'
    m1.updateStructureBayesian(data,silent=1)
    t1 = time.time()
    time2 = t1-t0    
    #m1.mapEM(data,40,0.1)
    print '\nTop down (',str(time2),'s ):'
    print m1.leaders
    print m1.groups
    print 'Top down model post:',mixture.get_loglikelihood(m1, data) + m1.prior.pdf(m1)
#    print 'Accuracy:',mixture.structureAccuracy(gen,m1)  # structureEditDistance(gen,m1)

    if silent == False:
        printModel(m1,'top down model')


    #print '#############################'
    

    #print '\n\n################# FULL FixedOrder #####################'
    m2 = copy.copy(m)
    t0 = time.time()
    m2.updateStructureBayesianFullEnumerationFixedOrder(data,silent=1)
    t1 = time.time()
    time2 = t1-t0    
    #m2.mapEM(data,40,0.1)

#    print
#    for j in range(m2.dist_nr):
#        print j,m2.leaders[j], m2.groups[j]
    print '\nFull enumeration Fixed Order  (',str(time2),'s ):'
    print m2.leaders
    print m2.groups
    print 'Full fixed order model post:',mixture.get_loglikelihood(m2, data) + m2.prior.pdf(m2)
#    print 'Accuracy:',mixture.structureAccuracy(gen,m2) # structureEditDistance(gen,m1)


    if silent == False:
        printModel(m2,'full fixed model')


        

    #print '\n\n################# BOTTUMUP #####################'    
    m3 = copy.copy(m)
    t0 = time.time()
    m3.updateStructureBayesianBottomUp(data,silent=1)
    t1 = time.time()
    time2 = t1-t0    
    #m3.mapEM(data,40,0.1)
#    print 
#    for j in range(m3.dist_nr):
#        print j,m3.leaders[j], m3.groups[j]
    print '\nBottom up: (',str(time2),'s ):'
    print m3.leaders
    print m3.groups
    print 'Bottom up model post:',mixture.get_loglikelihood(m3, data) + m3.prior.pdf(m3)
#    print 'Accuracy:',mixture.structureAccuracy(gen,m3) # structureEditDistance(gen,m1)


    if silent == False:
        printModel(m3,'bottom up model')

    
    #print '\n\n################# FULL enumeration #####################'
    m4 = copy.copy(m)
    t0 = time.time()
    fullEnumerationExhaustive.updateStructureBayesianFullEnumeration(m4, data, silent=0)
    t1 = time.time()
    time2 = t1-t0    
   # m4.mapEM(data,40,0.1)
#    print 
#    for j in range(m4.dist_nr):
#        print j,m4.leaders[j], m4.groups[j]
    print '\nFull enumeration: (',str(time2),'s )'
    print m4.leaders
    print m4.groups
    print 'Full enumeration model post:',mixture.get_loglikelihood(m4, data) + m4.prior.pdf(m4)
#    print 'Accuracy:',mixture.structureAccuracy(gen,m4)

    if silent == False:
        printModel(m4,'full enumeration model')


    print '-----------------------------------------------------------------------'



#    dtop = mixture.structureAccuracy(gen,m1)
#    dfull_fixed = mixture.structureAccuracy(gen,m2) 
#    dfull = mixture.structureAccuracy(gen,m4) 
#    dbottom = mixture.structureAccuracy(gen,m3)

    logp_top = mixture.get_loglikelihood(m1, data) + m1.prior.pdf(m1)
    logp_full_fixed = mixture.get_loglikelihood(m2, data) + m2.prior.pdf(m2)
    logp_full = mixture.get_loglikelihood(m4, data) + m4.prior.pdf(m4)
    logp_bottom = mixture.get_loglikelihood(m3, data) + m3.prior.pdf(m3)


    if (not (round(logp_top,3) <= round(logp_full,3) ) or not (round(logp_full_fixed,3) <= round(logp_full,3))
        or not (round(logp_bottom,3) <= round(logp_full,3)) ):
        raise ValueError


    return numpy.array([ logp_top, logp_full_fixed, logp_full, logp_bottom ])

    
#    for i in range(m.G):
#        print '\n',i,
#        for j in range(m.dist_nr):
#            print gen.components[i][j],
#        print '\n '
#        for j in range(m.dist_nr):
#            print m.components[i][j], 
#        print    
    
    #,m1.components[i][j],m2.components[i][j],m3.components[i][j],
    



#-----------------------------------------------------------------------------------    
def scoreStructureLearning_diffFullVsTopdown(N, gen, delta, seed=None, silent=False, skipAfterRNGcalls = False):
    """
    If skipAfterRNGcalls is True, the function terminates after all calls to RNGs have been done.
    """


    #print 'start scoring'

#    if seed != None:
#        random.seed(seed)
#        mixture._C_mixextend.set_gsl_rng_seed(seed)
#        print '*** given seed=',seed
#        
#    else: # XXX debug
#        seed = random.randint(1,999999999)
#        random.seed(seed)
#        mixture._C_mixextend.set_gsl_rng_seed(seed)
#        print '*** random seed=',seed


        
    data = gen.sampleDataSet(N)                



    # XXX update NormalGammaPrior hyperparameters
    for j in range(gen.dist_nr):
        if isinstance(gen.prior.compPrior[j], mixture.NormalGammaPrior):
            gen.prior.compPrior[j].setParams(data.getInternalFeature(j), gen.G)
    
    gen.prior.structPriorHeuristic(delta, data.N)
    
#    print '\nupdating generating model structure:'
#    print 'vorher:'
#    print gen.leaders
#    print gen.groups


    fullEnumerationExhaustive.updateStructureBayesianFullEnumeration(gen, data, silent=1)

#    print '\nnachher:'
#    print gen.leaders
#    print gen.groups




    m = copy.copy(gen)
    # reset structure
    m.initStructure()
    
    # training parameters
    nr_rep = 40 # XXX
    #nr_rep = 4 # XXX
    
    
    nr_steps = 400
    em_delta = 0.6
    
#    print 'start training'
#    print 'EM repeats:',nr_rep

    m.randMaxTraining(data,nr_rep, nr_steps,em_delta,silent=1,rtype=0)
#    print 'finished training'

    if skipAfterRNGcalls == True:
        print '*** Skipping !'
        return numpy.zeros(4)


    m1 = copy.copy(m)
    t0 = time.time()
    #print '\n\n################# TOPDOWN #####################'
    m1.updateStructureBayesian(data,silent=1)
    t1 = time.time()
    time2 = t1-t0    
    #m1.mapEM(data,40,0.1)
#    print 'Accuracy:',mixture.structureAccuracy(gen,m1)  # structureEditDistance(gen,m1)



    #print '#############################'
    

    #print '\n\n################# FULL FixedOrder #####################'
    m2 = copy.copy(m)
    t0 = time.time()
    m2.updateStructureBayesianFullEnumerationFixedOrder(data,silent=1)
    t1 = time.time()
    time2 = t1-t0    
    #m2.mapEM(data,40,0.1)

#    print
#    for j in range(m2.dist_nr):
#        print j,m2.leaders[j], m2.groups[j]
#    print 'Accuracy:',mixture.structureAccuracy(gen,m2) # structureEditDistance(gen,m1)



        

    #print '\n\n################# BOTTUMUP #####################'    
    m3 = copy.copy(m)
    t0 = time.time()
    m3.updateStructureBayesianBottomUp(data,silent=1)
    t1 = time.time()
    time2 = t1-t0    
    #m3.mapEM(data,40,0.1)
#    print 
#    for j in range(m3.dist_nr):
#        print j,m3.leaders[j], m3.groups[j]
#    print 'Accuracy:',mixture.structureAccuracy(gen,m3) # structureEditDistance(gen,m1)



    
    #print '\n\n################# FULL enumeration #####################'
    m4 = copy.copy(m)
    t0 = time.time()
    fullEnumerationExhaustive.updateStructureBayesianFullEnumeration(m4, data, silent=1)
    t1 = time.time()
    time2 = t1-t0    
   # m4.mapEM(data,40,0.1)
#    print 
#    for j in range(m4.dist_nr):
#        print j,m4.leaders[j], m4.groups[j]
#    print 'Accuracy:',mixture.structureAccuracy(gen,m4)


    logp_top = mixture.get_loglikelihood(m1, data) + m1.prior.pdf(m1)
    logp_full_fixed = mixture.get_loglikelihood(m2, data) + m2.prior.pdf(m2)
    logp_full = mixture.get_loglikelihood(m4, data) + m4.prior.pdf(m4)
    logp_bottom = mixture.get_loglikelihood(m3, data) + m3.prior.pdf(m3)


    if (not (round(logp_top,3) <= round(logp_full,3) ) or not (round(logp_full_fixed,3) <= round(logp_full,3))
        or not (round(logp_bottom,3) <= round(logp_full,3)) ):
        print 'ERROR:'
        print 'top:',logp_top
        print 'full fixed:',logp_full_fixed
        print 'full:',logp_full
        print 'bottom:',logp_bottom,'\n'
        
        printModel(gen,'generating model')
        printStructure(gen)
        print
        printModel(m4,'full enumeration model')
        printStructure(m4)
        print
        printModel(m2,'fixed full model')
        printStructure(m2)
        
        raise ValueError

#    # as a measure of separation of the component in the trained model, sum up 
#    # sym. KL divergence of all components and features
#    train_diff = 0
#    for j in range(gen.dist_nr):
#        for i1 in range(m.G):
#            for i2 in range(m.G):
#                train_diff += mixture.sym_kl_dist(m.components[i1][j], m.components[i2][j])

    mix_dist1 =  mixtureKLdistance(gen,m)
    mix_dist2 =  mixtureKLdistance(m, gen)
    
    max_dist1 = mixtureMaxKLdistance(gen,m)
    max_dist2 = mixtureMaxKLdistance(m, gen)

    # number of leaders in the full enumeration model
    nr_full_lead = 0
    for ll in m4.leaders:
        nr_full_lead += len(ll)

    match = matchModelStructures(gen, m)

    compred = checkComponentRedundancy(gen.leaders, gen.groups)
    if not(str(logp_top) == str(logp_full_fixed) == str(logp_full) ):

        print '-----------------------------------------------------------------------'

        print 'Different:'
        print 'top:',logp_top
        print 'full fixed:',logp_full_fixed
        print 'full:',logp_full
        print 'bottom:',logp_bottom,'\n'

        explain = 0
        if str(compred) != '[]':
            print '*** redundant components',compred
            explain = 1
        if gen.pi.min() < 0.05:
            print '*** vanishing component in generating model'
            explain = 1
        if m.pi.min() < 0.05:
            print '*** vanishing component in trained model'
            explain = 1

        if explain == 0:
            print '*** UNEXPLAINED !'


        printModel(gen,'generating model')
        printModel(m,'trained model')
        #print 'Trained model diff (simplistic):',train_diff
        print 'D: Mixture distance gen/trained:',mix_dist1
        print 'D: Mixture distance trained/gen:',mix_dist2

        print 'D: Mixture Max-distance gen/trained:',max_dist1
        print 'D: Mixture Max-distance trained/gen:',max_dist2


        print '\nGenerating distances to self:'
        cmap = {}
        for j in range(gen.dist_nr):
            print 'feature:',j
            for i1 in range(m.G):
                kldists = numpy.zeros(m.G)
                for i2 in range(m.G):
                    kldists[i2] = mixture.sym_kl_dist(gen.components[i1][j], gen.components[i2][j])
                print map(lambda x:'%.2f' % float(x),kldists)     # kldists.min()

        print '\nTrained distances to self:'
        cmap = {}
        for j in range(gen.dist_nr):
            print 'feature:',j
            for i1 in range(m.G):
                kldists = numpy.zeros(m.G)
                for i2 in range(m.G):
                    kldists[i2] = mixture.sym_kl_dist(m.components[i1][j], m.components[i2][j])
                print map(lambda x:'%.2f' % float(x),kldists)     # kldists.min()


        print '\nTrained distances to generating:'
        cmap = {}
        for j in range(gen.dist_nr):
            print 'feature:',j
            for i1 in range(m.G):
                kldists = numpy.zeros(m.G)
                for i2 in range(m.G):
                    kldists[i2] = mixture.sym_kl_dist(m.components[i1][j], gen.components[i2][j])
                print i1,'->', kldists.argmin(), map(lambda x:'%.2f' % float(x),kldists)     # kldists.min()



        print '\n True structure:'
        print 'True model post:',mixture.get_loglikelihood(gen, data) + gen.prior.pdf(gen)
        #for j in range(m.dist_nr):
        #    print j,gen.leaders[j], gen.groups[j]
        printStructure(gen)
        
        print '\nTop down:'
        printStructure(m1)
        print 'Top down model post:',mixture.get_loglikelihood(m1, data) + m1.prior.pdf(m1)
        printModel(m1,'top down model')

        print '\nFull enumeration Fixed Order:'
        printStructure(m2)
        print 'Full fixed order model post:',mixture.get_loglikelihood(m2, data) + m2.prior.pdf(m2)
        printModel(m2,'full fixed model')

        print '\nBottom up:'
        printStructure(m3)
        print 'Bottom up model post:',mixture.get_loglikelihood(m3, data) + m3.prior.pdf(m3)
        printModel(m3,'bottom up model')

        print '\nFull enumeration:' 
        printStructure(m4)
        print 'Full enumeration model post:',mixture.get_loglikelihood(m4, data) + m4.prior.pdf(m4)
        printModel(m4,'full enumeration model')


        print '-----------------------------------------------------------------------'

    elif str(compred) != '[]' and nr_full_lead > m4.p and match != 1:  # redundant components and not fully merged
        print '-----------------------------------------------------------------------'
        print 'Same but redundant components:', compred



        printModel(gen,'generating model')
        printModel(m,'trained model')
        #print 'Trained model diff:',train_diff        
        print 'S: Mixture distance gen/trained:',mix_dist1
        print 'S: Mixture distance trained/gen:',mix_dist2

        print 'S: Mixture Max-distance gen/trained:',max_dist1
        print 'S: Mixture Max-distance trained/gen:',max_dist2

        
        print '\nGenerating distances to self:'
        cmap = {}
        for j in range(gen.dist_nr):
            print 'feature:',j
            for i1 in range(m.G):
                kldists = numpy.zeros(m.G)
                for i2 in range(m.G):
                    kldists[i2] = mixture.sym_kl_dist(gen.components[i1][j], gen.components[i2][j])
                print i1,':', map(lambda x:'%.2f' % float(x),kldists)     # kldists.min()

        print '\nTrained distances to self:'
        cmap = {}
        for j in range(gen.dist_nr):
            print 'feature:',j
            for i1 in range(m.G):
                kldists = numpy.zeros(m.G)
                for i2 in range(m.G):
                    kldists[i2] = mixture.sym_kl_dist(m.components[i1][j], m.components[i2][j])
                print i1,':', map(lambda x:'%.2f' % float(x),kldists)     # kldists.min()


        print '\nTrained distances to generating:'
        cmap = {}
        for j in range(gen.dist_nr):
            print 'feature:',j

            for i1 in range(m.G):
                kldists = numpy.zeros(m.G)
                for i2 in range(m.G):
                    kldists[i2] = mixture.sym_kl_dist(m.components[i1][j], gen.components[i2][j])
                print i1,'->', kldists.argmin(), map(lambda x:'%.2f' % float(x),kldists)     # kldists.min()



        print '\n True structure:'
        print 'True model post:',mixture.get_loglikelihood(gen, data) + gen.prior.pdf(gen)
        #for j in range(m.dist_nr):
        #    print j,gen.leaders[j], gen.groups[j]
        printStructure(gen)
        

        print '\nTop down:'
        printStructure(m1)
        print 'Top down model post:',mixture.get_loglikelihood(m1, data) + m1.prior.pdf(m1)

        print '\nFull enumeration Fixed Order:'
        printStructure(m2)
        print 'Full fixed order model post:',mixture.get_loglikelihood(m2, data) + m2.prior.pdf(m2)

        print '\nBottom up:'
        printStructure(m3)
        print 'Bottom up model post:',mixture.get_loglikelihood(m3, data) + m3.prior.pdf(m3)

        print '\nFull enumeration:' 
        printStructure(m4)
        print 'Full enumeration model post:',mixture.get_loglikelihood(m4, data) + m4.prior.pdf(m4)

        print '-----------------------------------------------------------------------'
    
#    else:
#        print '-----------------------------------------------------------------------'
#        print 'S: Mixture distance gen/trained:',mix_dist1
#        print 'S: Mixture distance trained/gen:',mix_dist2
#        print '-----------------------------------------------------------------------'


#    else:
#        print '** all equal.'


#    dtop = mixture.structureAccuracy(gen,m1)
#    dfull_fixed = mixture.structureAccuracy(gen,m2) 
#    dfull = mixture.structureAccuracy(gen,m4) 
#    dbottom = mixture.structureAccuracy(gen,m3)



    return numpy.array([ logp_top, logp_full_fixed, logp_full, logp_bottom ])

    
#    for i in range(m.G):
#        print '\n',i,
#        for j in range(m.dist_nr):
#            print gen.components[i][j],
#        print '\n '
#        for j in range(m.dist_nr):
#            print m.components[i][j], 
#        print    
    
    #,m1.components[i][j],m2.components[i][j],m3.components[i][j],
    



#-----------------------------------------------------------------------------------    

    
    
def evaluateStructureLearning(rep, N, G, p, KL_lower, KL_upper, dtypes='discgauss', seed = None):

#    if seed:
#        random.seed(seed)
    
    dists = numpy.zeros(3)
    for r in range(rep):
    
        gen = getRandomCSIMixture(G, p, KL_lower, KL_upper, dtypes=dtypes)   
        dists += scoreStructureLearning(N, gen)
    

    print '\nAverage distances to generating ('+str(rep)+' runs)'    
    print 'Top down',dists[0] / float(rep)
    print 'Full enumeration',dists[1] / float(rep)
    print 'Bottom up',dists[2] / float(rep)
    
    
    
    
def timeStructureLearning(rep, N, G, p, KL_lower, KL_upper, M=8, dtypes='discgauss', seed = None, disc_sampling_dist = None):
    
    if seed:
        random.seed(seed)


    # training parameters
    nr_rep = 10
    nr_steps = 400
    delta = 0.5

    
    dists = numpy.zeros(3)
    
    t_history = []
    t_old = []
    t_bound = []
    t_historybound = []
    
    #avg_quick_reject = 0.0
    
    for r in range(rep):
    
        try:
        
            gen = getRandomCSIMixture(G, p, KL_lower, KL_upper, M=M, dtypes=dtypes, disc_sampling_dist = disc_sampling_dist )   
        except RuntimeError: # XXX
            print '*** skipping',r 
            continue # XXX
            
        
        data  = gen.sampleDataSet(N)

        m = copy.copy(gen)
        # XXX update NormalGammaPrior hyperparameters
        for j in range(m.dist_nr):
            if isinstance(m.prior.compPrior[j], mixture.NormalGammaPrior):
                m.prior.compPrior[j].setParams(data.getInternalFeature(j), m.G)
                #m.prior.compPrior[j].scale = m.prior.compPrior[j].scale * 100 # XXX TEST
                
        
        #m.prior.piPrior = mixture.DirichletPrior(G,[(N/10.0)/G]*G)  # XXX try strong pi prior
        
        m.prior.structPriorHeuristic(0.05, data.N)
    
        # reset structure
        m.initStructure()

        print '** Training'
    
        #m.randMaxTraining(data,nr_rep, nr_steps,delta,silent=1,rtype=0)
        m.mapEM(data, 1, 0.1, silent=1)  # one EM update for perturbation


     

        m1 = copy.copy(m)    
        m2 = copy.copy(m)    
        m3 = copy.copy(m)    
        m4 = copy.copy(m)   
        gc.disable()

        t0 = time.time()
        
        StructureLearningVariants.updateStructureBayesian_OLD(m2, data, silent=1)

        t1 = time.time()
        time2 = t1-t0
        print '\nold:',time2, 'seconds'    
        t_old.append(time2)

        gc.collect()
    
        t0 = time.time()
        m1.updateStructureBayesian( data,silent=1)
        t1 = time.time()
        time1 = t1-t0
        print 'history:',time1, 'seconds'    
        t_history.append(time1)

        gc.collect()
    
        t0 = time.time()
        StructureLearningVariants.updateStructureBayesian_OLDDECISIONBOUNDS(m4, data,silent=1)
        t1 = time.time()
        time4 = t1-t0
        print 'bound:',time4, 'seconds'    
        t_bound.append(time4)

        gc.collect()

        t0 = time.time()
        r = StructureLearningVariants.updateStructureBayesianDECISIONBOUNDS(m3, data, silent=1,returntypes='boundcounts')  #
        t1 = time.time()
        time3 = t1-t0
        print 'historybound:',time3, 'seconds'    
        t_historybound.append(time3)

#        print '  acc:', r[0] / (r[0]+r[1])
#        print '  rej:', r[2] / (r[2]+r[3]),'\n'
        #avg_quick_reject += r[2] / (r[2]+r[3])

        gc.enable()

#        if not (m1 == m2 == m3 == m4):
#            #print m1,'\n\n'
#            #print m2,'\n\n'
#            #print m3    
#            #if not (numpy.all(m1.pi == m2.pi == m3.pi)):
#            #    print m1.pi, m2.pi, m3.pi
#            
#            for j in range(m1.dist_nr): 
#                for i in range(m1.G):
#                    if not ( m1.components[i][j]== m2.components[i][j] == m3.components[i][j] == m4.components[i][j]):
#                        print i,j,m1.components[i][j],m2.components[i][j],m3.components[i][j], m4.components[i][j]
#            
#            
#            raise AssertionError
        
    #print '\navg. quick reject:',avg_quick_reject / rep

    return numpy.array(t_old), numpy.array(t_history), numpy.array(t_bound),  numpy.array(t_historybound)   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
