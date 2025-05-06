
import math
import numpy as np
import scipy

import poseigen_seaside.basics as se

import torch                                #!!!!!!!!!!!!!!!!!!!!!!!!!!!!
import torch.nn as nn



def MeanExpo(inp, expo = 2, root = False):
    
    #inp needs to be an array!!!!!!!!!!! NO LISTS 
    
    if expo != 2: root = False
    inp = inp**expo
    ME = inp.mean()
    if root: ME = ME**(1/2)

    return ME

def WeightsAdjuster(inp1, weights):
    if weights == None or isinstance(weights, int): 
        weights = 1
    return weights

def AError(inp1, inp2, weights = None, mean = True,
           revbase = None, revpseudo = False, 
           revsqrt = None, 
           rel = False, 
           expo = 2, root = False, weightbefore = False): 
    
    #added revbase and revpseudo to reverse log transformations.
    #Applies to both inputs. 

    if revbase != None:
        inp1, inp2 = [(revbase ** x) - revpseudo for x in [inp1, inp2]]
    elif revsqrt != None: 
        inp1, inp2 = [(x ** revsqrt) - revpseudo for x in [inp1, inp2]]

    if expo != 2: root = False
    w = WeightsAdjuster(inp1, weights)

    rr = inp2 if rel else 1

    e = abs((inp1-inp2) / rr) #RELATIVE ERROR 

    if weightbefore: w = w **expo
    
    AE = w * (e ** expo)
    
    if mean == True:
        if isinstance(w, int) == False: AE = AE.sum() / w.sum()
        else: AE = AE.mean()

        if root: AE = AE**(1/2)

    return AE



def ZError(inp1, inp2, std = 1, 
                    weights = None, mean = True,
                    pseudo = 1, log = False,
                    expo = 2, root = False, weightbefore = False): 
    
    if expo != 2: root = False
    weights = WeightsAdjuster(inp1, weights)

    e = (abs(inp1-inp2)) / (pseudo + std)
    if log: e = log(1 + e) #########

    AE = (weights * e)**expo if weightbefore else weights * (e **expo)
    
    if mean is True:  
        AE = AE.mean()
        if root: AE = AE**(1/2)
    
    return AE


def Correlation(inp1, inp2, weights = None, inverse = False): 

    weights = WeightsAdjuster(inp1, weights)
    if isinstance(weights, int): cor = np.corrcoef(inp1,inp2)[0, 1]
    else: 
        def m(inp1, weights): return np.sum(inp1 * weights) / np.sum(weights)
        def cov(inp1, inp2, weights): return np.sum(weights * (inp1 - m(inp1, weights)) * (inp2 - m(inp2, weights))) / np.sum(weights)
        cor = cov(inp1, inp2, weights) / np.sqrt(cov(inp1, inp1, weights) * cov(inp2, inp2, weights))
    return cor if inverse == False else 1/cor


metrics_smallest = {
                    Correlation: False,
                    AError: True,
                    }   




#====================== METRICS THAT REQUIRE PYTORCH ================================

def PearsonError(inp1, inp2, weights = None,
                 pyt = False): 
    
    # "weights" is a placeholder. 
    # Currently only works for single outputs! 
    
    cosf = nn.CosineSimilarity(dim=1, eps=1e-6) if pyt else se.CosineSimilarity
    mx = inp2.mean()
    my = inp1.mean()

    xm, ym = inp2 - mx, inp1 - my
    
    return (1-cosf(xm,ym)).sum()

def BetaPrime_msNLL(inp1, inp2, std = 1, 
                  weights = None, pyt = False, pseudo = 1e-10):
    
    # "weights" is a placeholder. 
    
    alpha1 = std[:, :, :, :, 0]
    alpha2 = std[:, :, :, :, 1]

    lix = tu.BetaPrime_Rel(inp1, alpha1, alpha2, pyt = pyt, pseudo = pseudo)
    
    return 0 - ((math.log(lix + pseudo)).mean())



def BetaPrime_msNMP(inp1, inp2, std = 1, 
                  weights = None, pyt = False, pseudo = 1e-10):
    
    alpha1 = std[:, :, :, :, 0]
    alpha2 = std[:, :, :, :, 1]

    lix = tu.BetaPrime_Rel(inp1, alpha1, alpha2, pyt = pyt, pseudo = pseudo)

    return 0 - (lix.mean())




#----------------------------------------

def TocherApprox(z):
    # REQUIRES INPUTS TO HAVE Z > 0 !!!!!!!!!    
    
    pi = 3.141592653589793
    e = 2.718281828459045
    bay = (2 / pi) ** 0.5
    pa = e ** (2 * bay * z)

    return pa / (1 + pa)

def TocherApproxTwoSide(z):
    # REQUIRES INPUTS TO HAVE Z > 0 !!!!!!!!!
    xo = TocherApprox(z)
    return (xo - 0.5) * 2

def TDistroCDF(z, n = None, pyt = False): 

    pi = 3.141592653589793

    if n is not None: 
        df = n - 1
        if df == 1: 
            arctanf = torch.atan if pyt else np.arctan
            zo = 0.5 + ((1 / pi) * arctanf(z))
        elif df == 2: 
            zo = 0.5 + ((z/2) * ((2+(z**2)) ** (-0.5)))
        elif df > 2: 
            zadj = z* (((4*df) + (z**2) - 1) / ((4*df) + 2 * (z**2)))
            zo = TocherApprox(zadj)
    
    return zo

def TDistroCDF_TwoSide(z, n = None, pyt = False): 
    xo = TDistroCDF(z, n = n, pyt = pyt)
    return (xo - 0.5) ** 2


def PhiError(inp1, inp2, std = 1, 
                    weights = None, mean = True,
                    pseudo = 1e-6, ######### VERY SMALL ERROR
                    log = False, cap = 30,
                    pyt = False, n = None,
                    expo = 2, root = False, weightbefore = False): 
    
    # IN THIS VERSION, THE PSEUDO IS THE LOWEST STD THAT YOU HAVE !!!!!!!!!!!!!!!!!!!!!!!
    # "n" is nuimber of samples for replicates.This is to adjust for the t-distribtion. 
    
    if expo != 2: root = False
    weights = se.WeightsAdjuster(inp1, weights)
    

    e1 = (abs(inp1-inp2)) 
    z = (e1 / (std + pseudo)) + pseudo

    if pyt: z = torch.clamp(z, min = None, max = cap)
    else: z = np.clip(z, a_min = None, a_max = cap)

    e2 = TDistroCDF_TwoSide(z, n = n, pyt = pyt)
    
    e4 = e1 * e2
    
    if log: e4 = np.log(1+e4) #########

    AE = (weights * e4)**expo if weightbefore else weights * (e4 **expo)
    
    if mean is True:  
        AE = AE.mean()
        if root: AE = AE**(1/2)
    
    return AE


#------------------- BETA PRIME STUFF ---------------------

def BetaPrime_Mode(alpha, beta, pseudo = 0):
    mox = (alpha - 1) / (beta + 1)
    bb = alpha <= 1
    mox[bb] = pseudo
    return mox

def Beta2BetaPrime(x, alpha, beta): 
    return ((1+x)**(-alpha-beta)) / ((1-x) ** (beta - 1))

def BetaPrime_PDF_sci(x, alpha, beta): 
    return scipy.stats.betaprime.pdf(x, alpha, beta)

################################

def LogBetaFunction_pyt(alpha, beta):
    #alpha and beta need to be tensors!!!!!!!! 
    return torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)

def BetaFunction_pyt(alpha, beta):
    e = math.e
    return e ** LogBetaFunction_pyt(alpha, beta)

def BetaPrime_PDF_pyt(x, alpha, beta):
 
    # NEED TO WORK IN LOG HERE @@@@@@@@@@@@@@@@@@@@@@

    psu = 1e-10 #THIS IS A TEMPORARY VALUE

    helper_tensor = torch.ones(x.shape)
    helper_tensor[x < 0] = 0

    devo = x.get_device()
    if devo > -1: helper_tensor = helper_tensor.to(devo)

    x2 = (x * helper_tensor) + psu

    p1a = torch.log(x2) * (alpha - 1)
    p1b = torch.log(1 + x2) * (-alpha-beta)
    p2 = LogBetaFunction_pyt(alpha, beta)

    e = math.e

    rex = (p1a + p1b - p2)
    
    ret = e ** rex
    ret = ret * helper_tensor

    return ret

def BetaPrime_Rel(inp, alpha, beta, mo = None, pyt = False, pseudo = 0):

    #---------------------------------------

    if mo is None: mo = BetaPrime_Mode(alpha, beta)

    inp, alpha, beta, mo = [x + pseudo for x in [inp, alpha, beta, mo]]

    #---------------------------------------

    if pyt:
        maxo = BetaPrime_PDF_pyt(mo, alpha = alpha, beta = beta)
        betaprimo= BetaPrime_PDF_pyt(inp, alpha = alpha, beta = beta)

    else: 
        maxo = BetaPrime_PDF_sci(mo, alpha, beta)
        betaprimo= BetaPrime_PDF_sci(inp, alpha, beta)
    
    return betaprimo / maxo

def BetaPrime_CompRel(inp, alpha, beta, mo = None, pyt = False, pseudo = 0):

    dxo = BetaPrime_Rel(inp, alpha, beta, mo = mo, pyt = pyt, pseudo = pseudo)
    
    return 1 - dxo

def CR_BetaPrime(inp1, inp2, var, pyt = False, pseudo = 0): 

    # VAR NEEDS TO BE SOMETHING!!!!!!!!!!!!!!!!!!!!!

    if pyt:

        alpha1 = torch.select(var, dim = -1, index = 0)
        alpha2 = torch.select(var, dim = -1, index = 1)

    else: 

        

        # alpha1 = np.take_along_axis(var, indices = 0, axis = -1)
        # alpha2 = np.take_along_axis(var, indices = 1, axis = -1)

        alpha1 = var[:, :, :, :, 0]
        alpha2 = var[:, :, :, :, 1]
    
    return BetaPrime_CompRel(inp1, alpha = alpha1, beta = alpha2, mo = inp2, 
                             pyt = pyt, pseudo = pseudo)




#------------- DEVIATION ERROR ----------------

def DeviaError(inp1, inp2, std = 1,
               comprel_mode = [CR_BetaPrime, {}],
                   scalefactor = 1, pseudo = 0,
                   pyt = False,
                   weights = None, mean = True,
                   expo = 2, root = False, weightbefore = False, 

                   modif_mode = None, usestd = True, ##############               
                   ): 
    
    # NEW MOD: MVOED PSEUDO FOR COMPREL MODE. 
    # INP1 IS PREDICTION, INP2 IS ACTUAL !!!!!!!!!!!!!! VERY IMPORTANT ORDER. 


    comprel_mode[1].update({'pyt': pyt})

    if pseudo is None: pseudo = 0
    
    if expo != 2: root = False
    weights = WeightsAdjuster(inp1, weights)

    inp1_m, inp2_m = [(inp * scalefactor)
                        for inp in [inp1, inp2]]

    e1 = (abs(inp1_m-inp2_m))

    if modif_mode is not None: 
        inp1_m, inp2_m = [modif_mode[0](inpx, **modif_mode[1]) 
                          for inpx in [inp1_m, inp2_m]]
    
    #--------------------------------------------------

    comprelprob = comprel_mode[0](inp1_m, inp2_m, std, 
                                  pseudo = pseudo,
                                  **comprel_mode[1]) if usestd else 1

    #--------------------------------------------------
    
    
    e4 = comprelprob * e1
    
    AE = (weights * e4)**expo if weightbefore else weights * (e4 **expo)
    
    if mean is True:  
        AE = AE.mean()
        if root: AE = AE**(1/2)
    
    return AE