import os
import pickle
import math
import itertools
import numpy as np

from denseweight import DenseWeight

from tqdm import tqdm
import urllib

####################################################################

def PickleDump(inp, pathname):
    if pathname.endswith('.p') is not True: pathname = pathname + '.p'
    return pickle.dump(inp, open(pathname, 'wb'))

def PickleLoad(pathname):
    if pathname.endswith('.p') is not True: pathname = pathname + '.p'
    return pickle.load(open(pathname, 'rb'))

def NewFolder(pathname, ext = None): 
    #Makes a folder if there isnt one and returns the newpathname to use in folder form 

    npn = pathname + '_' + ext if ext is not None and pathname is not None else pathname 
    
    if pathname is not None:
        if os.path.isdir(npn) is False: 
            os.mkdir(npn)
        npt = npn + '/'
        newpathname = npt if npt.startswith('./') or npt.startswith('/') else './' + npt
    else: newpathname = None
        
    return newpathname

def Rounder(x, base=1):
    return base * np.round(x/base).astype(int)

def Round2Int(inp): 
    return np.round(inp).astype(int)

def Ungroup(idx, group): 
    #idx is a list or 1D array of indixes of values 0 to max(group)
    #looks up every idx, returns the group of idxs for each idx
    LG = np.arange(len(group))
    return LG[np.isin(group, idx)]


def MAD(x): 
    return np.mean(np.abs(x-np.mean(x)))


def IQR(inp, q1, q2): 
    qX, qY = (np.percentile(inp, x) for x in [q1, q2])
    return qY-qX

def MinMax(x): return (f(x) for f in [np.min, np.max])

def ListWindower(inp, win_size): 
    return [inp[i: i+win_size] for i in range(len(inp) - win_size + 1)]

def ContinuousWindower(inp1, window): 
    
    li = len(inp1)
    inpx = np.hstack([inp1, inp1[:window - 1]])
    
    return np.lib.stride_tricks.sliding_window_view(inpx, window)

def Cutter(inp, idx): 
    li = len(inp)
    return np.hstack([inp[idx:], inp[:idx]])

def Reciprocal(inp, pseudo = False): 
    return 1/(inp+pseudo)

def RobustScale(inp): 
    q25, q50, q75 = (np.percentile(inp, x) for x in [25, 50, 75])
    return (inp - q50) / (q75 - q25) 

def RobustMean(inp, low = 25, high = 75): 
    qL, qH = (np.percentile(inp, x) for x in [low, high])
    return np.mean(inp[(inp > qL) * (inp < qH)])

def RobustStd(inp, low = 25, high = 75, ddof = 0): 
    qL, qH = (np.percentile(inp, x) for x in [low, high])
    return np.std(inp[(inp > qL) * (inp < qH)], ddof = ddof)



def MiMa(inp):
    return [np.nanmin(inp), np.nanmax(inp)] 


def Exponential(x, m = 1, b = 1, c = 0, t = 0):
    return m * np.exp(b * (x - t)) + c

def tanh(inp, base = math.e, a = 1):
    return (2 / (1 + (base ** (-2*a*inp)))) - 1

def revtanh(inp, base = math.e, a = 1): 
    return (np.log((2 / (inp + 1)) - 1) / np.log(base)) / (-2 * a)


def log(inp, base = math.e, pseudo = None): 
    ps = 0 if pseudo == None else pseudo
    return np.log(inp + ps)/np.log(base)

def revlog(inp, base = math.e, pseudo = None):
    ps = 0 if pseudo == None else pseudo
    return (base ** inp) - pseudo


def logmodules(inp, base = math.e):
    inp = np.array(inp)
    inp[inp>0] = np.log(1+inp[inp>0])
    inp[inp<0] = -np.log(1 + (-inp[inp<0]))
    return inp

def revlogmodules(inp, base = math.e):
    inp = np.array(inp)
    inp[inp>0] = math.e**(inp[inp>0]) - 1
    inp[inp<0] = 1 - math.e**(-(inp[inp<0]))

    return inp


def SlidingWindowMean(inpx, inpy, winsize = 0.1, reso = 0.001): 
    # creatres windows that are winsize* range of inpx, then gathers values in each window and gets mean. 
    
    inpx_mi, inpx_ma = MinMax(inpx)
    inpx_totrange = inpx_ma - inpx_mi
    win = winsize * inpx_totrange
    rez = reso * inpx_totrange

    print(win, rez)

    mex = []
    winstarts = np.arange(inpx_mi, inpx_ma - win, step = rez)
    winmids = winstarts + (win / 2)
    for r in winstarts: 
        st, en = r, r + win
        
        mw = np.mean(inpy[(inpx >= st) * (inpx <= en)])
        mex.append(mw)
    
    return winmids, np.array(mex)


def WindowVal(inp1, inp2, 
              mode1 = [np.mean, {}],  mode2 = [np.mean, {}],
              win_size = 0.1, step = 0.01):
    # This function slides across taking a window and applying a funciton over it. 
    # returns (x, y) where x is the center of the window and y is the val on inp2

    inp1_mi, inp1_ma = MinMax(inp1)
    totrange = inp1_ma - inp1_mi
    winx = (win_size * totrange)
    stepx = step * totrange
    
    startoz = np.arange(inp1_mi, inp1_ma-winx, stepx)

    inp1_ws, inp2_ws = [], []
    for few in startoz: 
        wer = np.where((inp1 > few) * (inp1 < few + winx))[0]
        inp1_w, inp2_w = [x[wer] for x in [inp1, inp2]]
        if mode1 is not None: inp1_w = mode1[0](inp1_w, **mode1[1])
        if mode2 is not None: inp2_w = mode2[0](inp2_w, **mode2[1])
        inp1_ws.append(inp1_w)
        inp2_ws.append(inp2_w)
    
    return inp1_ws, inp2_ws



def Epsilon(inp, expo = 2, gamma = 1, smallest = True):
    #inp has to be numpy array or tensor. 
    me = inp.mean()
    
    eoe = abs(inp - me)**expo
    
    sigma = (eoe.mean())**(1/expo)
    
    sigamma = sigma * gamma
    
    return me + sigamma if smallest else me - sigamma





def RevDistro(vals, select, alpha = 1.0): 
    
    dw = DenseWeight(alpha=alpha)
    weights = dw.fit(vals)
    weights = weights / np.sum(weights)

    idx = np.random.choice(np.arange(len(vals)), size = select, replace = False, p = weights)
    
    return idx


def DenseWeighter(inp, onlyidx = None, 
                  alpha = 1.0, multi = False, newrange = (0.01, 1)): 
    
    if multi is False: inp = [inp]
    ns = []
    
    for n in inp: 
        nx = n.shape
        bine = n if onlyidx is None else n[onlyidx]

        dw = DenseWeight(alpha = alpha)
        dw.fit(bine.reshape(-1, 1))
        wo = dw.eval(n.reshape(-1)).reshape(*nx)

        if newrange == True: 
            wo = wo / wo.max()
            
        ns.append(wo)

    ns = np.stack(ns) if multi else ns[0]
    
    if newrange is not None: 
        if isinstance(newrange, tuple): 
            ns = MinMaxNormalizer(ns, multi = multi, newrange = newrange)
    
    return np.array(ns)


def GeomNumSpacing(start, end, num, plier):
    #This == GeomSpacing_v2
    if plier == 1: numspace = np.linspace(start, end, num)
    else: 
        a = ((end - start) * (1-plier)) / (1 - (plier ** (num-1)))
        numspace = np.array([start + (a * (1 - plier ** n) / (1 - plier)) for n in range(num)])
    return numspace



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



def CosineSimilarity(a, b, eps = 1e-6): 
    return np.dot(a, b)/((np.linalg.norm(a)*np.linalg.norm(b)) + eps)



###################################################################################



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


###################################################################################




















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


def SubSample_Random(X, weights = False, proportion = 0.3, num_subsamples = 1, group = None):
    
    p = X / np.sum(X) if weights else None 

    if group is not None: 
        uni = np.unique(group)
        li = len(uni)
    else: 
        li = X if isinstance(X, int) else len(X)
        uni = np.arange(li) 
        
    if proportion < 1: proportion = Round2Int(proportion * li)
    
    ss = [np.random.choice(uni, proportion, replace = False, p = p) for _ in range(num_subsamples)]
    
    if group is not None: ss = [Ungroup(s, group) for s in ss]
    
    if num_subsamples == 1: ss = ss[0]
    
    return ss 

def SubSample_Select(inp, proportion = 0.3, num_subsamples = 1, select_mode = [RevDistro, {}]):
    
    #inp should be values here 
    if isinstance(inp, int): inp = np.arange(inp)
    
    li = len(inp) 
    if proportion < 1: proportion = Round2Int(proportion * li)
    
    ss = [select_mode[0](inp, proportion, **select_mode[1]) for _ in range(num_subsamples)]
    
    if num_subsamples == 1: ss = ss[0]
    return ss







def Normalizer(inp, axis = None, reciprocal = False, pseudo = True): 

    inp = np.array(inp)

    if reciprocal: 
        if inp.any(0): inp = inp + pseudo

    sums = np.expand_dims(np.sum(inp, axis = axis), axis) if axis is not None else np.sum(inp)
    nem = inp/sums

    if reciprocal:
        inp = Reciprocal(nem)
        sums = np.expand_dims(np.sum(inp, axis = axis), axis) if axis is not None else np.sum(inp)
        nem = inp/sums

    return nem

def Scaler(inp): 
    return (inp - np.min(inp)) / (np.max(inp) - np.min(inp))

def MinkDistance(inp1, inp2 = None, p = 2):
    if inp2 is None: inp2 = 0
    return (np.sum(np.abs(inp1-inp2)**p))**(1/p)


def NormMinkDistance(inp1, inp2 = None, p = 2, norm_axis = None): 
    #p = 1, 2 is manhattan, euclidean, respectively 
    inp1 = Normalizer(inp1, axis = norm_axis)
    if inp2 is not None: inp2 = Normalizer(inp2, axis = norm_axis)
    #The sum of each vector is 1. 
    return MinkDistance(inp1,inp2, p = p)

def ScaledNormMinkDistance(inp1, inp2 = None, p = 2): 

    #inps need to be vectors with sum of 1. 

    #max of a vector with sum of 1 is 2^p^(1/p) 
    maxM = 2
    
    return NormMinkDistance(inp1, inp2, p = p) / maxM

def NormalizedDistances(inpA, inpB = None, 
                           onlyidx = None, p = 2, 
                           reciprocal = False, pseudo = True, 
                           weight_bymem = False, 
                           summarize_mode = [Epsilon, {}]): 
    
    #inpA is a list OR an array where the ROWS (INDEX 0) are each A 

    norm_args = {'reciprocal': reciprocal, 'pseudo': pseudo}
    
    #START HERE BY NORMALIZING EACH INP BY ITSELF 
    if isinstance(onlyidx, list): inpA = [A[onlyidx] for A in inpA] 
    inpNA = [Normalizer(A, **norm_args) for A in inpA]
        
    #If inpB is None, then you go into PAIRWISE. If not you go into comparing with inp B 
    
    if inpB is None: 
        combs = list(itertools.combinations(np.arange(len(inpA)), 2))
        inpX = [[inpNA[x] for x in c] for c in combs] # Now this is the list of combinations 
    
    if inpB is not None:
        if isinstance(onlyidx, list): inpB = inpB[onlyidx]
        inpB = Normalizer(inpB, **norm_args)    
        inpX = [[A, inpB] for A in inpNA]
        
    #inpX is a list of lists where each list is a pair
    
    inpD = np.array([ScaledNormMinkDistance(*X, p = p) for X in inpX])

    if weight_bymem is not False: 

        recip_bm = True if weight_bymem == -1 else False

        Asums = [np.sum(X[0]) for X in inpX]
        Aprops = Normalizer(Asums, reciprocal = recip_bm, pseudo = True)

        inpD = inpD * Aprops
    
    if summarize_mode is not None: 
        inpD = summarize_mode[0](inpD, **summarize_mode[1]) 

    return inpD  







def UniqueNestedDict(inp, keepkey = False): 

    #returns a unique nested dictionary, good for cangen. 

    uniquedict = {}
    for key, val in inp.items(): 

        lud = len(uniquedict.keys())
        ko = key if keepkey else lud

        if len(uniquedict.keys()) > 0: 
            hg = 0
            for k,v in uniquedict.items(): 
                if v == val: 
                    hg = 1
                    continue
            
            if hg == 0: uniquedict[ko] = val
        
        else: 

            uniquedict[ko] = val

    return uniquedict



def PairwiseFuncer(inp1, inp2, mode1 = [np.mean, {}], mode2 = None):

    pfs = [mode1[0](*[i1, i2], **mode1[1]) for i2 in inp2 for i1 in inp1]
    if mode2 is not None: pfs = mode2[0](pfs, **mode2[1])

    return pfs



def RelativeChange(a,b, perc = False): 
    # a relative to b
    k = 100 if perc else 1
    return ((a-b) / b) * k










    #https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)





def Standardizer(inp, ddof = 0, psu = 1e-8): 
    mex, stdx = np.mean(inp), np.std(inp, ddof = ddof)
    return (inp - mex) / (stdx + psu)


