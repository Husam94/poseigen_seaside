import os
import pickle
import math
import itertools
import numpy as np

from denseweight import DenseWeight

from tqdm import tqdm
import urllib

from scipy.interpolate import interp1d

import concurrent.futures

import torch

####################################################################

def PickleDump(inp, pathname):
    if pathname.endswith('.p') is not True: pathname = pathname + '.p'
    return pickle.dump(inp, open(pathname, 'wb'))

def PickleLoad(pathname):
    if pathname.endswith('.p') is not True: pathname = pathname + '.p'
    return pickle.load(open(pathname, 'rb'))

def ensure_dir(path):
    """Create all directories in the given path if they do not exist."""
    os.makedirs(path, exist_ok=True)
    return path

def NewFolder(pathname, ext=None):
    # Makes a folder if there isn't one and returns the new pathname to use in folder form

    npn = pathname + '_' + ext if ext is not None and pathname is not None else pathname

    if pathname is not None:
        os.makedirs(npn, exist_ok=True)  # Safe for multiprocessing
        npt = npn + '/'
        newpathname = npt if npt.startswith('./') or npt.startswith('/') else './' + npt
    else:
        newpathname = None

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

def Complement(inp, pseudo = False): 
    return 1 - (inp + pseudo)

def RobustScale(inp): 
    q25, q50, q75 = (np.percentile(inp, x) for x in [25, 50, 75])
    return (inp - q50) / (q75 - q25) 

def RobustMean(inp, low=25, high=75, axis=None): 
    qL, qH = np.percentile(inp, [low, high], axis=axis, keepdims=True)
    mask = np.logical_and(inp > qL, inp < qH)  # Use logical_and for element-wise comparison
    return np.mean(mask * inp, axis=axis)

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

def speclog(x, a, b, c, v):
    # M(log(v + b*x)) ≈ a * log(v + b*x) + c
    return a * np.log(v + b * x) + c

def revspeclog(x, a, b, c, v): 
    return (np.exp((x - c) / a) - v) / b


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

def softplus_func(x, b, d, C):
    """
    Softplus curve: ln(C + e^(b*x + d))
    
    Parameters:
    - b: slope parameter
    - d: offset parameter
    - C: baseline constant
    """
    return np.log(C + np.exp(b * x + d))


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


# def WindowVal(inp1, inp2, 
#               mode1 = [np.mean, {}],  mode2 = [np.mean, {}],
#               win_size = 0.1, step = 0.01):
#     # This function slides across taking a window and applying a funciton over it. 
#     # returns (x, y) where x is the center of the window and y is the val on inp2

#     inp1_mi, inp1_ma = MinMax(inp1)
#     totrange = inp1_ma - inp1_mi
#     winx = (win_size * totrange)
#     stepx = step * totrange
    
#     startoz = np.arange(inp1_mi, inp1_ma-winx, stepx)

#     inp1_ws, inp2_ws = [], []
#     for few in startoz: 
#         wer = np.where((inp1 > few) * (inp1 < few + winx))[0]
#         inp1_w, inp2_w = [x[wer] for x in [inp1, inp2]]
#         if mode1 is not None: inp1_w = mode1[0](inp1_w, **mode1[1])
#         if mode2 is not None: inp2_w = mode2[0](inp2_w, **mode2[1])
#         inp1_ws.append(inp1_w)
#         inp2_ws.append(inp2_w)
    
#     return inp1_ws, inp2_ws


def RollingWindows(inp1 = None, win_size = 0.05, step = 0.01, extended = True):

    if inp1 is not None: inp1_mi, inp1_ma = MinMax(inp1)
    else: inp1_mi, inp1_ma = 0, 1
    
    totrange = inp1_ma - inp1_mi
    winx = win_size * totrange
    
    if step is None: 

        # Do Non overlap bins

        num_windows = int(totrange / win_size)
        winx = totrange / num_windows

        startoz = np.linspace(inp1_mi, inp1_ma - winx, num_windows)
        centers = startoz + winx / 2
        endoz = startoz + winx
    
    else:

        stepx = step * totrange

        if extended:
            num_windows = int(np.ceil((inp1_ma - inp1_mi) / stepx)) + 1
            centers = np.linspace(inp1_mi, inp1_ma, num_windows)
            startoz = centers - winx / 2
            endoz = centers + winx / 2
        else:
            startoz = np.arange(inp1_mi, inp1_ma - winx, stepx)
            centers = startoz + winx / 2
            endoz = startoz + winx
    
    startoz[startoz < np.min(inp1)] = np.min(inp1)
    endoz[endoz > np.max(inp1)] = np.max(inp1)

    return centers, startoz, endoz


    




def RollingCount(inp1, startoz, endoz, prop = False): 
    
    lx = len(inp1) if prop else 1
    
    def count_window(st, en):
        return np.sum((inp1 >= st) & (inp1 <= en)) / lx

    with concurrent.futures.ThreadPoolExecutor() as executor:
        counts = list(executor.map(count_window, startoz, endoz))

    return counts


def RollingVal(inp1, inp2, 
              mode =[np.mean, {}],
              win_size=0.1, step=0.01, extended=True):
    """
    Slides across inp1, applies a function over inp2 in each window.
    Returns (centers, f(y)), where centers are window centers.
    If extended=True, windows are centered from min to max, possibly including empty windows.
    Parallelized over windows.
    """

    centers, startoz, endoz = RollingWindows(inp1 = inp1, win_size = win_size, step = step, extended = extended)


    def process_window(st, en):
        wer = np.where((inp1 >= st) & (inp1 <= en))[0]
        inp1_w, inp2_w = [x[wer] for x in [inp1, inp2]]
        if mode is not None:
            inp2_w = mode[0](inp2_w, **mode[1]) if len(inp2_w) > 0 else np.nan
        return inp2_w

    with concurrent.futures.ThreadPoolExecutor() as executor:
        inp2_ws = list(executor.map(process_window, startoz, endoz))

    return np.array(centers), np.array(inp2_ws)

def RollingCount(inp1, startoz, endoz): 
     
    def count_window(st, en):
        return np.sum((inp1 >= st) & (inp1 <= en))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        counts = list(executor.map(count_window, startoz, endoz))

    return counts

def RollingDensity(inp1, win_size=0.1, step=0.01, extended=True, raw = False, 
                   scalefac = None):
    """
    Slides across inp1, counts the number of observations in each window,
    scales by the total number of observations, and fits a linear interpolation.
    Returns the interpolation object: winden.predict(x)
    """
    if scalefac is None: 
        centers, startoz, endoz = RollingWindows(inp1 = inp1, win_size = win_size, 
                                                 step = step, extended = extended)
    else: 
        centers, startoz, endoz = RollingWindows_MT(inp = inp1, win_size = win_size, 
                                                 step = step, extended = extended, 
                                                 scalefac = scalefac)

    counts = RollingCount(inp1, startoz, endoz)
    counts = np.array(counts)

    densities = counts if raw else  counts / len(inp1)
    winden = interp1d(centers, densities, kind='linear', bounds_error=False, fill_value=0.0)
    winden.predict = winden.__call__  # For API consistency

    return winden


def RollingBelong(inp1, win_size=0.1, step=0.01, extended=True):
    """
    For each value in inp1, determines which windows it belongs to.
    Returns an integer array of shape (len(inp1), max_bins), filled with window indices or -1.
    Parallelized over rows.
    """
    centers, startoz, endoz = RollingWindows(inp1 = inp1, win_size = win_size, step = step, extended = extended)

    # Binary mask: shape (N, W)
    inp1_col = inp1.reshape(-1, 1)
    startoz_row = startoz.reshape(1, -1)
    endoz_row = endoz.reshape(1, -1)
    mask = ((inp1_col >= startoz_row) & (inp1_col <= (endoz_row)))

    # Maximum bins a value can belong to
    max_bins = int(np.ceil(win_size / step))

    def get_bins(row):
        bins = np.where(row)[0]
        out = np.full(max_bins, -1, dtype=int)
        out[:len(bins)] = bins
        return out

    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = np.array(list(executor.map(get_bins, mask)), dtype=int)

    return result


def Belong2Rolling(window_belong, return_counts = False):
    """
    Given WindowBelong output (shape (N, max_bins)), returns a list of arrays,
    each containing the indices of inp1 that belong to each window.
    Parallelized over windows.
    """

    is_torch = torch.is_tensor(window_belong)

    num_windows = int(window_belong.max() + 1)

    def get_indices(i):
        if is_torch: return torch.where((window_belong == i).any(dim=1))[0]
        else: return np.where((window_belong == i).any(axis=1))[0]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        indices_list = list(executor.map(get_indices, range(num_windows)))
    
    if return_counts: indices_list = [len(x) for x in indices_list] 

    return indices_list

def RollingScaleFac(inp, mode = [tanh, {}], 
                    win_size = 0.05, step = 0.01, extended = True,
                    minobs = 10, start = 1, increm = 0.1): 
    
    '''Requires inp to be max scaled'''

    if inp.max() > 1: 
        print('FAILED: NOT MAX SCALED')
        return
    inpx = inp.reshape(-1)
    minz = 0
    scalefax = start - increm
    while minz < minobs: 
        scalefax = scalefax + increm
        trax = mode[0](scalefax * inpx, **mode[1])
        centers, startoz, endoz = RollingWindows(inp1 = trax, win_size = win_size, step = step, 
                                                 extended = extended)
    
        
        counts = RollingCount(trax, startoz, endoz)
        print(counts)
        minz = np.min(counts)

    return scalefax



def RollingWindows_MT(inp = None, win_size = 0.05, step = 0.01, extended = True, 
                      scalefac = 1, 
                      mode = [tanh, {}], mode_rev = [revtanh, {}]): 
    
    '''Requires inp to be max scaled'''

    if inp.max() > 1: 
        print('FAILED: NOT MAX SCALED')
        return
    
    trax = mode[0](scalefac * inp, **mode[1])
    centers, startoz, endoz = RollingWindows(inp1 = trax, win_size = win_size, step = step, 
                                                 extended = extended)
    
    centers_rev, startoz_rev, endoz_rev = [mode_rev[0](x, **mode_rev[1]) / scalefac 
                                           for x in [centers, startoz, endoz]]
    
    startoz_rev = np.nan_to_num(startoz_rev, nan = np.min(inp))
    startoz_rev[startoz_rev < np.min(inp)] = np.min(inp)
    
    endoz_rev = np.nan_to_num(endoz_rev, nan = np.max(inp))
    endoz_rev[endoz_rev > np.max(inp)] = np.max(inp)


    return centers_rev, startoz_rev, endoz_rev






#=========================================================================


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





def CosineSimilarity(a, b, eps = 1e-6): 
    return np.dot(a, b)/((np.linalg.norm(a)*np.linalg.norm(b)) + eps)




#----------------------------------------


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

def MultiPairwiseFuncer(inps, mode1=[np.mean, {}], mode2=None):
    """
    For a list of inputs, computes all unique unordered pairs,
    applies mode1 to each pair, and optionally applies mode2 to the results.
    """
    import itertools

    # Get all unique unordered pairs (i < j)
    pairs = list(itertools.combinations(inps, 2))
    # Apply mode1 to each pair
    results = [mode1[0](i, j, **mode1[1]) for i, j in pairs]
    # Optionally apply mode2 to the list of results
    if mode2 is not None:
        results = mode2[0](results, **mode2[1])
    return results


def RelativeChange(a,b, perc = False): 
    # a relative to b
    k = 100 if perc else 1
    return ((a-b) / b) * k










    #https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads

import time

def download_url(url, output_path):
    last_update_time = time.time()
    downloaded = 0

    def reporthook(block_num, block_size, total_size):
        nonlocal last_update_time, downloaded
        downloaded = block_num * block_size
        current_time = time.time()
        if current_time - last_update_time >= 60:  # Print every 60 seconds
            downloaded_gb = downloaded / (1024 ** 3)  # Convert to GB
            total_gb = total_size / (1024 ** 3) if total_size else "Unknown"
            print(f"Downloaded {downloaded_gb:.2f} GB of {total_gb:.2f} GB")
            last_update_time = current_time

    urllib.request.urlretrieve(url, filename=output_path, reporthook=reporthook)





def Standardizer(inp, ddof = 0, psu = 1e-8): 
    mex, stdx = np.mean(inp), np.std(inp, ddof = ddof)
    return (inp - mex) / (stdx + psu)


def Tester():
    print('test')
    return 




def filter_nested_dict(d, wanted_keys_per_level):
    """
    d: nested dict
    wanted_keys_per_level: list of None (for all) or set/list of wanted keys per level
    Example: [None, None, {1,2}, {'a','b'}]
    """
    def _filter(level, subdict):
        if level >= len(wanted_keys_per_level):
            return subdict
        wanted = wanted_keys_per_level[level]
        if wanted is None:
            keys = subdict.keys()
        else:
            keys = [k for k in subdict if k in wanted]
        out = {}
        for k in keys:
            v = subdict[k]
            if isinstance(v, dict):
                out[k] = _filter(level+1, v)
            else:
                out[k] = v
        return out
    return _filter(0, d)