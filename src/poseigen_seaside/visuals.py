
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import poseigen_seaside.basics as se


linep_args = {'fill': False, 'fillbetween': None, 'hatch': '/', 
                'bounds': None, 'boundspad': None, 
                'legend': None, 'legend_title': None,
                'color': None, 'cmap': 'viridis', 'alpha': 1.0, 'hline': None, 
                'xlabel': None, 'ylabel': None, 'suptitle': None, 'text': None, 
                'xticklabel': None, 
                'figsize': (5,5), 'fontsize': 10, 'ax': None}

def LinePlot(inp, 
             fill = False, fillbetween = None, hatch = '/', 
                bounds = None, boundspad = None, 
                legend = None, legend_title = None,
                color = None, cmap = 'viridis', alpha = 1.0, hline = None, 
                xlabel = None, ylabel = None, suptitle = None, text = None, 
                xticklabel = None, 
                figsize = (5,5), fontsize = 10, ax = None): 
    
    plt.rc('font', size=fontsize)
    titlesize = int(fontsize + fontsize * 0.2)
    if ax is None: fig, ax = plt.subplots(figsize = figsize)
    
    if isinstance(inp, list) == False and len(inp.shape) < 2: 
        inp = [inp]
        if color is not None and isinstance(color, list) == False: color = [color]
    li = len(inp)
    lx = len(inp[0])
    
    h = 1 if hline is not None else 0
    fb = 1 if fillbetween is not None else 0
    if color is None: 
        color = [plt.get_cmap(cmap, li + h + fb)(i) for i in range(li + h + fb)]

    for ik,k in enumerate(inp):
        if isinstance(k, list): ax.plot(*k, alpha = alpha, color = color[ik])
        else: ax.plot(k, alpha = alpha, color = color[ik])
        if fill is not False: ax.fill_between(np.arange(lx), fill, k, color = color[ik], alpha = alpha)
    
    if fillbetween is not None: 
        al = fillbetween if isinstance(fillbetween, float) else 0.5
        ax.fill_between(np.arange(lx), *inp, color = color[-1], alpha = al, hatch = hatch)
    
    if bounds is not None: 
        if isinstance(bounds, int): bounds = (0, bounds)
        ax.set_ylim(bounds)
    
    if hline is not None: ax.axhline(hline, ls = '--', color = color[len(inp)])
    
    if xlabel is not None: ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None: ax.set_ylabel(ylabel, fontsize=fontsize)
    
    if xticklabel is not None: 
        ax.set_xticks(np.linspace(0, lx-1, len(xticklabel)), labels = [])
        ax.set_xticklabels(xticklabel, fontsize = fontsize)
    
    if legend is not None: ax.legend(legend, title = legend_title)
        
    
    if text is not None: ax.text(*text)
    if suptitle is not None: ax.set_title(suptitle, size = titlesize, fontweight='bold')
    
    return ax

def GridPlot(inp, bounds = (None, None), boundspad = None, 
                cmap = 'viridis', alpha = 1.0, colorbar = False, 
                xlabel = None, ylabel = None, suptitle = None, 
                xticklabel = None, yticklabel = None, 
                xtick_rotation = None, ytick_rotation = None, 
                figsize = (5,5), fontsize = 10, ax = None): 
    
    plt.rc('font', size=fontsize)
    titlesize = int(fontsize + fontsize * 0.2)
    if ax is None: fig, ax = plt.subplots(figsize = figsize)
    
    if boundspad is not None and bounds is not None: 
        bounds = [bounds[0] + boundspad, bounds[1] - boundspad]
    
    g = ax.imshow(inp, vmin = bounds[0], vmax = bounds[1], cmap = cmap)
    
    if xticklabel is not None: 
        ax.set_xticks(np.linspace(0, inp.shape[1]-1, len(xticklabel)))

        ax.set_xticklabels(xticklabel, fontsize = fontsize, rotation = xtick_rotation)
    
    if yticklabel is not None: 
        ax.set_yticks(np.linspace(0, inp.shape[0]-1, len(yticklabel)))
        ax.set_yticklabels(yticklabel, fontsize = fontsize, rotation = ytick_rotation)
    
    if colorbar == True: plt.colorbar(g, ax=ax) 
    
    if xlabel is not None: ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None: ax.set_ylabel(ylabel, fontsize=fontsize)
    
    if suptitle is not None: ax.set_title(suptitle, size = titlesize, fontweight='bold')
    
    return g

def ScaHistPlot(inp, bins = 50,
                switch = False,
                bounds = (None, None), sharebounds = False, boundspad = None, 
                pointsize = 10, color = 'maroon', alpha = 0.5, 
                line = False, linewidth = 1.0, 
                xlabel = None, ylabel = None, suptitle = None, text = None,
                figsize = (5,5), fontsize = 10, ax = None): 
    
    #inp are [x,y]
    x, y = (1, 0) if switch else (0, 1)
    if bounds is None: bounds = [None, None]
    if isinstance(bounds, tuple): list(bounds)
    
    plt.rc('font', size=fontsize) 
    titlesize = int(fontsize + fontsize * 0.2)
    if ax is None: fig, ax = plt.subplots(figsize = figsize)
    
    if sharebounds == True: 
        combinp = np.stack(inp).flatten()
        xbounds, ybounds = ([f(combinp) for f in [np.min, np.max]] for j in inp[:2])
    else: 
        xbounds, ybounds = ([f(j) for f in [np.min, np.max]] for j in inp[:2])
    for ib, b in enumerate(bounds): 
        if b is not None: 
            xbounds[ib], ybounds[ib] = bounds[ib], bounds[ib]
    
    if boundspad is not None:
        bpx, bpy = (boundspad * (g[1] - g[0]) for g in [xbounds, ybounds])
        xbounds, ybounds = ([g[0] - j, g[1] + j] for g,j in zip([xbounds, ybounds], [bpx, bpy]))
    
    w = 1 
    if len(inp) > 2: 
        wmin, wmax = (f(inp[2]) for f in [np.min, np.max])
        w = (inp[2] - wmin) / (wmax-wmin)
    s = w*pointsize if isinstance(pointsize, int) else pointsize[0] + pointsize[1] * w
    
    
    ax.scatter(inp[x], inp[y], alpha = alpha, c = color, s = s)
    #ax.set_aspect(1.)
    if xlabel is not None: ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None: ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlim(xbounds)
    ax.set_ylim(ybounds)
    
    divider = make_axes_locatable(ax)
    ax_histx = divider.append_axes("top", pad = '5%', size = '15%', sharex=ax)
    ax_histy = divider.append_axes("right", pad = '5%', size = '15%', sharey=ax)

    ax_histx.xaxis.set_tick_params(labelbottom=False)
    ax_histy.yaxis.set_tick_params(labelleft=False)
    
    xbins, ybins = (np.linspace(g[0], g[1], bins+1) for g in [xbounds, ybounds])
    
    histmax = np.max([np.histogram(w, bins = bn)[0] for w, bn in zip(inp[:2], [xbins, ybins])])
    
    ax_histx.hist(inp[x], bins=xbins, alpha = alpha, color = color)
    ax_histy.hist(inp[y], bins=ybins, orientation='horizontal', alpha = alpha, color = color)
    ax_histx.set_ylim(0, histmax)
    ax_histy.set_xlim(0, histmax)
    
    if line == True: ax.plot(inp[x], inp[x], 'm', linewidth = linewidth)       
    
    if text is not None: ax.text(*text)
    
    if suptitle is not None: ax_histx.set_title(suptitle, size = titlesize, fontweight='bold')
        
    return 

def ScaPlot(inp, switch = False, 
            bounds = None, sharebounds = False, boundspad = None,
            pointsize = 10, color = 'maroon', alpha = 0.5, line = False, 
            xlabel = None, ylabel = None, suptitle = None, text = None,
            figsize = (5,5), fontsize = 10, ax = None): 
    
    #bounds is only list from now on #jan 5 24
    
    x, y = (1, 0) if switch else (0, 1)
    
    plt.rc('font', size=fontsize) 
    titlesize = int(fontsize + fontsize * 0.2)
    if ax is None: fig, ax = plt.subplots(figsize = figsize)
    
    if bounds is None: 
        if sharebounds == True: 
            combinp = np.stack(inp).flatten()
            xbounds, ybounds = ([f(combinp) for f in [np.min, np.max]] for j in inp[:2])
        else: 
            xbounds, ybounds = ([f(j) for f in [np.min, np.max]] for j in inp[:2])
    
    else: 

        if isinstance(bounds, int) or isinstance(bounds, float): 
            bounds = [0, bounds]
        if isinstance(bounds[0], list) is False: 
            bounds = [bounds, bounds]
        xbounds, ybounds = bounds


    #     if sharebounds: 


    #     xbounds, ybounds = None, None
    
    # else: 
    
    #     if isinstance(bounds, tuple): list(bounds)

    #     if sharebounds == True: 
    #         combinp = np.stack(inp).flatten()
    #         xbounds, ybounds = ([f(combinp) for f in [np.min, np.max]] for j in inp[:2])

    #     else: 
    #         xbounds, ybounds = ([f(j) for f in [np.min, np.max]] for j in inp[:2])
        
    #     for ib, b in enumerate(bounds): 
    #         if b is not None: 
    #             xbounds[ib], ybounds[ib] = bounds[ib], bounds[ib]




    if boundspad is not None:
        bpx, bpy = (boundspad * (g[1] - g[0]) for g in [xbounds, ybounds])
        xbounds, ybounds = ([g[0] - j, g[1] + j] for g,j in zip([xbounds, ybounds], [bpx, bpy]))
        
    w = 1 
    if len(inp) > 2: 
        wmin, wmax = (f(inp[2]) for f in [np.min, np.max])
        w = (inp[2] - wmin) / (wmax-wmin)
    s = w*pointsize if isinstance(pointsize, int) else pointsize[0] + pointsize[1] * w


    
    ax.scatter(inp[x], inp[y], alpha = alpha, c = np.array([color]), s = s)
    #ax.set_aspect(1.)
    if xlabel is not None: ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None: ax.set_ylabel(ylabel, fontsize=fontsize)
    if xbounds is not None: ax.set_xlim(*xbounds)
    if ybounds is not None: ax.set_ylim(*ybounds)
    
    if line == True: ax.plot(inp[0], inp[0], 'm')       
    
    if text is not None: ax.text(*text)
    
    if suptitle is not None: ax.set_title(suptitle, size = titlesize, fontweight='bold')
        
    return  ax

def HistPlot(inp, 
             bins = 50, density = False, proportion = False, 
             
             histtype = 'stepfilled', 
             bounds = (None, None), boundspad = None,
             color = 'maroon', alpha = 0.5, hline = None, vline = None, 
             xlabel = None, ylabel = None, suptitle = None, text = None,
             figsize = (5,5), fontsize = 10, ax = None): 
    
    if bounds is None: bounds = [None, None]
    if isinstance(bounds, tuple): list(bounds)
    
    plt.rc('font', size=fontsize) 
    titlesize = int(fontsize + fontsize * 0.2)
    if ax is None: fig, ax = plt.subplots(figsize = figsize)

    weights = np.ones(len(inp)) / len(inp)  if proportion else None
    
    xbounds = [f(inp) for f in [np.min, np.max]]
    
    for ib, b in enumerate(bounds): 
        if b is not None: xbounds[ib] = bounds[ib]

    if boundspad is not None:
        bpx = boundspad * (xbounds[1] - xbounds[0])
        xbounds = [xbounds[0] - bpx, xbounds[1] + bpx]
        
    bins = np.linspace(xbounds[0], xbounds[1], bins + 1)

    ax.hist(inp, bins = bins, alpha = alpha, color = color, density = density, 
            histtype = histtype, weights = weights)
    
    if xlabel is not None: ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None: ax.set_ylabel(ylabel, fontsize=fontsize)
    
    ax.set_xlim(xbounds)
    
    if vline is not None: ax.axvline(vline, ls = '--', color = color)
    if hline is not None: ax.axhline(hline, ls = '--', color = color)
    
    if text is not None: ax.text(*text)
    
    if suptitle is not None: ax.set_title(suptitle, size = titlesize, fontweight='bold')
 



#Jan 6 addition

def BarPlot(inp, width = 0.8, bounds = None, skip = False, 
            color = 'maroon', alpha = 1, hline = None,
            label = None, labelheight = False, 

            yerr = None, 

            group = None, legend = False, 

            empty = '//', 
            xlabel = None, ylabel = None, suptitle = None, text = None, xtick_rotation = None,
            figsize = (5,5), fontsize = 10, ax = None): 
    

    #Mar 10 24, added group for grouped barplot. 

    cla, val1, = inp
    x = cla if skip else np.arange(len(cla))
    inp1 = [x, val1]
    
    plt.rc('font', size=fontsize) 
    titlesize = int(fontsize + fontsize * 0.2)
    if ax is None: fig, ax = plt.subplots(figsize = figsize)

    if group is None: 

        rects = ax.bar(*inp1, width = width, alpha = alpha, color = color, label = label, yerr = yerr)

        if empty:
            emps = [ig for ig, g in enumerate(val1) if np.isnan(g) or g is None]
            for e in emps: 
                ax.text(x=e, y=0, s=empty, ha='center', va='center', fontsize=fontsize)
            
    else: 

        if color is not None: 
            if isinstance(color, list) is False: 
                color = [color] 
        else: color = [None] * len(val1)

        if yerr is not None: 
            if isinstance(yerr, list) is False: yerr = [yerr]
    
        for ib, b in enumerate(val1):

            attr = group[ib] if isinstance(group, list) else None

            yex = yerr[ib] if yerr is not None else None

            offset = width * ib
            rects = ax.bar(x + offset, b, width = width, 
                           alpha = alpha, color = color[ib], label = attr, 
                           yerr = yex)
    

    if isinstance(group, list) and legend: ax.legend()

    offy = ((len(val1) / 2) - 0.5) * width if group is not None else 0
    ax.set_xticks(x + offy, [str(e) for e in cla], rotation = xtick_rotation)
    
    if xlabel is not None: ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None: ax.set_ylabel(ylabel, fontsize=fontsize)
    
    if labelheight is True: ax.bar_label(rects, padding = 1)

    if hline is not None: ax.axhline(hline, ls = '--', color = color)
    
    if bounds is not None: 
        ax.set_ylim(*bounds) if isinstance(bounds, tuple) else ax.set_ylim(0, bounds)
    
    if text is not None: ax.text(*text)
    
    if suptitle is not None: ax.set_title(suptitle, size = titlesize, fontweight='bold')
        
    return ax

def PairedBarPlot(inp, width = 0.35, 
                  bounds = None, ylabels = None, colors = None,  
                  cmap = 'viridis', bp_args = {}, legend = False, xtick_rotation = None, 
                  figsize = (10,5), fontsize = 10, ax = None, 
                  hline = None, linewidth = 0.5, linestyle = '--'): 
    
    #input is [class, values1, values2] 
    #bounds and ylabels are lists. 

    if hline is not None and isinstance(hline, list) == False: hline = [hline]
    
    cla, val1, val2 = inp
    x = np.arange(len(cla))
    h = len(hline) if hline is not None else 0
    if colors is None: colors = [plt.get_cmap(cmap, 2 + h)(i) for i in range(2 + h)]
    
    plt.rc('font', size=fontsize) 
    
    if ax is None: fig, ax = plt.subplots(figsize = figsize)
    
    inp1, inp2 = [[x-width/2, val1], [x+width/2, val2]]
    
    bp_args1, bp_args2 = bp_args.copy(), bp_args.copy()
    if ylabels is not None: 
        bp_args1['ylabel'], bp_args2['ylabel'] = ylabels
        bp_args1['label'], bp_args2['label'] = ylabels
    if bounds is not None: bp_args1['bounds'], bp_args2['bounds'] = bounds
        
    rects1 = BarPlot(inp1, width, color = colors[0], ax = ax, skip = True, **bp_args1)
    ax.set_xticks(x, [str(e) for e in cla], rotation = xtick_rotation)

    if hline is not None: 

        for ih, hl in enumerate(hline):
            ax.axhline(hl, ls = linestyle, color = colors[-1-ih], linewidth = linewidth)
    
    axR = ax.twinx()
    rects2 = BarPlot(inp2, width, color = colors[1], ax = axR, skip = True, **bp_args2)
    axR.set_xticks(x, [str(e) for e in cla], rotation = xtick_rotation)
    
    if ylabels is not None and legend is True: 
        ax.set_ylabel(ylabels[0]) 
        axR.set_ylabel(ylabels[1]) 
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = axR.get_legend_handles_labels()
        axR.legend(lines + lines2, labels + labels2, loc=0)
           
    #fig.tight_layout() 
    
    return  


import matplotlib 


######################################################################

def simple_beeswarm2(y, nbins=None, width=1.):
    """
    Returns x coordinates for the points in ``y``, so that plotting ``x`` and
    ``y`` results in a bee swarm plot.
    """
    y = np.asarray(y)
    if nbins is None:
        # nbins = len(y) // 6
        nbins = np.ceil(len(y) / 6).astype(int)

    # Get upper bounds of bins
    x = np.zeros(len(y))

    nn, ybins = np.histogram(y, bins=nbins)
    nmax = nn.max()

    #Divide indices into bins
    ibs = []#np.nonzero((y>=ybins[0])*(y<=ybins[1]))[0]]
    for ymin, ymax in zip(ybins[:-1], ybins[1:]):
        i = np.nonzero((y>ymin)*(y<=ymax))[0]
        ibs.append(i)

    # Assign x indices
    dx = width / (nmax // 2)
    for i in ibs:
        yy = y[i]
        if len(i) > 1:
            j = len(i) % 2
            i = i[np.argsort(yy)]
            a = i[j::2]
            b = i[j+1::2]
            x[a] = (0.5 + j / 3 + np.arange(len(b))) * dx
            x[b] = (0.5 + j / 3 + np.arange(len(b))) * -dx

    return x

def BoxPlot(inp, 
            widths = 0.5,
            colors = None, cmap = None, alpha = 1.0, 
            notch = False, 
            connect = None, #connect is a tuple or a list of 2 where the first pos is the positons to connect, the other is text. 
            label = None, table = None, 
            xlabel = None, ylabel = None, suptitle = None, 
            
            addmarks = None, addmarks_args = {}, bounds = None, 

            swarm = False, swarm_ps = 10, swarm_alpha = 1.0, swarm_bins = 10,
            
            figsize = (5,5), fontsize = 10, ax = None): 

    plt.rc('font', size=fontsize) 
    titlesize = fontsize #int(fontsize + fontsize * 0.2)
    if ax is None: fig, ax = plt.subplots(figsize = figsize)

    li = len(inp)
    

    pa = True if colors is not None or cmap is not None else False
    
    if colors is not None:
        if isinstance(colors, list) is False: colors = [colors]
        colors = [matplotlib.colors.to_rgba(c) for c in colors]
        if len(colors) == 1: colors = colors * li
        
    if cmap is not None:
        colors = [plt.get_cmap(cmap, li)(i) for i in range(li)]
            
    bplab = label if table is None else None

    bplot = ax.boxplot(inp, 
                       patch_artist = pa, 
                       notch = notch, 
                       labels = bplab, widths = widths)

    if colors is not None: 
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(alpha)
        
    bplot = ax.boxplot(inp, 
                       patch_artist = False, 
                       notch = notch, 
                       labels = bplab, widths = widths)
        
    for patch in bplot['medians']: patch.set(color = 'black')

    
    if table is not None:
        ax.table(table,
                 cellLoc = 'center', 
                 rowLabels = label, fontsize = fontsize)
        ax.set_xticks([])


    if connect is not None: 
        mi, ma = (f(inp) for f in [np.min, np.max])
        space1, space2 = 0.025 * (ma-mi), 0.05 * (ma-mi)
        j,k = ma + space1, ma + space2
        pos1 = connect[0][0] + 1
        for ix, x in enumerate(connect[0]):
            if ix != 0: 
                pos2 = x + 1
                ax.plot([pos1, pos1, pos2, pos2], [j, k, k, j], lw=1.5, color = 'black')
                if ix == len(connect[0]) - 1: 
                    ax.text((pos1+pos2) / 2, k, connect[1], ha='center', va='bottom', color = 'black') 

    if xlabel is not None: ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None: ax.set_ylabel(ylabel, fontsize=fontsize)

    if addmarks is not None: ax.scatter(np.arange(1, li + 1), addmarks, **addmarks_args, zorder = 2)

    if suptitle is not None: ax.set_title(suptitle, size = titlesize, fontweight='bold')

    if bounds is not None: 
        ax.set_ylim(*bounds) if isinstance(bounds, tuple) else ax.set_ylim(0, bounds)

    if swarm:

        for ix, x in enumerate(inp):
            col = 'black' #colors[ix] if colors is not None else 'black'
            xo = simple_beeswarm2(x, width = widths*0.5, nbins=swarm_bins)
            ax.scatter(xo+1.+ix, x, s=swarm_ps, alpha = swarm_alpha, c = col, zorder = 2)
            
    return 

def reject_outliers(data, m=2, bystd = True): 
    #if bystd: by standard dev or by median
    if bystd: 
        noouts = data[abs(data - np.mean(data)) < m * np.std(data)]
    else: 
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d/mdev if mdev else np.zeros(len(d))
        noouts = data[s<m]

    return noouts

def ViolinPlot(inp, 
               no_outliers = False, rej_outliers = 2., 
            widths = 0.5,
            colors = None, cmap = None, alpha = 1.0, 
            connect = None, #connect is a tuple or a list of 2 where the first pos is the positons to connect, the other is text. 
            label = None, table = None, 
            xlabel = None, ylabel = None, suptitle = None, 
            
            addmarks = None, addmarks_args = {}, bounds = None, 

            swarm = False, swarm_ps = 10, swarm_alpha = 1.0, swarm_bins = 10,
            
            figsize = (5,5), fontsize = 10, ax = None): 

    plt.rc('font', size=fontsize) 
    titlesize = fontsize #int(fontsize + fontsize * 0.2)
    if ax is None: fig, ax = plt.subplots(figsize = figsize)

    li = len(inp)


    if colors is not None:
        if isinstance(colors, list) is False: colors = [colors]
        colors = [matplotlib.colors.to_rgba(c) for c in colors]
        if len(colors) == 1: colors = colors * li
        
    if cmap is not None:
        colors = [plt.get_cmap(cmap, li)(i) for i in range(li)]

    if swarm is False: no_outliers = False

    inaq = [reject_outliers(np.array(r), m=rej_outliers) for r in inp] if no_outliers else inp

    parts = ax.violinplot(inaq, widths = widths, showextrema=False, showmeans = True)

    for ipc, pc in enumerate(parts['bodies']):
        if colors is not None: pc.set_facecolor(colors[ipc])
        pc.set_edgecolors('black')
        pc.set_edgecolor('black')
        pc.set_alpha(alpha)
    parts['cmeans'].set_colors('black')
    parts['cmeans'].set_linewidth(3)


    if label is not None: 

        ax.set_xticks(np.arange(1, len(label) + 1), labels=label)
        ax.set_xlim(0.25, len(label) + 0.75)
    
    if table is not None:
        ax.table(table,
                 cellLoc = 'center', 
                 rowLabels = label, fontsize = fontsize)
        ax.set_xticks([])


    if connect is not None: 
        mi, ma = (f(inaq) for f in [np.min, np.max])
        space1, space2 = 0.025 * (ma-mi), 0.05 * (ma-mi)
        j,k = ma + space1, ma + space2
        pos1 = connect[0][0] + 1
        for ix, x in enumerate(connect[0]):
            if ix != 0: 
                pos2 = x + 1
                ax.plot([pos1, pos1, pos2, pos2], [j, k, k, j], lw=1.5, color = 'black')
                if ix == len(connect[0]) - 1: 
                    ax.text((pos1+pos2) / 2, k, connect[1], ha='center', va='bottom', color = 'black') 

    if xlabel is not None: ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None: ax.set_ylabel(ylabel, fontsize=fontsize)

    if addmarks is not None: ax.scatter(np.arange(1, li + 1), addmarks, **addmarks_args, zorder = 2)

    if suptitle is not None: ax.set_title(suptitle, size = titlesize, fontweight='bold')

    if bounds is not None: 
        ax.set_ylim(*bounds) if isinstance(bounds, tuple) else ax.set_ylim(0, bounds)

    
    if swarm:

        for ix, x in enumerate(inp):
            col = 'black' #colors[ix] if colors is not None else 'black'
            xo = simple_beeswarm2(x, width = widths*0.5, nbins=swarm_bins)
            ax.scatter(xo+1.+ix, x, s=swarm_ps, alpha = swarm_alpha, c = col, zorder = 2)       
            
    return 


######################################################################



def ECDFPlot(inp,
             complementary = False, ylim = 1.05, 

             color = 'maroon', alpha = 0.5, linestyle = '-', 
             hline = None, vline = None, 
             xlabel = None, ylabel = True, 
             suptitle = None, text = None,
             figsize = (5,5), fontsize = 10, ax = None): 
       
    plt.rc('font', size=fontsize) 
    titlesize = int(fontsize + fontsize * 0.2)
    if ax is None: fig, ax = plt.subplots(figsize = figsize)

    ax.ecdf(inp, complementary = complementary, color = color, alpha = alpha, 
            linestyle = linestyle) 
    if ylim is not None: ax.set_ylim(0, ylim)
    
    if xlabel is not None: ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None: 
        if ylabel is True: ylabel = 'Probability of Occurence'
        ax.set_ylabel(ylabel, fontsize=fontsize)
    
    if vline is not None: 
        ax.axvline(vline, ls = '--', color = color)
        #ax.set_xticks(list(ax.get_xticks()) + [np.round(vline, 2)])
    
    if hline is not None: ax.axhline(hline, ls = '--', color = color)
    
    if text is not None: ax.text(*text)
    
    if suptitle is not None: ax.set_title(suptitle, size = titlesize, fontweight='bold')
        
    return  


def TablePlot(cellText, colLabels = None, cellLoc = 'center', loc = 'center', fontsize = 10, ax = None, figsize = (5,5)): 

    plt.rc('font', size=fontsize)

    if ax is None: fig, ax = plt.subplots(figsize = figsize)

    # Hide axes

    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    ax.table(cellText=cellText,colLabels=colLabels,loc=loc, cellLoc = cellLoc)

    fig.tight_layout()

    return



def MultiPlot(inp, multi = False, paired = False, transpose = False, 
                 plt_mode = [GridPlot, {}], customgrid = None, 
                 suptitle = None, columntitle = None, rowtitle = None, rowtitle_rotation = True, 
              indiv_args = None, # each one in here is a list for each plot. 
                 colors = None, cmap = None, colorbar = False,
                 tight = True, 
                 figsize = (5,5), fontsize = 10, sharey = False, sharex = False): 
    
    
    #multi is a list or an array of different data required. 
    #Paired is a list of lists or arrays to be paired 
    # a single pair is the same as multi + Transpose 
    
    if multi is False: paired = True
    
    if customgrid is None: 
        if paired is False: 
            lm, lk = len(inp), 1
        elif multi is True:
            lm, lk = len(inp[0]), len(inp)
        else: lm, lk = 1, len(inp)
    else: lm, lk = customgrid
    
    k = -1 if transpose == True else 1
    
    grid = [lm, lk][::k] 
    
    plt.rc('font', size=fontsize)  
    titlesize = int(fontsize + fontsize * 0.2)
    fig, axs = plt.subplots(*grid, figsize = figsize, sharey = sharey, sharex = sharex)
    
    if paired is True:
        if multi is True: 
            if transpose is True: inp = [ww for ee in inp for ww in ee]
            else: inp = [o[p] for p in range(len(inp[0])) for o in inp]
    
    li = len(inp) 
    
    plt_mode[1]['fontsize'] = fontsize
    
    if colors is None and cmap is not None:
        colors = [plt.get_cmap(cmap, li)(i) for i in range(li)]
    
    for iax ,ax in enumerate(axs.flatten()): 
        
        if iax >= li: 
            ax.axis('off')
            continue 
        
        if colors is not None: plt_mode[1]['color'] = colors[iax]
        if indiv_args is not None: plt_mode[1].update({k:v[iax] for k,v in indiv_args.items()}) 
        im = plt_mode[0](inp[iax], ax = ax, **plt_mode[1])
        
    if suptitle is not None: fig.suptitle(suptitle, fontsize = titlesize, fontweight='bold') #Does not work for ScatterHist* 



    pad, pad2 = 5, 1

    if plt_mode[0] == ScaHistPlot: pad2 = 1.2
        
    if plt_mode[0] in [ScaHistPlot, PairedBarPlot]: 
        if isinstance(rowtitle_rotation, bool): rowtitle_rotation = not rowtitle_rotation

    if rowtitle is not None: 
        rowtitle_rotation = 90 if rowtitle_rotation is True else None

        popo = axs[:,0] if grid[1] > 1 else axs
        for ax, row in zip(popo, rowtitle):
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points', rotation = rowtitle_rotation, 
                fontsize = fontsize, fontweight = 'bold', ha='right', va='center')

    if columntitle is not None: 

        popo = axs[0] if grid[0] > 1 else axs
        for ax, col in zip(popo, columntitle):
            ax.annotate(col, xy=(0.5, pad2), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                fontsize = fontsize, fontweight = 'bold', ha='center', va='baseline')
    
    '''
    if columntitle is not None: 
        bb = axs.flatten()[:2] if multi == True else axs.flatten()
        for i, ax in enumerate(bb):
            ax.set_title(columntitle[i], fontsize = fontsize, fontweight='bold')'''
    
    if tight: fig.tight_layout()

    if colorbar == True: colorbar = 'right'
    if colorbar: 
        fig.colorbar(im, ax = axs.ravel().tolist(), location = colorbar, shrink=0.75)

    return axs





import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties

def LogoPlot(arr, ax = None, figsize = (5,5), fontsize = 10, 
             ylabel = 'Proportion', xticks = True): 

    #########################################

    fp = FontProperties(weight="bold") 
    globscale = 1.35
    LETTERS = { "T" : TextPath((-0.32, 0), "T", size=1, prop=fp),
                "G" : TextPath((-0.45, 0), "G", size=1, prop=fp),
                "A" : TextPath((-0.4, 0), "A", size=1, prop=fp),
                "C" : TextPath((-0.4, 0), "C", size=1, prop=fp) }
    COLOR_SCHEME = {'G': 'orange', 'A': 'darkgreen', 'C': 'blue', 'T': 'red'}

    def letterAt(letter, x, y, yscale=1, ax=None):
        text = LETTERS[letter]

        t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
            mpl.transforms.Affine2D().translate(x,y) + ax.transData
        p = PathPatch(text, lw=0, fc=COLOR_SCHEME[letter],  transform=t)
        if ax != None:
            ax.add_artist(p)
        return p

    #########################################

    if ax is None: fig, ax = plt.subplots(figsize = figsize)

    lets = np.array(['A', 'C', 'G', 'T'])

    x = 1
    maxi = 0
    for scores in arr:
        y = 0
        sor = np.argsort(scores)
        scores_sortd, lets_sorted = [jk[sor] for jk in [scores, lets]]
        for base, score in zip(lets_sorted, scores_sortd):
            letterAt(base, x,y, score, ax)
            y += score
        x += 1
        maxi = max(maxi, y)

    ax.set_xlim((0, x))
    ax.set_ylim((0, maxi))
    ax.set_ylabel(ylabel, fontsize = fontsize)
    if xticks: 
        ax.set_xticks(range(1,x))
        ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)

    return ax