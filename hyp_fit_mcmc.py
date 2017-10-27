#!/usr/bin/env python

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import newton
import matplotlib.pyplot as plt
from sys import argv
from matplotlib.ticker import Formatter,FuncFormatter

## Constants (depends on the isotopes beying analised)
rAx = 9.8
rMx = 12.5

tipo = argv[2]
data = np.loadtxt(argv[1], skiprows=1, delimiter=',')
if len(data[:,0]) <= 3:
    exit()

## Define some elemtn specific parameters and guesses for the model
if tipo=="Ar":
    rAy = 296  # atmospheric
    ylab = r'$\mathbf{^{40}{Ar}/^{36}{Ar}}$'
    beta0 = [3.0, np.max(data[:,0])]
    size0 = [1, 500]
    #ylims = [rAy-0.02*rAy, 100000]
    ylims = [rAy-0.02*rAy, 12e3]
    strok = r'$=$ $%.1f^{+%.1f}_{-%.1f}$'
    strlo = r'$>$ $%.1f$'
    strtxt = r'Neon-B $^{40}{\rm Ar}/^{36}{\rm Ar}$ '
    class SciFormatter2(Formatter):
        def __call__(self, x, pos=None):
            return r"$\mathbf{%i}$" % x

elif tipo=='129Xe':
    rAy = 0.9832
    ylab = r'$\mathbf{^{129}{Xe}/^{132}{Xe}}$'
#    beta0 = (110, 1.0)
    beta0 = [6.0, np.max(data[:,0])]
    size0 = [1, 0.1]
    ylims = [rAy-0.02*rAy, 1.2]
    strok = r'$=$ $%.2f^{+%.2f}_{-%.2f}$'
    strlo = r'$>$ $%.2f$'
    strtxt = r'Neon-B $^{129}{\rm Xe}/^{132}{\rm Xe}$ '
    class SciFormatter2(Formatter):
        def __call__(self, x, pos=None):
            return r"$\mathbf{%.2f}$" % x
elif tipo=='136Xe':
    rAy = 0.3294
    ylab = r'$\mathbf{^{136}{Xe}/^{132}{Xe}}$'
    beta0 = [3.0, np.max(data[:,0])]
    size0 = [1, 0.1]
    ylims = [rAy-0.02*rAy, 0.55]
    strok = r'$=$ $%.2f^{+%.2f}_{-%.2f}$'
    strlo = r'$>$ $%.2f$'
    strtxt = r'Neon-B $^{136}{\rm Xe}/^{132}{\rm Xe}$ '
    class SciFormatter2(Formatter):
        def __call__(self, x, pos=None):
            return r"$\mathbf{%.2f}$" % x
else:
    print "Tipo nao reconhecido", argv[2]


def pcorr(p):
    cMy, rMy = p
    A = cMy*rMy - rAy
    B = 1 - cMy
    C = cMy*rAx - rMx
    D = rAy*rMx - cMy*rMy*rAx
    return (A,B,C,D)

class hhh():

    def __init__(self, p):
        # Global p needed for root finding
        self.p = p

    def evalhyp(self, x,p):

        # Parametere array (this is what you fit)
        cMy, rMy = p

        # Parameters correlation
        A, B, C, D = pcorr(p)

        # Compute model in X and Y
        my = -(A*x + D)/(B*x + C)

        return my

    def evalhypdx(self, x):
        # First derivative
        p = self.p
        A, B, C, D = pcorr(p)
        return (B*D - A*C)/(B*x + C)**2

    def evalhypdx2(self, x):
        # Second derivative
        p = self.p
        A, B, C, D = pcorr(p)
        return 2.*B*(A*C - B*D)/(B*x + C)**3

def getnear(p, data):

    # Parametere array (this is what you fit)
    cMy, rMy = p

    # Apply correlations
    A, B, C, D = pcorr(p)

    # Aproximate method. Should use Newton-Raphson to be identical to Rita
    xout = []
    yout = []
    tf = hhh(p)
    for x, y, dx, dy in zip(data[:,2], data[:,0], data[:,3], data[:,1]):
        dddx = lambda mx: -2.*(x - mx)/dx**2 - 2.*(y - tf.evalhyp(mx,p))*tf.evalhypdx(mx)/dy**2
        dddx2 = lambda mx: 2./dx**2 + (2./dy**2)*((tf.evalhyp(mx,p) - y)*tf.evalhypdx2(mx) + tf.evalhypdx(mx)**2)
        try:
           x0 = newton(dddx, 9.8, fprime=dddx2, maxiter=3000)
        except:
           x0 = np.Inf
        xout.append(x0)
        yout.append(tf.evalhyp(x0,p))

    xout = np.array(xout)
    yout = np.array(yout)
    return xout, yout

def getchi(p, data):

    # Parametere array (this is what you fit)
    cMy, rMy = p

    rmymax = 100000
    # Priors
    if '136Xe-Ne' in argv[1]:
        upl, dol = 6, 100
    elif '136Xe-20Ne' in argv[1]:
        upl, dol = 0, 1000
    elif '129Xe-20Ne' in argv[1]:
        upl, dol = -10, 20
        rmymax = 10
    elif '129Xe-20Ne' in argv[1]:
        upl, dol = 0, 20
    else:
        upl, dol = 0, 1000
    #if cMy < upl or rMy < np.max(data[:,0])+0.001*np.max(data[:,0]) or rMy > 100000 or cMy > dol or rMy < 0:
    if cMy < upl or rMy < np.max(data[:,0]) or rMy > rmymax or cMy > dol or rMy < 0:
        return -np.Inf

    # Data array, never changes, just organizing
    x, y = data[:,2], data[:,0]
    dx, dy = data[:,3], data[:,1]
    dx = dx

    # Parameters correlation
    A, B, C, D = pcorr(p)

    # Get nearest model point to each data point
    xnear, ynear = getnear(p, data)
    i = (xnear < 12.5)
    if len(xnear[i]) < len(xnear):
        return -np.Inf

    # Compute model in Y value in the X that gives to shortest distance
    tf = hhh(p)
    my = tf.evalhyp(xnear,p)

    if len(xnear[i]) > 3:
        chisq =  1.0/(len(x[i])-2-1) * np.sum(((x[i] - xnear[i])/dx[i])**2 + ((y[i] - my[i])/dy[i])**2)
    else:
        print "Warning: less than 3 points for fit"
        chisq = -np.Inf

    ## Return ln(L) (log of the likelihood)
    return -chisq

if argv[3]=='fit':
    ## Run fit
    import emcee
    import matplotlib.patheffects as path_effects

    ndim = 2
    nwalkers = 32
    p0 = []
    n = 1
    while n < nwalkers:
        ptrial =  (np.array(beta0)+[size0[0]*np.random.randn(), size0[1]*np.random.randn()])
        if getchi(ptrial, data) > -np.Inf:
            p0.append(ptrial)
            n = n +1
            print n

    p0 = [np.array(beta0)+[size0[0]*np.random.randn(), size0[1]*np.random.randn()] for i in xrange(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, getchi, args=[data], threads=4)
    pos, prob, state = sampler.run_mcmc(p0, 500)
    sampler.reset()
    sampler.run_mcmc(pos, 10000, rstate0=state)
    T = sampler.flatchain
    P = sampler.flatlnprobability
    np.savez(argv[1].replace('.csv','.npz'), T=T, P=P)

else:

    from matplotlib import rc,rcParams
    rc('text', usetex=True)
    rc('axes', linewidth=1.2)
    rc('font', weight='bold')
    rc('font',**{'family':'serif','serif':['Times']})
    rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
    f = np.load(argv[1].replace('.csv','.npz'))
    T = f['T']
    P = f['P']

    j = (P - np.max(P) > - 1)

    #plt.hist(P-np.max(P),bins=100)
    #plt.show()
    #exit()

    #import triangle
    #lbl = "p1 p2".split()
    #triangle.corner(T[j], labels=lbl)

    plt.figure(figsize=(7,7))

## Select the most likely
    bidx = np.argmax(P)
## Select the median of the chain
    #try:
    #    midx = np.where((P==np.median(P[j])))[0]
    #    print len(midx)
    #    midx = midx[0]
    #except:
    #    val = np.abs((P - np.median(P[j])))
    #    midx = np.argmin(val)
    #    print np.median(P[j])
    #    print 'single', midx
    #pf = (T[midx,0], T[midx,1])

    pf = (np.median(T[j,0]), np.median(T[j,1]))
    mpf = (T[bidx,0], T[bidx,1])
    sdpf = [[pf[0]-np.min(T[j,0]), np.max(T[j,0])-pf[0]], [pf[1]-np.min(T[j,1]), np.max(T[j,1])-pf[1]]]
    dsdpf = [[np.min(T[j,0]), np.max(T[j,0])], [np.min(T[j,1]), np.max(T[j,1])]]

    outtpl = """### Best fit for %s dataset ###
    cMy, rMy = %.4f, %.4f (mean; solid line)
    ecMy, erMy = [%.4f,%.4f], [%.4f, %.4f] (error bar shown in plot @ 12.5)

    cMy, rMy = %.4f, %.4f (maximum likelihood; dashed line)

    mincMy, maxcMy, minrMy, maxrMy = %.4f, %.4f %.4f, %.4f (68pct confidence interval)
    """ % (argv[1], pf[0], pf[1], sdpf[0][0], sdpf[0][1], sdpf[1][0],
           sdpf[1][1], mpf[0], mpf[1], dsdpf[0][0], dsdpf[0][1], dsdpf[1][0],
           dsdpf[1][1])

    print outtpl

    ## Re-read data for some reason
    y, dy, x, dx = np.loadtxt(argv[1], skiprows=1, delimiter=',', unpack=True)

    ## Compute best fit model
    tf = hhh(pf)
    X = np.arange(rAx, 12.5, 0.01)
    Y = tf.evalhyp(X, pf)
    mX = np.arange(rAx, 12.5, 0.01)
    mY = tf.evalhyp(mX, mpf)
    plt.plot(X,Y,'k-', zorder=98)
    #plt.plot(mX,mY,'k--', zorder=98)

    #Test nearest point

    for i in np.unique(np.random.randint(len(T[:,0]),size=1000)):
        if P[i] - np.max(P) > -2.4:
            ptrial = (T[i,0], T[i,1])
            tft = hhh(ptrial)
            Y = tft.evalhyp(X, ptrial)
            plt.plot(X,Y,'-', color='0.5', alpha=0.05, zorder=0, rasterized=False)

    #Test nearest point
    #xn, yn = getnear((3.28, 7347), data)
    #plt.plot(xn,yn, 'ro')

    ## Plot stuff

    ## Air value
    plt.plot(rAx, rAy, 's', c='k', ms=8, label='1')

    import matplotlib.patheffects as path_effects
    pef = [path_effects.Stroke(linewidth=3, foreground='white'),
          path_effects.Normal()]

    ## Last point (actually a model parameter)
    ## Trick to plot assymetric errorbar for a single point
    xtemp = np.array(2*[12.5])
    ytemp = np.array(2*[pf[1]])
    yerr = np.array([sdpf[1],sdpf[1]]  ).T
    if sdpf[1][1]+pf[1] > ylims[1]:
        lolims = True
        sdpf[1][1] = ylims[1] - pf[1]
        yerr = np.array([sdpf[1],sdpf[1]]  ).T
        plt.errorbar(xtemp, ytemp, yerr=yerr, fmt='w^', ms=8, ecolor='k', capsize=2, elinewidth=1.2, zorder=99, lolims=lolims)
        sdpf[1][1] = 0  # dont plot over lolime sign
        tstr = strlo % (pf[1]-sdpf[1][0])
    else:
        tstr = strok % (pf[1], sdpf[1][1], sdpf[1][0])
        lolims = False
    print yerr

    yerr = np.array([sdpf[1],sdpf[1]]  ).T
    plt.errorbar(xtemp, ytemp, yerr=yerr, fmt='w^', ms=8, ecolor='k', capsize=2, elinewidth=1.2, zorder=99, label='2')
    plt.errorbar(x, y, yerr=dy, xerr=dx, fmt='ws', capsize=2, ecolor='k', elinewidth=1.2, zorder=99, label='3')

    plt.xlabel(r'$\mathbf{^{20}{Ne}/^{22}{Ne}}$', fontsize=20)
    plt.ylabel(ylab, fontsize=20)
    plt.xlim(9.7, 13)

    #if plt.ylim()[1] > ylims[1]:
    plt.ylim(ylims)

    plt.minorticks_on()
    plt.gca().tick_params('both', width=1.2, which='major')
    plt.gca().tick_params('both', width=1.2, which='minor')



#    plt.gca().get_xticklabels()[0].set_weight('bold'

    ## Add some text
    #plt.text(0.05, 0.9,strtxt+tstr, ha='left', va='center', transform=plt.gca().transAxes)

    # get handles
    handles, labels = plt.gca().get_legend_handles_labels()
    # remove the errorbars
    thand = []
    for h in handles:
        try:
            thand.append(h[0])
        except:
            thand.append(h)
    handles = thand

    class SciFormatter(Formatter):
        def __call__(self, x, pos=None):
            return r"$\mathbf{%.1f}$" % x

    plt.gca().xaxis.set_major_formatter(SciFormatter())
    plt.gca().yaxis.set_major_formatter(SciFormatter2())

    # use them in the legend
    #plt.legend(handles, labels, loc='lower right', numpoints=1, frameon=False, fontsize=14)

    try:
        a = argv[4]
        plt.show()
    except:
        pass

    odir = 'plots_all/'
    outname = argv[1].split('/')[-1].replace('csv','txt')
    a = open(odir+outname, 'w')
    a.write(outtpl)
    a.close()
    figname = argv[1].split('/')[-1].replace('csv','png')
    plt.savefig(odir+figname)
    figname = argv[1].split('/')[-1].replace('csv','pdf')
    plt.savefig(odir+figname)

    ## FIRST TRY USING CHI-SQUARED. FAIL. LEAVE COMMENTED
    #def evalhyp(x, p):
    #
    #    # Parametere array (this is what you fit)
    #    cMy, rMy = p
    #
    #    # Parameters correlation
    #    A = cMy*rMy - rAy
    #    B = 1 - cMy
    #    C = cMy*rAx - rMx
    #    D = rAy*rMx - cMy*rMy*rAx
    #
    #    # Compute model in X and Y
    #    my = -(A*x + D)/(B*x + C)
    #    #my[my < 0] = 0
    #
    #    return my
    #
