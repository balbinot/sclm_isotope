
#!/usr/bin/env python

from matplotlib import pyplot as plt
import numpy as np
import triangle
lbl = "p1 p2".split()
triangle.corner(T, labels=lbl)

plt.figure()
ptest = (np.median(T[:,0]), np.median(T[:,1]))

pf = ptest
sdpf = (np.std(T[:,0]), np.std(T[:,1]))

tf = hhh(ptest)
X = np.arange(rAx, 12.5, 0.01)
Y = tf.evalhyp(X, ptest)
y, dy, x, dx = np.loadtxt(argv[1], skiprows=1, delimiter=',', unpack=True)

#Test nearest point
xn, yn = getnear(ptest, data)
plt.plot(xn,yn, 'ro')
plt.plot(X,Y,'k-')

for i in np.unique(np.random.randint(len(T[:,0]),size=1000)):
    print i

    ptrial = (T[i,0], T[i,1])
    tft = hhh(ptrial)
    Y = tft.evalhyp(X, ptrial)
    plt.plot(X,Y,'k-', alpha=0.05)


#Test nearest point
#xn, yn = getnear((3.28, 7347), data)
#plt.plot(xn,yn, 'ro')

## Plot stuff
plt.plot(X,Y,'k-')

## Air value
plt.plot(rAx, rAy, 's', c='c', ms=8)
plt.plot(rAx, rAy, '+', c='k', ms=8)

## Last point (actually a model parameter)
peff = [path_effects.Stroke(linewidth=3, foreground='white'),path_effects.Normal()]
plt.errorbar(12.5, pf[1], yerr=sdpf[1], xerr=0, fmt='None', ecolor='w', elinewidth=3, capsize=2)
plt.errorbar(12.5, pf[1], yerr=sdpf[1], xerr=0, fmt='w^', ecolor='k', capsize=2)

plt.errorbar(x, y, yerr=dy, xerr=dx, fmt='ws', capsize=2, ecolor='w', elinewidth=3)
plt.errorbar(x, y, yerr=dy, xerr=dx, fmt='ws', capsize=2, ecolor='k')

plt.xlabel(r'$^{20}{\rm Ne}/^{22}{\rm Ne}$')
plt.ylabel(ylab)
plt.xlim(9.7, 13)

outtpl = """### Best fit for %s dataset ###
cMy, rMy = %.4f, %.4f
ecMy, erMy = %.4f, %.4f
""" % (argv[1], pf[0], pf[1], sdpf[0], sdpf[1])

print outtpl
plt.show()

exit()
odir = 'plots_morb/'
outname = argv[1].replace('dados_morb/','').replace('csv','txt')
a = open(odir+outname, 'w')
a.write(outtpl)
a.close()
figname = argv[1].replace('dados_morb/','').replace('csv','png')
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
