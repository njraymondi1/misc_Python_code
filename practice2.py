import comms
import numpy as np
import natecode
import matplotlib.pyplot as plt

"""
mod_type = [('PSK',4)]
rrs = [[2.75]]#, [2.72, 4.87], [0], [0], [0], [0], [0]]
esno = np.inf
sps =5
b = comms.rc_imp(sps)
Ns = 30000

for mod, rr in zip(mod_type, rrs):
    sym = comms.gen_symbols(Ns, mod, rr)
    sam = comms.gen_samples(Ns, mod, rr, sps, b)

    plt.figure(1)
    plt.title(mod)
    plt.plot(sam.real, sam.imag, alpha=.25, c='C1')
    plt.scatter(sym.real, sym.imag, lw=0, alpha=.25, c='C0')
    plt.axes().set_aspect('equal')
    plt.ylim(-2,2)
    plt.xlim(-2,2)

    plt.figure(2)
    plt.stem(sym)

    plt.figure(3)
    plt.stem(sam)
    plt.show()



alpha = np.arange(0,1,0.001)

#Sx, alphao, fo = correlation.ssca(sam, 1, 1/64, 1/64)
con = natecode.symrate(sam, alpha, 0.0005)
con = np.absolute(con)

plt.figure(4)
plt.plot(alpha,con)


#noncon = correlation.cac(sam, alpha, 1)
#plt.figure(5)
#plt.stem(noncon)
plt.show()

"""






















