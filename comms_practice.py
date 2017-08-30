import sys
sys.path.insert(0, '../scripts/')
import comms
import numpy as np
from scipy import signal
import correlation

#%matplotlib inline
import matplotlib.pyplot as plt


Ns = 10
sps=8
M=6
alpha = .5

symbols = (2*np.random.randint(0,2,Ns)-1)
up_symbols = np.hstack((symbols.reshape(symbols.shape[0], 1), np.zeros((symbols.shape[0], sps-1)))).flatten()
shaped = up_symbols

plt.figure(1)
plt.stem(shaped)

b = comms.rc_imp(sps, alpha, M)

shaped = signal.lfilter(b,1,shaped)

plt.figure(2)
plt.stem(shaped)

hello = comms.gen_symbols(Ns, 'PSK', 2)
plt.figure(3)
plt.stem(hello)

plt.show()










