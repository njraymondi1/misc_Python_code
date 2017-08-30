import math
import numpy as np

def nextpow2(number):

    return (math.ceil(math.log(number, 2)))

def ccac(x, a, Ts, N=0):
    """
    Author:
    Aaron Smith

    Description:
    This calculates the zero lag conjugate cyclic auto-correlation
    as defined in "Energy-Efficient Processor for Blind Signal
    Classification in Cognitive Radio Networks"

    Inputs:
        x : Input data sequence
        a : Cyclic frequency (scalar or numpy array)
        Ts : Sampling Period
        N : length of sequence to correlate

    Returns:
        r : The conjugate cyclic auto-correlation for lag=0 
            and cyclic frequency a

    """

    # If not N given, correlate over entire sequence
    if N == 0:
        N = len(x)

    # if a is a scalar, convert it to numpy array
    if not type(a).__module__ == np.__name__:
        a = np.array([a])

    # time matrix
    n = np.arange(N)
    n = np.tile(n, (len(a), 1))

    # exponential matrix
    e = a.reshape((a.shape[0]), 1)
    e = np.repeat(e, N, axis=1)
    e = np.exp(-1j*2*np.pi*e*n*Ts).T

    # Calculate the ccac
    r = 1/float(N)*np.dot(abs(x)**2, e)

    return r
        

def cac(x, a, Ts, N=0):
    """
    Author:
    Aaron Smith

    Description:
    This calculates the zero lag cyclic auto-correlation
    as defined in "Energy-Efficient Processor for Blind Signal
    Classification in Cognitive Radio Networks"

    Inputs:
        x : Input data sequence
        a : Cyclic frequency (scalar or numpy array)
        Ts : Sampling Period
        N : length of sequence to correlate

    Returns:
        r : The conjugate cyclic auto-correlation for lag=0 
            and cyclic frequency a

    """

    # If not N given, correlate over entire sequence
    if N == 0:
        N = len(x)

    # if a is a scalar, convert it to numpy array
    if not type(a).__module__ == np.__name__:
        a = np.array([a])

    # time matrix
    n = np.arange(N)
    n = np.tile(n, (len(a), 1))

    # exponential matrix
    e = a.reshape((a.shape[0]), 1)
    e = np.repeat(e, N, axis=1)
    e = np.exp(-1j*2*np.pi*e*n*Ts).T

    # Calculate the ccac
    r = 1/float(N)*np.dot(x*x, e)

    return r

def ssca(data, fs, df, da):
    """ 
    This function computes the Spectral Correlation Density using the Strip Spectral Correlation Algorithm

    Inputs:
        data: The sequence to be analyzed
        fs: Sample frequency
        df: Distance between adjacent frequency bins
        da: Distance between adjacent cyclo frequency bins

    Returns:
        Sx: The spectral correlation density
        alphao: An array representing the cyclo frequency bins
        fo: An array representing the frequency bins

    Sources:
        Detection and identification of cyclostationary signals
        Costa, Evandro Luiz da.
        and
        commP25ssca.m
        https://www.mathworks.com/help/comm/examples/p25-spectrum-sensing-with-synthesized-and-captured-data.html
    """

    # window length
    Np = 2^nextpow2(fs/df)

    # Number of samples to shift between window captures
    L = Np/4

    # Number of captures
    P = 2^nextpow2(fs/(da/L))

    # Observed samples
    N = round(P*L,0)
    N = math.floor(N)

    # cyclic frequency
    alphao = np.linspace(-fs, fs, 1/(2*N+1))

    # spectral frequency
    fo = np.linspace(-0.5*fs, 0.5*fs, 1/(Np+1))

    # Only keep N samples
    if len(data)<N:
        #print("Forced N to be a power of two by zero padding input data.\nlen(data=" 
        #    + str(len(data)) + " N= " + str(N) + ".\nSuggest increasing capture period to N.")
        data[N:] = 0
    elif len(data)>N:
        print("Only " + str(N) + " of the " + str(len(data)) + " samples were used.")
        data = data[0:N]

    # determine number of total elements in strided matrix
    NN = math.floor((P-1)*L+Np)

    data_NN = data

    # if extend input data with zeros, or truncate if needed
    if len(data_NN)<NN:
        data_NN = np.hstack((data, np.zeros(NN-len(data))))
    else:
        data_NN = data_NN[0:NN]

    # channelize the input data
    #x = strided_matrix(data_NN, Np, L).T # .T is a non-conj transpose
    print(P)
    for k in range(P):
        x[:,k] = data_NN[k*L:k*L+Np]

    # prepare a windowing funtion
    a = np.diag(np.hamming(Np))

    XW = np.dot(a, x)

    # First FFT
    XFFT1 = np.fft.fftshift(np.fft.fft(XW, axis=0))
    
    X = np.hstack((XFFT1[:,P/2:P], XFFT1[:,0:P/2]))
    
    # Downshift in frequency
    t = np.arange(P)*L
    f = np.reshape(np.arange(Np) / float(Np) - .5, (Np,1))
    t = np.tile(t, (Np,1))
    f = np.tile(f, (1,P))
    E = np.exp(-1j*2*np.pi*f*t)

    XT = X*E
    
    XRep = np.zeros((Np,N)) # (Np,P*L)
    XRep = np.repeat(XT, L, axis=1)

    dataRep = np.tile(data, (Np,1))

    XMul = (XRep*np.conj(dataRep)).T

    XFFT2 = np.fft.fftshift(np.fft.fft(XMul, axis=0))
    XFFT2 = np.hstack((XFFT2[:,Np/2:Np], XFFT2[:,0:Np/2]))

    M = abs(XFFT2)

    Sx = np.zeros((Np+1, 2*N+1), dtype=np.float64)

    # Convert M to Sx
    for k1 in range(int(N)):
        k2 = np.arange(Np)
        alpha = k1/float(N) + k2/float(Np) - 1
        f = (k2/float(Np) - k1/float(N))/2.
        k = Np*(f+0.5)
        n = N*(alpha+1)
        for k2 in range(int(Np)):
            Sx[int(round(k[k2])),int(round(n[k2]))] = M[k1,k2]
            
    return Sx, alphao, fo