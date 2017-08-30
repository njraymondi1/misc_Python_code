import numpy as np
from scipy import signal
"""
Modulations:
QAM, APSK, GMSK, PSK, 

Filters:
Raised cosine (rc_imp)
Root Raised Cosine (srrc_imp)
Lowpass/Bandpass Butterworth

"""

def avg_energy(x):
    """
    Author:
    Aaron Smith

    Description:
    Avg signal power/energy. Unit variance if signal mean = 0
    """

    return 1/float(len(x)) * np.dot(x, np.conjugate(x)).real

def cpx_awgn(x, esno, sps=1):
    """
    Author:
    Aaron Smith
    
    Source:
        Mark Wickert - Digital Communications Function Module

    Description:
    Adds complex AWGN to a signal such that it has a specific esno.
    """

    a = np.sqrt(sps*np.var(x)*10**(-esno/10.)/2.)
    n = a * (np.random.randn(len(x)) + 1j*np.random.randn(len(x)))

    return x+n, n  

def apply_twta(x):
    """
    Author:
    Aaron Smith

    Source:
    Joseph Downey (matlab code)

    Description:
    Applies Saleh model using coefficients from a working TWTA amp.
    IBO is determined by scaling the signal appropriatlly before calling this
    function. A signal without scaling is at saturation.
    """

    alpha_a = 1.8740
    beta_a = 0.8680
    alpha_phi = -0.4244
    beta_phi = 0.5403
    
    magx = abs(x)
    magy = alpha_a * magx / (1 + beta_a * magx**2)
    angy = alpha_phi * magx**2 / (1 + beta_phi * magx**2)
    y = magy * np.exp(-1j*angy) / magx
    
    return x * y

###################
### MODULATION ###
###################

def gen_samples(Ns, mod_type, ring_ratios=0,sps=1, b=0):
    """
    Author:
    Aaron Smith

    Description:
    Supports PSK, APSK, QAM
    A function to provide a pulse shape filtered, baseband, random, zero mean, modulated time sequence of samples.
    
    Input:
    Ns - Number of symbols
    mod_type - tuple of form ('APSK', 16)
    sps - samples per symbol
    b - numerator coefficients for pulse shape filter
    """

    symbols = gen_symbols(Ns, mod_type, ring_ratios)
    samples = pulse_shape_symbols(symbols, sps, b)

    return samples / np.sqrt(avg_energy(samples))

def gen_symbols(Ns, mod_type, ring_ratios=0):
    """
    Author:
    Aaron Smith

    Source:
    Some elements came from Mark Wickert - Digital Communications Function Module

    Description:
    Supports PSK, APSK (16,32), QAM (4,16,64,256) (square constellations)
    Generate Ns random symbols for a given modulation type

    Inputs:
    Ns - Number of symbols
    mod_type - tuple of form ('APSK', 16)
    ring_ratio - defines distances between rings (see dvb-s2 standard) for APSK

    Output:
    A normalized sequence of symbols.
    """

    # PSK
    if mod_type[0].lower() == 'psk':
        x_ints = np.random.randint(0, mod_type[1], Ns)
        symbols = np.exp(-1j*2*np.pi/mod_type[1]*x_ints)
    
    # APSK
    elif mod_type[0].lower() == 'apsk':
        x_ints = np.random.randint(0, mod_type[1], Ns)

        supported = [16,32]

        try:
            assert mod_type[1] in supported
        except:
            print("APSK constellations supported " + str(supported))
            return

        if mod_type[1] == 16:

            # Verify valid ring ratio
            if len(ring_ratios) != 1:
                print("Expected scalar ring ratio for APSK16")
                return

            # APSK16 -> 4 inner ring, 12 outer ring
            R2_mag = 1.0
            R1_mag = R2_mag / ring_ratios[0]
            R1_phase = np.arange(4) * (2*np.pi/4.) + (np.pi/4.)
            R2_phase = np.arange(12) * (2*np.pi/12.) + (np.pi/12.)
            R1 = R1_mag * np.exp(1j*R1_phase)
            R2 = R2_mag * np.exp(1j*R2_phase)
            c = np.hstack((R1, R2))
        elif mod_type[1] == 32:
            
            # Verify valid ring ratio
            if len(ring_ratios) != 2:
                print("Expected 2 element ring ratio for APSK32")
                return

            # APSK32 -> 4 inner ring, 12 middle ring, 16 outer ring
            R3_mag = 1.0
            R1_mag = R3_mag / ring_ratios[1]
            R2_mag = R1_mag * ring_ratios[0]
            R1_phase = np.arange(4) * (2*np.pi/4.) + (np.pi/4.)
            R2_phase = np.arange(12) * (2*np.pi/12.) + (np.pi/12.)
            R3_phase = np.arange(16) * (2*np.pi/16.) + (np.pi/8.)
            R1 = R1_mag * np.exp(1j*R1_phase)
            R2 = R2_mag * np.exp(1j*R2_phase)
            R3 = R3_mag * np.exp(1j*R3_phase)
            c = np.hstack((np.hstack((R1, R2)), R3))
        symbols = c[x_ints]

    # QAM
    elif mod_type[0].lower() == 'qam':
        
        # Supported QAM modulations
        supported = [4,16,32,64,256]

        # Make sure valid request
        try:
            assert mod_type[1] in supported
        except:
            print("QAM constellations supported " + str(supported))
            return

        # Define the constellation locations by i and q ints
        i_ints = np.random.randint(0, np.sqrt(mod_type[1]), Ns)
        i_ints = 2*i_ints - (np.sqrt(mod_type[1])-1)
        q_ints = np.random.randint(0, np.sqrt(mod_type[1]), Ns)
        q_ints = 2*q_ints - (np.sqrt(mod_type[1])-1)

        symbols = i_ints + 1j*q_ints

    # Invalid request
    else:
        print("Unsupported modulation type.")

    return symbols / np.sqrt(avg_energy(symbols))

def pulse_shape_symbols(symbols, sps, b):
    """
    Author:
    Aaron Smith

    Description:
    Upsample and pulse shape filter a sequence of symbols
    """

    if sps > 1:
        # Upsample, prep for filter
        samples = np.hstack((symbols.reshape(len(symbols),1), np.zeros((len(symbols), sps-1)))).flatten()
        samples = signal.lfilter(b,1,samples)
    else:
        samples = symbols

    return samples #/ np.sqrt(avg_energy(samples))



def GMSK_bb(Nb, spb, MSK = 0,BT = 0.35):
    """

    Source:
    Mark Wickert - Digital Communications Function Module
    """

    bits = 2.*np.random.randint(0,2,Nb)-1
    x = np.repeat(bits, spb)

    M = 4
    n = np.arange(-M*spb, M*spb+1)

    p = np.exp(-2*np.pi**2*BT**2/np.log(2)*(n/spb)**2)

    p = p/np.sum(p)
    
    # Gaussian pulse shape if MSK not zero
    if MSK != 0:
        x = signal.lfilter(p,1,x)
    y = np.exp(1j*np.pi/2*np.cumsum(x)/spb)

    return y, bits

###################
######FILTERS######
###################

def rc_imp(sps,alpha=.35,M=6):
    """
    A truncated raised cosine pulse

    Source:
        Mark Wickert - Digital Communications Function Module

    The pulse shaping factor 0< alpha < 1 is required as well as the 
    truncation factor M which sets the pulse duration to be 2*M*Tsymbol.

    Parameters
    ----------
    Ns : number of samples per symbol
    alpha : excess bandwidth factor on (0, 1)
    M : equals RC one-sided symbol truncation factor

    Returns
    -------
    b : ndarray containing the pulse shape

    """
    # Design the filter
    n = np.arange(-M*sps,M*sps+1)
    b = np.zeros(len(n));
    a = alpha;
    sps *= 1.0
    for i in range(len(n)):
        if (1 - 4*(a*n[i]/sps)**2) == 0:
            b[i] = np.pi/4*np.sinc(1/(2.*a))
        else:
            b[i] = np.sinc(n[i]/sps)*np.cos(np.pi*a*n[i]/sps)/(1 - 4*(a*n[i]/sps)**2)
    
    return b

def srrc_imp(sps, alpha=.35, M=6):
    """

    Implementation of a truncated square root raise cosine filter

    Source:
        Mark Wickert - Digital Communications Function Module

    Inputs:
        sps : Samples per symbol
        alpha : excess bandwidth factor (0,1)
        M : truncation factor, M * sps * 2 + 1 taps

    Outputs:
        Coeff for a srrc filter with filter energy equal to one.

    When used in conjunction with a matched SRRC filter on the rx side,
    this would result in a raised cosine shape which has zero intersymbol interference
    and optimal removal of additive white noise at the reciever.
    """

    n = np.arange(-M*sps,M*sps+1)
    b = np.zeros(len(n))
    sps *= 1.0
    a = alpha
    for i in range(len(n)):
       if abs(1 - 16*a**2*(n[i]/sps)**2) <= np.finfo(np.float).eps/2:
           b[i] = 1/2.*((1+a)*np.sin((1+a)*np.pi/(4.*a))-(1-a)*np.cos((1-a)*np.pi/(4.*a))+(4*a)/np.pi*np.sin((1-a)*np.pi/(4.*a)))
       else:
           b[i] = 4*a/(np.pi*(1 - 16*a**2*(n[i]/sps)**2))
           b[i] = b[i]*(np.cos((1+a)*np.pi*n[i]/sps) + np.sinc((1-a)*n[i]/sps)*(1-a)*np.pi/(4.*a))

    return b / np.sqrt(sum(b**2))

def butter_bandpass(f1, f2, fs, order=5):
    """
    Author:
    Aaron Smith

    Source:
    https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
    
    Description:
    Given a frequency band, identify the numerator and denomincator taps for use in a bandpass filter.
    """

    nyq = 0.5 * fs
    low = f1 / nyq
    high = f2 / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, f1, f2, fs, order=5):
    """
    Author:
    Aaron Smith

    Source:
    https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
    
    Description:
    A function which generates a bandpass filter, applies it on the given data vector, and returns the result.
    """

    b, a = butter_bandpass(f1, f2, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


def butter_lowpass(cutoff, fs, order=5):
    """
    Author:
    Aaron Smith

    Source:
    https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
    
    Description:
    Given a frequency cutoff, identify the numerator and denomincator taps for use in a bandpass filter.
    """

    nyq = 0.5 * fs
    normalized_cutoff = cutoff / nyq
    b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    """
    Author:
    Aaron Smith

    Source:
    https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
    
    Description:
    A function which generates a lowpass filter, applies it on the given data vector, and returns the result.
    """

    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

#################
## Simulations ##
#################

def matched_sim(Ns, mod_type, ring_ratios, esno=np.inf, fo=0, twta=False, ibo=10.):
    """
    Author:
    Aaron Smith

    Description:
    This function simulates a signal with a given modulation type and random
    symbols. Assumes transmitter/reciever have perfect symbol timing and
    nyquist pulse shape filtering. Applies awgn noise such that signal has
    requested esno. Skips upsample/tx filter/rx filter/timing.
    Optional awgn, fo, twta.

    Inputs:
    Ns - Number of symbols/samples
    mod_type - modulation type of form ('APSK', 16)
    ring_ratios - used for APSK to define distance between rings
    esno - AWGN noise in dB. Energy per symbol per noise density
    fo - frequency offset. Doppler, down/up conversion error.
    twta - bool. Do you want to try adding a non-linear amplifier?
    ibo - if using the twta, how many dB are you backed off from saturation?

    Output:
    The simulated symbols a receiver would have captured under these conditions.
    """

    symbols = gen_symbols(Ns, mod_type, ring_ratios)
    symbols = symbols if not twta else apply_twta(symbols*10**(-ibo/20.))
    symbols = symbols if esno==np.inf else cpx_awgn(symbols, esno)[0]
    symbols = symbols if fo==0 else symbols * np.exp(-1j*2*np.pi*fo*np.arange(len(symbols)))

    return symbols / np.sqrt(avg_energy(symbols))
