import numpy as np
import math

def Mpq(x, p, q):
    """
    Author:
    Aaron Smith
    
    Description:
    Calculates the mixed moment of a signal with p total terms and q conjugate terms.

    Input:
    Signal to evaluate

    Output:
    Mpq(x)
    """

    return 1./len(x) * np.dot(x**(p-q), np.conjugate(x)**q)

def C20(x):
    return Mpq(x,2,0)

def C21(x):
    return Mpq(x,2,1)

def C40(x):
    return Mpq(x,4,0)-3.*Mpq(x,2,0)**2

def C41(x):
    return Mpq(x,4,1)-3*Mpq(x,2,0)*Mpq(x,2,1)

def C42(x):
    return Mpq(x,4,2)-abs(Mpq(x,2,0))**2-2.*Mpq(x,2,1)**2

def C60(x):
    return Mpq(x,6,0)-15*Mpq(x,2,0)*Mpq(x,4,0)+30*Mpq(x,2,0)**3

def C61(x):
    return Mpq(x,6,1)-5*Mpq(x,2,1)*Mpq(x,4,0)-10*Mpq(x,2,0)*Mpq(x,4,1)+30*Mpq(x,2,0)**2*Mpq(x,2,1)

def C62(x):
    return Mpq(x,6,2)-6*Mpq(x,2,0)*Mpq(x,4,2)-8*Mpq(x,2,1)*Mpq(x,4,1)-Mpq(x,2,2,)*Mpq(x,4,0)+6*Mpq(x,2,0)**2*Mpq(x,2,2)+24*Mpq(x,2,1)**2*Mpq(x,2,0)

def C63(x):
    return Mpq(x,6,3)-6.*Mpq(x,2,0)*Mpq(x,4,1)-9.*Mpq(x,2,1)*Mpq(x,4,2)+18.*Mpq(x,2,0)**2*Mpq(x,2,1)+12.*Mpq(x,2,1)**3

def C80(x):
    return Mpq(x,8,0)-35.*Mpq(x,4,0)**2-28.*Mpq(x,6,0)*Mpq(x,2,0)+420.*Mpq(x,4,0)*Mpq(x,2,0)**2-630.*Mpq(x,2,0)**4

def fast_cumulants(x):
    """
    Author:
    Aaron Smith
    
    Description:
    Calculate the set of mixed cumulants [C20, C40, C41, C42, C60, C61, C62, C63, C80].
    This function reduces number of calls to Mpq(x,p,q) by reusing returns in multiple cumulant
    calculations.

    Input:
    Signal to evaluate

    Output:
    [C20, C40, C41, C42, C60, C61, C62, C63, C80]
    """

    # Calculate the moments
    M20 = Mpq(x,2,0)
    M21 = Mpq(x,2,1)
    M22 = Mpq(x,2,2)
    M40 = Mpq(x,4,0)
    M41 = Mpq(x,4,1)
    M42 = Mpq(x,4,2)
    M43 = Mpq(x,4,3)
    M60 = Mpq(x,6,0)
    M61 = Mpq(x,6,1)
    M62 = Mpq(x,6,2)
    M63 = Mpq(x,6,3)
    M80 = Mpq(x,8,0)

    # Calculate the cumulants using the moment to cumulant function (compare with moment_to_cumulant function)
    C20 = M20
    C40 = M40 - 3*M20**2
    C41 = M41 - 3*M20*M21
    C42 = M42 - abs(M20)**2 - 2*M21**2
    C60 = M60 - 15*M20*M40 + 30*M20**3
    C61 = M61 - 5*M21*M40 - 10*M20*M41 + 30*M20**2*M21
    C62 = M62 - 6*M20*M42 - 8*M21*M41 - M22*M40 + 6*M20**2*M22 + 24*M21**2*M20
    C63 = M63 - 3*M20*M43 - 9*M21*M42 - 3*M22*M41 + 18*M20*M21*M22 + 12*M21**3
    C80 = M80 - 35*M40**2 - 28*M20*M60 + 420*M20**2*M40 - 630*M20**4

    return np.array([C20, C40, C41, C42, C60, C61, C62, C63, C80])

def partition(collection):
    """
    Source:
    http://stackoverflow.com/questions/19368375/set-partitions-in-python

    Description:
    This function takes in a collection, and returns all possible partitions.

    Use:
    Used in the moment_to_cumulant function.
    
    Input:
    A collection. If [1,2,3] -> [[1], [2,3]], [[1,2],[3]], [[1,2,3]], [[1,3],[2]], [[1],[2],[3]]
    
    Output:
    A generator that iterates through all possbile partitions of the input collection
    """    
    
    if len(collection) == 1:
        yield [ collection ]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        # put `first` in its own subset 
        yield [ [ first ] ] + smaller


def moment_to_cumulant(c):
    """
    Author:
    Aaron Smith

    Description:
    Finds an equation of moments which equal the requested cumulant.
    The result can (and is) simplified by assuming the random variable is zero mean.
    This function also assumes a single random variable "X" and it's conjugate "X*"

    Input:
    c is a string such as 'C20', 'C21', which denotes the desired mixed cumulant

    Output:
    a string representation of the output of the moment to cumulant function.
    
    """
    
    # The first number in the mixed moment signifies how many variables are present
    indicies = list(range(int(c[1])))

    strings_by_pi = {}
    
    for p in partition(indicies):
        
        # pi is the number of elements in this partition. [[1],[2,3]] -> 2
        pi = len(p)

        moments = ''
        for B in p:
            moment = np.array([0,0])
            for b in B:
                # Because our signal is either X or X*
                moment[0] += 1
                # if this is X*
                moment[1] += 1 if b >= int(c[1]) - int(c[2]) else 0
            
            # Assuming zero mean, distribution symetric about zero, odd 
            # moments equal to zero
            if moment[0] % 2 != 0:
                moments = ''
                # break causes E[X^(odd)]*anything = 0
                break
            else:
                moments = moments + 'M' + str(moment[0]) + str(moment[1])
  
        if moments != '':
            try:
                strings_by_pi[pi].append(moments)
            except:
                strings_by_pi[pi] = []
                strings_by_pi[pi].append(moments)

    # Make a dictionary to hold the coefficients for a given value pi    
    coeff_by_pi = {}

    # Many products are repeated, reduce length by grouping like terms.
    for pi in strings_by_pi.keys():
        # Gather like elements to reduce string length
        elements, counts = np.unique(strings_by_pi[pi], return_counts=True)

        # Prep and empty string
        reduced = ''

        # Calculate the pi dependent coefficient
        coeff_by_pi = (-1)**(pi-1)*math.factorial(pi-1)

        # Combine terms and apply coeff
        for e,c in zip(elements,counts):
            reduced = reduced + str(coeff_by_pi * c) + "*" + e + " + "
        reduced = reduced[:-2]
        strings_by_pi[pi] = reduced
            
    # Convert the dictionary to a single string
    ret = ''
    for pi in strings_by_pi.keys():
        ret = ret + ' + ' + strings_by_pi[pi]

    return ret[2:]


def decide(signal_Cuv, m_Cuv, n_Cuv):
    """
    Author:
    Aaron Smith

    Description:
    Used in classify function to determine direction at junction
    """

    if abs(signal_Cuv) < (abs(m_Cuv) + abs(n_Cuv))/2.:
        return 1
    else:
        return 0
    
def classify_signal(x, c):
    """
    Author:
    Aaron Smith

    Description:
    Dtree classifier, based on Flohberger / Swami
    """

    if not decide(cumulants.C42(x), c[2,2], c[3,2]):
        # 0 right
        return 2#('psk', 2) 
    else:
        # 0 left
        if not decide(cumulants.C42(x), c[3,2], c[0,2]):
            # 1 right
            if not decide(cumulants.C40(x), c[3,1], c[4,1]):
                # 2 right
                return 3#('psk', 4)
            else:
                # 2 left
                return 4#('psk', 8)
        else:
            # 1 left
            if not decide(cumulants.C40(x), c[5,1], c[0,1]):
                # 3 right
                if not decide(cumulants.C40(x), c[5,1], c[6,1]):
                    # 4 right
                    return 5#('qam', 16)
                else:
                    # 4 left
                    return 6#('qam', 64)
            else:
                # 3 left
                if not decide(cumulants.C42(x), c[0,2], c[1,2]):
                    # 5 right
                    return 0#('apsk', 16)
                else:
                    # 5 left
                    return 1#('apsk', 32)

def classify_cumulants(c40, c42, expected_cumulants):
    """
    Author:
    Aaron Smith

    Description:
    Dtree classifier, based on Flohberger / Swami
    """


    if not decide(c42, expected_cumulants[2,2], expected_cumulants[3,2]):
        # 0 right
        return 2#('psk', 2) 
    else:
        # 0 left
        if not decide(c42, expected_cumulants[3,2], expected_cumulants[0,2]):
            # 1 right
            if not decide(c40, expected_cumulants[3,1], expected_cumulants[4,1]):
                # 2 right
                return 3#('psk', 4)
            else:
                # 2 left
                return 4#('psk', 8)
        else:
            # 1 left
            if not decide(c40, expected_cumulants[5,1], expected_cumulants[0,1]):
                # 3 right
                if not decide(c40, expected_cumulants[5,1], expected_cumulants[6,1]):
                    # 4 right
                    return 5#('qam', 16)
                else:
                    # 4 left
                    return 6#('qam', 64)
            else:
                # 3 left
                if not decide(c42, expected_cumulants[0,2], expected_cumulants[1,2]):
                    # 5 right
                    return 0#('apsk', 16)
                else:
                    # 5 left
                    return 1#('apsk', 32)