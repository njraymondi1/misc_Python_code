import numpy as np
import matplotlib.pyplot as plt


def to_onehot(yy):
    """
    Author:
    Tim O'Shea

    Description:
    Given a list of indicies where each index is represented, create a 
    2-D matrix where the rows are examples and all columns are equal to
    zero, except for the column which represents the index, which equals 1.
    Used in NN training and evaluation.

    Example:
    [0,3,1,2,2] -> [[1,0,0,0],[0,0,0,1],[0,1,0,0],[0,0,1,0],[0,0,1,0]]
    """

    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    """
    Source:
    Tim O'Shea

    Description:
    Plot a confusion matrix.
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def strided_matrix(x, w, s=1):
    """
    Inputs:
        x - Input signal
        w - Width of output segements
        s - Stride between the start samples of each captured sequence

    Returns:
        A matrix where rows contain time continuous sequences
    """

    # Shape of the output matrix
    shape = ((x.shape[0] - w) / s + 1, w)
    #shape = ((x.shape[0] - w) / s, w)

    # Using strides allows this to compile in C
    strides = (x.strides[0]*s, x.strides[0])
    
    return as_strided(x, shape, strides)  


def nextpow2(i):
    """
    Input
        i : The lowest acceptable number

    Returns
        The smallest number which is larger than i, while being a
        power of 2.
    """

    n = 1
    while n < i: n *= 2

    return int(n)

def quant_encode(x, n, v):
    """
    Quantize and encode floating-point inputs to integer outputs

    Source:
        https://github.com/rikrd/matlab/blob/master/signal/signal/uencode.m

    Inputs:
        x : numpy array of floating point values which need quantized and encoded.
        n : bit depth of the output array [-2^n/2, (2^n/2)-1]
        v : range of floating point numbers in x [-v,v]

    Returns:
        an array of signed integer values  in the range [-2^n/2, (2^n/2)-1]
    """

    # Attempt to ensure data is a numpy array
    if type(x).__module__ != np.__name__:
        print("Make sure input data is in a numpy array.")
        return 0

    # Capture data
    u = x

    # Range of data (actual range is 2*V as v is assumed to be symetric about zero)
    V = v

    # Quantization steps
    Q = 2**n - 1

    # Size of each step needed to cover range
    T = (Q+1)/(2.*V)

    # Raise the input values to [0,2*V] then multiply by the step size
    u = (u+V)*T

    # Set the min/max values of the output array
    ma = 2**(n-1)-1
    mi = -2**(n-1)

    # shift input array to meet output min
    u = u + mi

    # where data is above or below the min/max, saturate.
    u[u < mi] = mi
    u[u > ma] = ma

    # Quantize
    u = np.int32(np.floor(u))

    return u

def downsample(x,M,p=0):
    """

    Source:
        Mark Wickert - Digital Communications Function Module
    """

    x = x[0:int(np.floor(len(x)/float(M))*M)]
    x = x.reshape((int(len(x)/float(M)),int(M)))
    y = x[:,p]
    return y