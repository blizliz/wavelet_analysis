import matplotlib.pyplot as plt
import numpy as np

SCALE = 3
N = 2 ** SCALE

def f1(x):
    return np.sin(2*x)

def f2(x):
    if x <= 10:
        return np.cos(x) + np.sin(2*x)
    else:
        return np.cos(x*0.3) + np.sin(6*x)

def f3(x):
    return np.sin(x**2)

def f4():
    return np.sin(2*np.pi*(2**np.linspace(2,10,N))*np.arange(N)/48000) + np.random.normal(0, 1, N) * 0.15

def check_level(level):
    max_level = int(np.log2(N))
    if level is None:
        return max_level
    elif level < 0:
        raise ValueError(
            "Level value of %d is too low . Minimum level is 0." % level)
    elif level > max_level:
        print(("Level value of {0} is too high. The maximum level value of {1} will be used.").format(level, max_level))
        return max_level
    return level

def dwt(data):
    size = len(data) // 2
    cA = np.zeros(size)
    cD = np.zeros(size)
    
    for i, j in zip(range(0, len(data), 2), range(size)):
        c = 2 * (data[i] + data[i + 1]) / np.sqrt(N)
        cA[j] = c

    for i, j in zip(range(0, len(data), 2), range(size)):
        c = 2 * (data[i] - data[i + 1]) / np.sqrt(N)
        cD[j] = c

    return cA, cD

def wavedec(data, level=None):
    '''
    Multilevel 1D Discrete Wavelet Transform of data.

    Parameters
    ----------
    data: array_like
        Input data
    level : int, optional
        Decomposition level (must be >= 0). If level is None (default) then it
        will be calculated using the `dwt_max_level` function.

    Returns
    -------
    [cA_n, cD_n, cD_n-1, ..., cD2, cD1] : list
        Ordered list of coefficients arrays
        where `n` denotes the level of decomposition. The first element
        (`cA_n`) of the result is approximation coefficients array and the
        following elements (`cD_n` - `cD_1`) are details coefficients arrays.
    '''

    coeffs_list = []
    
    level = check_level(level)
    if level == 0:
        return [np.array(data)]
    a = data
    for i in range(level):
        a, d = dwt(a)
        coeffs_list.append(d)

    coeffs_list.append(a)
    coeffs_list.reverse()

    return coeffs_list

def idwt(a, d):
    res = []
    for i in range(len(a)):
        x = (a[i] + d[i]) * np.sqrt(N) / 4
        y = (a[i] - d[i]) * np.sqrt(N) / 4
        res.extend([x, y])
    return np.array(res)

def waverec(coeffs):
    """
    Multilevel 1D Inverse Discrete Wavelet Transform.

    Parameters
    ----------
    coeffs : array_like
        Coefficients list [cAn, cDn, cDn-1, ..., cD2, cD1]
    """
    if len(coeffs) < 1:
        raise ValueError(
            "Coefficient list too short (minimum 1 arrays required).")
    elif len(coeffs) == 1:
        # level 0 transform (just returns the approximation coefficients)
        return coeffs[0]

    a, ds = coeffs[0], coeffs[1:]

    for d in ds:
        a = idwt(a, d)

    return a  