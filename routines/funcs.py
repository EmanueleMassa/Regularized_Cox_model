import numpy as np
from numba import njit, vectorize


@vectorize
def st(x, a):
    y = 0.
    if(x>a):
        y = (x-a)
    if(x<-a):
        y = (x+a)
    if(a==0):
        y = x
    return y


@vectorize
def dst(x, a):
    y = 0.
    if((x > a) or (x<-a)):
        y = 1.
    if(a == 0):
        y = 1.
    return y

@njit
def na_est(c, elp):
    n = len(c)
    ch = np.zeros(n)
    R = sum(elp)
    if c[0]==1:
        ch[0] = 1.0/R
    else:
        ch[0] = 0 
    for i in range(1, n):
        ch[i] = ch[i-1]
        R = R - elp[i-1]
        if c[i]==1:
            ch[i] = ch[i] + 1.0/R
    return ch


def c_index(t, c, lp):
    den = 0 
    c_ind = 0
    gamma_n = np.sqrt(np.mean(lp**2))
    if(gamma_n <1.0e-5):
        return 0.5
    for i in range(len(t)):
        a = c * np.array(t < t[i], int)
        den = den + np.mean(a)
        c_ind = c_ind + np.mean( a * np.array(lp > lp[i], int))
    return c_ind / den

def breslow_est(c, elp):
    n = len(elp)
    bh = np.zeros(n)
    R = sum(elp)
    if(c[0] == 1):
        bh[0] = 1.0 / R
    for i in range(1, n):
        R = R - elp[i-1]
        if(c[i] == 1):
            bh[i] = 1.0 / R
    return bh