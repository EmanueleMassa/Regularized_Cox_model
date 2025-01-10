import numpy as np
from scipy.special import lambertw
import time
from numba import njit
from routines.funcs import st, na_est
import warnings
warnings.filterwarnings('ignore')

vl = np.vectorize(lambertw)


@njit
def cd_update(x, hess, c, alpha, eta, beta0):
    lp0 = x @ beta0
    beta1 = beta0
    p = len(beta0)
    for k in range(p):
        lp = x @ beta1
        I_k = x[:, k] @ (hess * x[:,k])
        xi_k = x[:, k] @ (hess * (lp -lp0))
        s_k = x[:, k] @ (hess - c)
        phi = s_k + xi_k - I_k * beta1[k]
        tau = I_k + eta
        beta1[k] = st(-phi, alpha)/tau
    return beta1


def cd_cox(eta, alpha, c, x, beta0, tau0,  max_its = 300, tol = 1.0e-8, verbose = False):
    its = 0
    err = 1.0
    flag = True
    p = len(beta0)
    n = len(c)
    zeta = p / n
    elp =  np.exp(x @ beta0)
    tic = time.time()
    while(err >= tol and flag):
        H = na_est(c, elp)
        hess = H * elp
        beta = cd_update(x, hess, c, alpha, eta, beta0)
        elp =  np.exp(x @ beta)
        err =  np.sqrt(sum((beta - beta0)**2) )
        beta0 = beta
        its = its + 1 
        if(np.isnan(err)):
            flag = False
            print('error is Nan')
        if(its >= max_its):
            flag = False
            if(verbose):
                print('CD not converged! alpha = '+str(alpha)+', error = ' + str(err))
    toc = time.time()
    if(flag and verbose):
        print('CD alpha = ' + str(alpha) + ', time elapsed = ' + str((toc-tic)/60) + ', its =' + str(its) )
    av_norm0_beta = np.mean(np.array(np.abs(beta0) > 1.0e-8, int))
    tau = compute_tau(zeta * av_norm0_beta , hess, zeta * eta, tau0)
    hat_tau =  tau / (av_norm0_beta - eta * tau)
    return beta0, hat_tau, tau, flag


@njit
def compute_tau(z, x, gamma, y):
    err = 1.0
    f = 1.0 - z - np.mean(1.0 / (1.0 + x * y)) + gamma * y
    while(err>1.0e-13):
        df = np.mean(x / (1.0 + x * y)**2) + gamma
        y = y - f / df
        f = 1.0 - z - np.mean(1.0 / (1.0 + x * y)) + gamma * y
        err = abs(f)
    return y