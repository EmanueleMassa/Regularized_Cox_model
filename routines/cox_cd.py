import numpy as np
from scipy.special import lambertw
import time
from numba import njit
from routines.funcs import st, na_est
import warnings
warnings.filterwarnings('ignore')

vl = np.vectorize(lambertw)



@njit
def cd_update(ddg, dg, x, alpha, eta, beta0):
    beta1 = beta0
    hess = np.transpose(x) @ np.diag(ddg) @ x
    score = dg @ x
    p = len(beta0)
    for k in range(p):
        I_k = hess[k, k]
        xi_k =  hess[k, :] @ (beta1 - beta0)
        s_k = score[k]
        phi =  I_k * beta1[k] - s_k - xi_k
        tau = I_k + eta
        beta1[k] = st(phi, alpha)/tau
    return beta1

def cd_cox(eta, alpha, c, x, beta0, tau0,  max_its = 300, tol = 1.0e-6, verbose = False):
    its = 0
    err = 1.0
    flag = True
    p = len(beta0)
    n = len(c)
    zeta = p / n
    elp =  np.exp(x @ beta0)
    H0 = na_est(c, elp)
    tic = time.time()
    while(err >= tol and flag):
        ddg = H0 * elp
        dg = ddg - c
        beta = cd_update(ddg, dg, x, alpha, eta, beta0)
        # beta = cd_update(x, ddg, c, alpha, eta, beta0)
        elp =  np.exp(x @ beta)
        H = na_est(c, elp)
        err =  np.sqrt(max((beta - beta0) ** 2) + max((H - H0) ** 2))
        beta0 = beta
        H0 = H
        its = its + 1 
        if(np.isnan(err)):
            flag = False
            print('error is Nan')
        if(its >= max_its):
            flag = False
            if(verbose):
                print('CD not converged! alpha = '+str(alpha)+', error = ' + str(err))
    toc = time.time()
    elapsed_time = (toc - tic) / 60
    if(flag and verbose):
        print('CD alpha = ' + str(alpha) + ', time elapsed = ' + str(elapsed_time) + ', its =' + str(its) )
    av_norm0_beta = np.mean(np.array(np.abs(beta0) > tol, int))
    tau = compute_tau(zeta * av_norm0_beta , ddg, zeta * eta, tau0)
    if(tau == 0.0):
        hat_tau = zeta / np.mean(H * elp)
    else:
        hat_tau =  tau / (av_norm0_beta - eta * tau)
    return beta0, hat_tau, tau, flag, elapsed_time


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

