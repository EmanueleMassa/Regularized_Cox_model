import numpy as np
from scipy.special import lambertw
from routines.funcs import st, dst, na_est
import time
import warnings
warnings.filterwarnings('ignore')

vl = np.vectorize(lambertw)


def gamp_cox_update(zeta, eta, alpha, C, X, beta0, xi0, hat_tau0, tau0, H0, eps):
    z = xi0 + tau0 * C - np.array(vl(tau0 * H0 * np.exp(xi0 + tau0 * C)), float)
    xi1 =  X @ beta0 + tau0 * (H0 * np.exp(z) - C)
    H1 = na_est(C, np.exp(z))
    z = xi1 + tau0 * C - np.array(vl(tau0 * H1 * np.exp(xi1 + tau0 * C)), float)
    hat_tau1 = zeta/ np.mean(H1 * np.exp(z) / (1.0 + tau0 * H1 * np.exp(z)))
    phi =  beta0 -  hat_tau1 * (np.transpose(X) @ (H1 * np.exp(z) - C))
    beta1 = st(phi, hat_tau1 * alpha) / (1.0  + hat_tau1 * eta)
    tau1 =  hat_tau1 * np.mean(dst(phi, hat_tau1 * alpha)) / (1.0 + hat_tau1 * eta) 
    beta = eps * beta1 + (1.0 - eps) * beta0
    tau =  eps * tau1  + (1.0 - eps) * tau0
    hat_tau =  eps * hat_tau1  + (1.0 - eps) * hat_tau0
    xi = eps * xi1 + (1.0 - eps) * xi0
    H = eps * H1 + (1.0 - eps) * H0
    return beta, xi, hat_tau, tau, H


def amp_cox(eta, alpha, C, X, beta0, xi0, tau0, hat_tau0, eps, max_its = 300, tol = 1.0e-8, verbose = False):
    n = len(C)
    p = len(beta0)
    zeta =  p / n
    H0 = na_est(C, np.exp(xi0))
    flag = True
    its = 0
    err = 1.0
    tic = time.time()
    while(err>= tol and flag):
        beta, xi, hat_tau, tau, H = gamp_cox_update(zeta, eta, alpha, C, X, beta0, xi0, hat_tau0, tau0, H0, eps)
        err = np.sqrt( sum((beta - beta0)**2)+ sum((xi-xi0)**2)+ (tau - tau0)**2 + (hat_tau - hat_tau0)**2 + sum((H- H0)**2) )
        beta0 = beta
        tau0 = tau
        hat_tau0 = hat_tau
        H0 = H
        xi0 = xi
        its = its + 1 
        if(np.isnan(err)):
            flag = False
            print('error dio cane')
        if(its >= max_its):
            flag = False
            if(verbose):
                print('AMP not converged! alpha = ' + str(alpha) + ', error = ' + str(err))
    toc = time.time()
    if(flag and verbose):
        print('AMP converged, alpha = '+str(alpha)+', time elapsed = '+str((toc-tic)/60) + ', its =' + str(its))
    return beta, xi, hat_tau, tau, flag
