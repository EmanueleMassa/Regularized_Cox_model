import numpy as np
import numpy.random as rnd
from scipy.special import lambertw
from scipy.special import erf
from routines.funcs import na_est, c_index
import warnings
warnings.filterwarnings('ignore')

vl = np.vectorize(lambertw)

def Q(x):
    return 0.5 - 0.5*erf(x/np.sqrt(2))

def G(x):
    return np.exp(-0.5*x*x)/np.sqrt(2 * np.pi)



class rs_cox:

    def __init__(self, delta, zeta, gm, m):
        self.t, self.c, self.z0, self.q = gm.data_gen(m)
        self.m = m
        self.hat_w = 0.0
        self.hat_v = 1.0e-3
        self.hat_tau = 1.0e-3
        self.delta = delta
        self.zeta = zeta
        self.alpha = 0.0
        self.eta = 0.0
        self.H = na_est(self.c, np.ones(self.m))
        self.theta0 = gm.theta
        self.dg0 = gm.ch(self.t) * np.exp(self.theta0 * self.z0) - self.c

    
    def solve(self, damp = 0.5):
        err = 1.0
        its = 0
        hat_w0 = self.hat_w
        hat_v0 = self.hat_v
        hat_tau0 = self.hat_tau
        H0 = self.H
        while(err>1.0e-8):
            x1 = self.alpha * hat_tau0 / np.sqrt(hat_w0**2 / self.delta + hat_v0**2)
            x2 = self.alpha * hat_tau0 / hat_v0
            w = 2 * hat_w0 * Q(x1) / (1.0 + self.eta * hat_tau0)
            tau = 2 * hat_tau0 * (self.delta * Q(x1) + 
                                  (1-self.delta) * Q(x2) )/ (1.0 + self.eta * hat_tau0)
            f =  (self.delta * (Q(x1) * (1.0 / x1**2 + 1) - G(x1) / x1) + 
                  (1-self.delta) * (Q(x2)* ( 1.0 /x2**2 + 1) - G(x2) / x2))
            v = np.sqrt(max( 2 * (self.alpha * hat_tau0)**2 * f/ (1 + self.eta * hat_tau0)**2 - w**2, 0 ))
            y = tau * self.c + w * self.z0 + v * self.q
            chi = np.array(vl(tau * H0 * np.exp(y)),float)
            xi =  y - chi 
            exi =np.exp(xi)
            ddg_xi = H0 * exi
            dg_xi = ddg_xi - self.c
            hat_tau1 = self.zeta  / np.mean(ddg_xi / (1.0 + tau * ddg_xi))
            hat_w1 = self.theta0 * hat_tau0 * np.mean(dg_xi * self.dg0) / self.zeta
            hat_v1 = hat_tau0 * np.sqrt(np.mean(dg_xi ** 2) / self.zeta)
            H1 = na_est(self.c, np.exp(xi))
            hat_w = damp * hat_w1 + (1.0 - damp) * hat_w0
            hat_v = damp * hat_v1 + (1.0 - damp) * hat_v0
            hat_tau = damp * hat_tau1 + (1.0 - damp) * hat_tau0
            H = damp * H1 + (1.0 - damp) * H0
            err = np.sqrt( (hat_v-hat_v0)**2 + (hat_tau-hat_tau0)**2  + (hat_w-hat_w0)**2 + (H-H0)@(H-H0))
            its = its + 1
            hat_v0 = hat_v
            hat_tau0 = hat_tau
            hat_w0 = hat_w
            H0 = H
            # print(self.alpha,tau,w,v,err,its)
        self.hat_w = hat_w0
        self.hat_v = hat_v0
        self.hat_tau = hat_tau0
        self.H = H0
        self.w = w
        self.v = v
        self.tau = tau
        self.xi = xi
        lp_loo = self.w * self.z0 + self.v * self.q
        self.rs_loo_hc = c_index(self.t, self.c, lp_loo)
        return 
    

class gauss_model:
    def __init__(self, theta, phi, rho, t1, t2, model):
        self.theta = theta
        self.phi = phi
        self.rho = rho
        self.tau1 = t1
        self.tau2 = t2
        self.model = model

    def bh(self,t):
        if(self.model == 'weibull'):
            return self.rho*np.exp(self.phi)*(t**(self.rho-1))
        if(self.model == 'log-logistic'):
            return self.rho*np.exp(self.phi)*(t**(self.rho-1))/(1.0+ np.exp(self.phi)*(t**self.rho))
        
    def ch(self,t):
        if(self.model == 'weibull'):
            return np.exp(self.phi)*(t**self.rho)
        if(self.model == 'log-logistic'):
            return np.log(1.0+ np.exp(self.phi)*(t**self.rho))

    def data_gen(self, n):
        #generate the data
        Z0 = rnd.normal(size = n)
        Q = rnd.normal(size = n)
        u = rnd.random(size = n)
        lp = self.theta * Z0
        if(self.model == 'weibull'):
            T1 = np.exp((np.log(-np.log(u))-lp-self.phi)/self.rho)
        if(self.model == 'log-logistic'):
            T1 = np.exp( (np.log(np.exp(-np.log(u)*np.exp(-lp))-1)-self.phi)/self.rho)
        if(self.model != 'weibull' and self.model != 'log-logistic'):
            raise TypeError("Only weibull and log-logistic are available at the moment") 
        u = rnd.random(size = n)
        T0 = (self.tau2 - self.tau1) * u + self.tau1
        T = np.minimum(T1,T0)
        C = np.array(T1<T0,int)
        #order the observations by their event times
        idx = np.argsort(T)
        T = np.array(T)[idx]
        C = np.array(C,int)[idx]
        Z0 = np.array(Z0)[idx]
        return T, C, Z0, Q
    
