import numpy as np
import numpy.random as rnd

class surv_models:
    def __init__(self, A, beta, phi, rho, t1, t2, model):
        self.p = len(beta)
        self.beta = beta
        self.phi = phi
        self.rho = rho
        self.tau1 = t1
        self.tau2 = t2
        self.A = A
        self.model = model

    def bh(self, t):
        if(self.model == 'weibull'):
            return self.rho*np.exp(self.phi)*(t**(self.rho-1))
        if(self.model == 'log-logistic'):
            return self.rho*np.exp(self.phi)*(t**(self.rho-1))/(1.0+ np.exp(self.phi)*(t**self.rho))
        
    def ch(self, t):
        if(self.model == 'weibull'):
            return np.exp(self.phi)*(t**self.rho)
        if(self.model == 'log-logistic'):
            return np.log(1.0+ np.exp(self.phi)*(t**self.rho))

    def gen(self, n):
        X = rnd.multivariate_normal(mean=np.zeros(self.p), cov= self.A, size = n)
        lp = X @ self.beta
        u = rnd.random(size = n) 
        T0 = self.tau1 + u*(self.tau2-self.tau1)
        #sample the latent event times 
        u = rnd.random(size = n)
        if(self.model == 'weibull'):
            T1 = np.exp((np.log(-np.log(u))-lp-self.phi)/self.rho)
        if(self.model == 'log-logistic'):
            T1 = np.exp( (np.log(np.exp(-np.log(u)*np.exp(-lp))-1)-self.phi)/self.rho)
        #generate the observations 
        T = np.minimum(T1,T0)
        C = np.array(T1<T0,int)
        #order the observations by their event times
        idx = np.argsort(T)
        T = np.array(T)[idx]
        C = np.array(C,int)[idx]
        X = X[[idx],:][0,:,:]
        return T, C, X