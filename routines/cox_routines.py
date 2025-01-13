import numpy as np
import numpy.random as rnd 
from routines.cox_amp import amp_cox
from routines.cox_cd import cd_cox, compute_tau
from routines.funcs import c_index, na_est, breslow_est


#class that contains function to fit the cox model 
class cox_model:
    def __init__(self, p, vals, ratio):
        self.p = p
        self.rho = vals
        self.ratio = ratio
        self.alphas = vals * ratio
        self.etas = vals * (1.0 -ratio)
        self.l = len(vals)

    def fit(self, t, c, x, method, eps = 0.5, verb_flag = False, warm_start_amp = False, tolerance = 1.0e-8):
        idx = np.argsort(t)
        self.t = np.array(t)[idx]
        self.c = np.array(c,int)[idx]
        self.x = x[[idx],:][0,:,:]
        self.n = len(t)
        cox_model.compute_rho_max(self)
        self.zeta = self.p / self.n
        beta = np.zeros(self.p)
        tau = 0.0
        hat_tau = 0.0
        xi = self.x @ beta
        beta_in = np.zeros(self.p)#rnd.normal(loc = 0, scale = 1  / self.p, size = self.p)
        xi_in = self.x @ beta#rnd.normal(loc = 0, scale = 1 / self.p, size = self.n)
        hat_tau_in = 0.0#rnd.normal(loc = 0, scale = 1 / self.p)
        tau_in = 0.0#rnd.normal(loc = 0, scale = 1 / self.p)        
        self.betas = np.zeros((self.l,self.p))
        self.flags = np.zeros(self.l)
        self.hat_taus = np.zeros(self.l)
        self.taus = np.zeros(self.l)
        self.ws = np.zeros(self.l)
        self.vs = np.zeros(self.l)
        self.hat_ws = np.zeros(self.l)
        self.hat_vs = np.zeros(self.l)
        self.dev_diffs = np.zeros(self.l)
        for j in range(self.l):
            eta = self.etas[j]
            alpha = self.alphas[j]
            if(method == 'cd'):
                beta, hat_tau, tau, flag = cd_cox(eta, alpha, self.c, self.x, beta, tau, tol = tolerance, verbose = verb_flag)
            if(method == 'amp'):
                if(warm_start_amp):
                    beta, xi, hat_tau, tau, flag = amp_cox(eta, alpha, self.c, self.x, beta, xi, tau, hat_tau, eps, tol = tolerance, verbose = verb_flag)
                else:      
                    beta, xi, hat_tau, tau, flag = amp_cox(eta, alpha, self.c, self.x, beta_in, xi_in, tau_in, hat_tau_in, eps, tol = tolerance, verbose = verb_flag)
            self.flags[j] = flag
            self.betas[j,:] = beta
            self.taus[j] = tau
            self.hat_taus[j] = hat_tau
            self.ws[j], self.vs[j], self.hat_ws[j], self.hat_vs[j], self.dev_diffs[j] = cox_model.compute_observables(self, beta, hat_tau, tau)
        return 
    
    def compute_observables(self, beta, hat_tau, tau):
        lp = self.x @ beta
        elp = np.exp(lp)
        H = na_est(self.c, elp)
        score = (H * elp - self.c)
        db_beta = beta - hat_tau * np.transpose(self.x) @ score
        hat_v = hat_tau * np.sqrt( np.mean( score ** 2 ) / self.zeta)
        hat_w = np.sqrt(max(np.mean(db_beta**2) - hat_v**2, 0))
        gamma = np.mean((lp + tau * score)**2)
        if(tau == 0.0):
            w = 0.0
        else:
            w = (np.mean(lp * (lp + tau * score)) - gamma * (1.0 - self.zeta * tau / hat_tau)) / (hat_w * self.zeta * tau /hat_tau)
        v = np.sqrt(max(gamma - w**2, 0))
        h = breslow_est(self.c, elp)
        loss_train = sum(H * elp - self.c * lp) - sum([np.log(h[i]) for i in range(self.n) if self.c[i] != 0]) 
        dev_diff = np.abs(loss_train - self.loss_null) / self.loss_null
        return w, v, hat_w, hat_v, dev_diff 
    
    def compute_Harrel_c_train(self):
        self.hc_index_train = np.array([c_index(self.t, self.c, self.x @ self.betas[j, :]) for j in range(self.l)], float)
        return   

    def compute_Harrel_c_test(self, T_test, C_test, X_test):
        self.hc_index_test = np.array([c_index(T_test, C_test, X_test @ self.betas[j, :]) for j in range(self.l)], float)
        return  
    
    def compute_rho_max(self):
        self.h_null = breslow_est(self.c, np.ones(self.n))
        self.H_null = na_est(self.c, np.ones(self.n))
        self.loss_null =  sum(self.H_null) - sum([np.log(self.h_null[i]) for i in range(self.n) if self.c[i] != 0])
        s_null = (self.H_null - self.c) @ self.x 
        self.rho_max = np.max(s_null) / self.ratio
        return 

    

