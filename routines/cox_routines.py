import numpy as np
import numpy.random as rnd 
from routines.cox_amp import amp_cox
from routines.cox_cd import cd_cox, compute_tau
from routines.funcs import c_index, na_est, breslow_est


#class that contains function to fit the cox model 
class cox_model:
    def __init__(self, vals, ratio):
        self.rho = vals
        self.ratio = ratio
        self.alphas = vals * ratio
        self.etas = vals * (1.0 -ratio)
        self.l = len(vals)

    def get_dimensions(self, t, c, x):
        idx = np.argsort(t)
        self.p = len(x[0,:])
        self.t = np.array(t)[idx]
        self.c = np.array(c,int)[idx]
        self.x = x[[idx],:][0,:,:]
        self.n = len(t)
        self.zeta = self.p / self.n
        self.betas = np.zeros((self.l,self.p))
        self.flags = np.zeros(self.l)
        self.hat_taus = np.zeros(self.l)
        self.taus = np.zeros(self.l)
        self.ws = np.zeros(self.l)
        self.vs = np.zeros(self.l)
        self.hat_ws = np.zeros(self.l)
        self.hat_vs = np.zeros(self.l)
        self.rs_loo_hc = np.zeros(self.l)
        self.times = np.zeros(self.l)

    def initialize(self):
        self.beta = np.zeros(self.p)
        self.tau = 0.0
        self.hat_tau = 0.0
        self.xi = self.x @ self.beta

    def fit(self, t, c, x, method, eps = 0.9, verb_flag = False, warm_start_amp = False, tolerance = 1.0e-7):
        cox_model.get_dimensions(self, t, c, x)
        cox_model.initialize(self)
        for j in range(self.l):
            eta = self.etas[j]
            alpha = self.alphas[j]
            if(method == 'cd'):
                self.beta, self.hat_tau, self.tau, flag, time = cd_cox(eta, alpha, self.c, self.x, self.beta, self.tau, tol = tolerance, verbose = verb_flag)
            if(method == 'amp'):
                if(warm_start_amp == False):
                    cox_model.initialize(self)
                self.beta, self.xi, self.hat_tau, self.tau, flag, time = amp_cox(eta, alpha, self.c, self.x, self.beta, self.xi, self.tau, self.hat_tau, eps, tol = tolerance, verbose = verb_flag)      
            self.flags[j] = flag
            self.betas[j,:] = self.beta
            self.taus[j] = self.tau
            self.hat_taus[j] = self.hat_tau
            self.times[j] = time
            self.ws[j], self.vs[j], self.hat_ws[j], self.hat_vs[j], self.rs_loo_hc[j] = cox_model.compute_observables(self)
        return 
    
    def compute_observables(self):
        lp = self.x @ self.beta
        elp = np.exp(lp)
        H = na_est(self.c, elp)
        dg = (H * elp - self.c)
        db_beta = self.beta - self.hat_tau * np.transpose(self.x) @ dg
        hat_v = self.hat_tau * np.sqrt( np.mean( dg ** 2 ) / self.zeta)
        hat_w = np.sqrt(max(np.mean(db_beta ** 2) - (hat_v ** 2), 0))
        lp_loo = lp + self.tau * dg
        gamma = np.mean(lp_loo ** 2)
        if(hat_w == 0):
            w = 0.0
        else:
            w = (gamma - self.hat_tau * np.mean(lp_loo * dg) / self.zeta) / hat_w
        #     x = self.zeta * self.tau / self.hat_tau
        #     w = (np.mean(lp * lp_loo) - gamma * (1.0 - x)) / (hat_w * x)

        v = np.sqrt(max(gamma - (w ** 2), 0))
        rs_loo_hc = c_index(self.t, self.c, lp_loo)
        return w, v, hat_w, hat_v, rs_loo_hc
    
    def compute_Harrel_c_train(self):
        self.hc_index_train = np.array([c_index(self.t, self.c, self.x @ self.betas[j, :]) for j in range(self.l)], float)
        return   

    def compute_Harrel_c_test(self, T_test, C_test, X_test):
        self.hc_index_test = np.array([c_index(T_test, C_test, X_test @ self.betas[j, :]) for j in range(self.l)], float)
        return  


    

