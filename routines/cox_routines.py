import numpy as np
from routines.cox_amp import amp_cox
from routines.cox_cd import cd_cox, compute_tau
from routines.funcs import c_index, na_est


#class that contains function to fit the cox model 
class cox_model:
    def __init__(self, p, vals, ratio):
        self.p = p
        self.alphas = vals * ratio
        self.etas = vals * (1.0 -ratio)
        self.l = len(vals)

    def fit(self, t, c, x, method, eps = 0.5, verb_flag = False):
        idx = np.argsort(t)
        self.t = np.array(t)[idx]
        self.c = np.array(c,int)[idx]
        self.x = x[[idx],:][0,:,:]
        self.n = len(t)
        self.zeta = self.p / self.n
        beta = np.zeros(self.p)
        tau = 0.0
        hat_tau = 0.0
        xi = self.x @ beta
        self.betas = np.zeros((self.l,self.p))
        self.flags = np.zeros(self.l)
        self.hat_taus = np.zeros(self.l)
        self.taus = np.zeros(self.l)
        self.ws = np.zeros(self.l)
        self.vs = np.zeros(self.l)
        self.hat_ws = np.zeros(self.l)
        self.hat_vs = np.zeros(self.l)
        for j in range(self.l):
            eta = self.etas[j]
            alpha = self.alphas[j]
            if(method == 'cd'):
                beta, hat_tau, tau, flag = cd_cox(eta, alpha, self.c, self.x, beta, tau, verbose = verb_flag)
            if(method == 'amp'):
                beta, xi, hat_tau, tau, flag = amp_cox(eta, alpha, self.c, self.x, beta, xi, tau, hat_tau, eps, verbose = verb_flag)
            self.flags[j] = flag
            self.betas[j,:] = beta
            self.taus[j] = tau
            self.hat_taus[j] = hat_tau
            self.ws[j], self.vs[j], self.hat_ws[j], self.hat_vs[j] = cox_model.compute_observables(self, beta, hat_tau, tau)
        return 
    
    def compute_observables(self, beta, hat_tau, tau):
        lp = self.x @ beta
        elp = np.exp(lp)
        H = na_est(self.c, elp)
        score = (H * np.exp(lp) - self.c)
        db_beta = beta - hat_tau * np.transpose(self.x) @ score
        hat_v = hat_tau * np.sqrt( np.mean( score ** 2 ) / self.zeta)
        hat_w = np.sqrt(max(np.mean(db_beta**2) - hat_v**2, 0))
        gamma = np.mean((lp + tau * score)**2)
        w = (np.mean(lp * (lp + tau * score)) - gamma * (1.0 - self.zeta * tau / hat_tau)) / (hat_w * self.zeta * tau /hat_tau)
        v = np.sqrt(max(gamma - w**2, 0))
        return w, v, hat_w, hat_v
    
    def compute_Harrel_c_train(self):
        self.hc_index_train = np.array([c_index(self.t, self.c, self.x @ self.betas[j, :]) for j in range(self.l)], float)
        return   

    def compute_Harrel_c_test(self, T_test, C_test, X_test):
        self.hc_index_test = np.array([c_index(T_test, C_test, X_test @ self.betas[j, :]) for j in range(self.l)], float)
        return  

