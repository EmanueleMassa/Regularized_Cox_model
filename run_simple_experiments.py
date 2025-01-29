import numpy as np
from routines.cox_routines import cox_model
from routines.surv_data_generator import surv_models
from routines.cox_rs_routines import gauss_model
from experiments_routines import isotropic_setting
from joblib import Parallel, delayed
import pandas as pd 
import time 
#number of covariates
p = 2000
#overfitting ratio
zeta = 2.0
#number of covariates
n = int(p / zeta)
#sparsity
delta = 0.005
#signal strength 
theta0 = 1.0

method = 'amp'

#define the population covariance matrix and the true beta
A0, beta0 = isotropic_setting(p, delta, theta0)
#define the true parameters of the cumulative hazard
phi0 = - np.log(2)
rho0 = 2.0
model = 'log-logistic'
#define the interval in which the censoring is uniform
tau1 = 1.0
tau2 = 2.0
#data generating process
data_gen_process = surv_models(A0, beta0, phi0, rho0, tau1, tau2, model) 
#equivalent gaussian generating process
gauss_process = gauss_model(theta0, phi0, rho0, tau1, tau2, model)

#lambda values  
values = np.exp(np.linspace(np.log(10.0), np.log(0.9), 100))
#l1_ratio
ratio = 0.95

def run_sim_simple(p, n, values, ratio, GM, method, m, parallel_flag = False):
    def experiment(counter, GM, cox_m, method, n):
        tic = time.time()
        #generate data
        T, C, X = GM.gen(n)
        cox_m.fit(T, C, X, method, verb_flag = True)
        #compute cosine similarity along the path
        w = np.mean(cox_m.betas * GM.beta, axis = 1) / np.sqrt(GM.beta @ GM.beta / p)
        #compute average mse beta
        v = np.sqrt(np.mean((cox_m.betas)**2, axis = 1) - w**2)
        toc = time.time()
        print('simple_experiment '+str(counter)+ ' time elapsed = '+str((toc-tic)/60))
        return w, v

    cox_m = cox_model(values, ratio)

    if(parallel_flag):
        tic = time.time()
        results = Parallel(n_jobs=12)(delayed(experiment)(counter, GM, cox_m, method, n) for counter in range(m))
        t_df = pd.DataFrame(results)
        w = np.stack(t_df.iloc[:, 0].to_numpy())
        v = np.stack(t_df.iloc[:, 1].to_numpy())
        toc = time.time()
        print('total elapsed time = ' + str((toc-tic)/60))

    else:
        w = np.zeros((m, len(values)))
        v= np.zeros((m, len(values)))
        big_tic = time.time()
        for i in range(m):
            w[i,:], v[i,:] = experiment(i, GM, cox_m, method, n)
        big_toc = time.time()
        print('total elapsed time = ' + str((big_toc-big_tic)/60))

    data = {
        'vals' : values,
        'w_mean' : np.mean(w, axis = 0),
        'w_std' : np.std(w, axis = 0),
        'v_mean' : np.mean(v, axis = 0),
        'v_std' : np.std(v, axis = 0)
    }
    df = pd.DataFrame(data)
    return df

sim = run_sim_simple(p, n, values, ratio, data_gen_process, method, 20, parallel_flag = True)
fmt = '_zeta'+"{:.2f}".format(zeta) +'_l1_ratio'+"{:.2f}".format(ratio) 
fmt = fmt + '_delta' + "{:.3f}".format(delta)
sim.to_csv('data/simple_sim' + fmt + '_method_'+ method + '.csv', index = False)