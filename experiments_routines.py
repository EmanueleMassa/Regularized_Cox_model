import numpy as np
import numpy.random as rnd
from routines.cox_rs_routines import rs_cox
from routines.cox_routines import cox_model
from joblib import Parallel, delayed
import time
import pandas as pd 

def run_rs(values, ratio, delta, zeta, gm, m):
    metrics = np.empty((len(values),7))
    cox_rs = rs_cox(delta, zeta, gm, m)
    # loop over the values of lambda
    for l in range(len(values)):
        cox_rs.alpha = values[l] * ratio
        cox_rs.eta = values[l] * (1-ratio)
        cox_rs.solve()
        res = np.array([values[l], cox_rs.w, cox_rs.v, cox_rs.tau, cox_rs.hat_w, cox_rs.hat_v, cox_rs.hat_tau], float)
        print(res)
        metrics[l,:] = res
    df = pd.DataFrame(metrics, columns=['vals', 'w', 'v', 'tau', 'hat_w', 'hat_v', 'hat_tau'])
    return df
    

def run_sim(p, n, values, ratio, GM, method, m, parallel_flag = False):

    def experiment(counter, GM, cox_m, method, n):
        tic = time.time()
        #generate data
        T, C, X = GM.gen(n)
        cox_m.fit(T, C, X, method, verb_flag = True)
        toc = time.time()
        print('experiment '+str(counter)+ ' time elapsed = '+str((toc-tic)/60))
        return cox_m.ws, cox_m.vs, cox_m.taus, cox_m.hat_ws, cox_m.hat_vs, cox_m.hat_taus
    
    cox_m = cox_model(p, values, ratio)

    if(parallel_flag):
        tic = time.time()
        results = Parallel(n_jobs=24)(delayed(experiment)(counter, GM, cox_m, method, n) for counter in range(m))
        t_df = pd.DataFrame(results)
        w = np.stack(t_df.iloc[:, 0].to_numpy())
        v = np.stack(t_df.iloc[:, 1].to_numpy())
        tau = np.stack(t_df.iloc[:, 2].to_numpy())
        hat_w = np.stack(t_df.iloc[:, 3].to_numpy())
        hat_v = np.stack(t_df.iloc[:, 4].to_numpy())
        hat_tau = np.stack(t_df.iloc[:, 5].to_numpy())
        toc = time.time()
        print('total elapsed time = ' + str((toc-tic)/60))

    else:
        w = np.zeros((m, len(values)))
        v= np.zeros((m, len(values)))
        tau = np.zeros((m, len(values)))
        hat_w = np.zeros((m, len(values)))
        hat_v = np.zeros((m, len(values)))
        hat_tau = np.zeros((m, len(values)))

        big_tic = time.time()
        for i in range(m):
            w[i,:], v[i,:], tau[i,:], hat_w[i,:], hat_v[i,:], hat_tau[i,:] = experiment(i, GM, cox_m, method, n)
        big_toc = time.time()
        print('total elapsed time = ' + str((big_toc-big_tic)/60))

    data = {
        'vals' : values,
        'w_mean' : np.mean(w, axis = 0),
        'w_std' : np.std(w, axis = 0),
        'v_mean' : np.mean(v, axis = 0),
        'v_std' : np.std(v, axis = 0),
        'tau_mean' : np.mean(tau, axis = 0),
        'tau_std' : np.std(tau, axis = 0),
        'hat_w_mean' : np.mean(hat_w, axis = 0),
        'hat_w_std' : np.std(hat_w, axis = 0),
        'hat_v_mean' : np.mean(hat_v, axis = 0),
        'hat_v_std' : np.std(hat_v, axis = 0),
        'hat_tau_mean' : np.mean(hat_tau, axis = 0),
        'hat_tau_std' : np.std(hat_tau, axis = 0)
    }
    df = pd.DataFrame(data)
    return df 

def isotropic_setting(p, nu, theta0):
    beta0 = rnd.normal(size = p)* np.array(rnd.random(size = p)<nu, int)
    beta0 = beta0 / np.sqrt(beta0 @ beta0)
    beta0 = theta0 * np.sqrt(p) * beta0
    #generate covariance matrix
    Sigma0 = np.diag(np.ones(p))/p
    return Sigma0, beta0