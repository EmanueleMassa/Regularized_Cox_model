import numpy as np
from routines.cox_routines import cox_model
from routines.surv_data_generator import surv_models
from routines.cox_rs_routines import gauss_model
from experiments_routines import run_rs, run_sim, isotropic_setting
import pandas as pd 
#number of covariates
p = 2000
#overfitting ratio
zeta = 2.0
#number of covariates
n = int(p / zeta)
#sparsity
delta = 0.01
#signal strength 
theta0 = 1.0



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
values = np.exp(np.linspace(np.log(10.0), np.log(0.5), 100))
#l1_ratio
ratio = 0.75

#label for the csv files
fmt = '_zeta'+"{:.2f}".format(zeta) +'_l1_ratio'+"{:.2f}".format(ratio) 
fmt = fmt + '_delta' + "{:.3f}".format(delta)

#solve the rs equations
m = 5000   #population size
rs_df = run_rs(values, ratio, delta, zeta, gauss_process, m)
rs_df.to_csv('data/rs' + fmt + '.csv', index = False)


# #simulate the data and perform regressions with COX - AMP
m = 50 #number of repetitions to compute average
amp_sim_df = run_sim(p, n, values, ratio, data_gen_process, 'amp', m)
amp_sim_df.to_csv('data/sim' + fmt + '_method_amp.csv', index = False)

#simulate the data and perform regressions with COX - CD
cd_sim_df = run_sim(p, n, values, ratio, data_gen_process, 'cd', m)
cd_sim_df.to_csv('data/sim' + fmt + '_method_cd.csv', index = False)