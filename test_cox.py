import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
from numba import njit, vectorize
import time
from routines.surv_data_generator import surv_models
from routines.cox_routines import cox_model

def configuration_AMP(p, zeta, nu, theta0):
    #number of non-zero components
    s = int(nu * p)
    #sample size
    n = int(p/zeta)
    #Covariance matrix
    A0 = np.identity(p)/ p
    #true associations
    # active_comp = rnd.normal(size = s) * (2 * np.array(rnd.random(size = s) > 0.5, int) - np.ones(s))
    beta0 = rnd.normal(size = p) * np.array(rnd.random(size = p) < nu, float)#np.append(active_comp, np.zeros(p-s))
    beta0 =  theta0 * beta0 / np.sqrt((beta0@A0@beta0))  
    return beta0, A0, n

p = 2000
zeta = 2.0
nu = 0.005
theta0 = 1.0
phi0 = -np.log(2)
rho0 = 2.0
tau1 = 1.0
tau2 = 2.0
mu0 = np.zeros(p)
vals = np.exp(np.linspace(np.log(10.0), np.log(0.9), 100))
ratio = 0.75

beta0, A0, n = configuration_AMP(p, zeta, nu, theta0)

fmt = '_zeta'+"{:.2f}".format(zeta) +'_l1_ratio'+"{:.2f}".format(ratio) + '_delta' + "{:.3f}".format(nu)

gen_mod = surv_models(A0, beta0, phi0, rho0, tau1, tau2, 'log-logistic') 
T, C, X = gen_mod.gen(n)
T_test, C_test, X_test = gen_mod.gen(n)

#fit with Cox AMP
coxm = cox_model(vals, ratio)
coxm.fit(T, C, X, 'amp', eps = 0.9, verb_flag= True)
betas_amp = coxm.betas
flags_amp = coxm.flags
coxm.compute_Harrel_c_train()
train_err_amp = coxm.hc_index_train
coxm.compute_Harrel_c_test(T_test, C_test, X_test)
test_err_amp = coxm.hc_index_test
rs_loo_hc_amp = coxm.rs_loo_hc

#fit with Cox CD
coxm.fit(T, C, X, 'cd', verb_flag= True)
betas_cd = coxm.betas
coxm.compute_Harrel_c_train()
train_err_cd = coxm.hc_index_train
coxm.compute_Harrel_c_test(T_test, C_test, X_test)
test_err_cd = coxm.hc_index_test
rs_loo_hc_cd = coxm.rs_loo_hc


mse = np.sqrt(np.mean((betas_cd - betas_amp)**2, axis = 1))#/ np.sqrt(np.sum((betas_cd)**2, axis = 1))


plt.figure()
plt.title(r'$\|\mathbf{\beta}_{AMP} - \mathbf{\beta}_{CD}\|_{\infty}$')
plt.plot(vals, mse, 'r-')
plt.xlabel(r'$\rho$')
plt.xlim(left = min(vals), right = 7.0)
plt.savefig('figures/error_AMP_CD' + fmt + '.jpg')

plt.figure()
plt.title('Elbow Plot')
plt.plot(vals, betas_amp, 'r-')
plt.plot(vals, betas_cd, 'b-')
for j in range(len(flags_amp)):
    if(flags_amp[j]!=True):
        plt.axvline(x = vals[j])
plt.xlabel(r'$\rho$')
plt.xlim(left = min(vals), right = 7.0)
plt.savefig('figures/elbow_plot' + fmt + '.jpg')


plt.figure()
plt.title('C index train ')
plt.plot(vals, train_err_amp, 'r-')
plt.plot(vals, train_err_cd, 'b-')
plt.xlabel(r'$\rho$')
plt.xlim(left = min(vals), right = 7.0)
plt.savefig('figures/c_ind_train' + fmt + '.jpg')

plt.figure()
plt.title('C index test ')
plt.plot(vals, test_err_amp, 'r-')
plt.plot(vals, rs_loo_hc_amp, 'ro')
plt.plot(vals, test_err_cd, 'b-')
plt.plot(vals, rs_loo_hc_cd, 'bo')
plt.xlabel(r'$\rho$')
plt.xlim(left = min(vals), right = 7.0)
plt.savefig('figures/c_ind_test' + fmt + '.jpg')


fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.plot(vals, betas_amp, 'r-')
ax1.plot(vals, betas_cd, 'b-')
for j in range(len(flags_amp)):
    if(flags_amp[j]!=True):
        ax1.axvline(x = vals[j])
# ax1.set_xlabel(r'$\rho$', fontsize = 10)
ax1.set_xlim(left = min(vals), right = 7.0)
ax1.set_ylabel(r'$\mathbf{\beta}$', fontsize = 10)

ax2.plot(vals, test_err_amp, 'r-')
ax2.plot(vals, rs_loo_hc_amp, 'ro')
ax2.plot(vals, test_err_cd, 'b-')
ax2.plot(vals, rs_loo_hc_cd, 'bo')
ax2.set_xlabel(r'$\rho$', fontsize = 10)
ax2.set_xlim(left = min(vals), right = 7.0)
ax2.set_ylabel(r'${\rm HC}_{test}$', fontsize = 10)
plt.savefig('figures/elbow_and_c_ind_test' + fmt + '.jpg')
plt.show()