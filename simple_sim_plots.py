import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


zeta = 2.0
theta0 = 1.0
ratio = 0.75
delta = 0.01
fmt = '_zeta'+"{:.2f}".format(zeta) +'_l1_ratio'+"{:.2f}".format(ratio)+ '_delta' + "{:.3f}".format(delta)
method = 'cd'

rs_df = pd.read_csv('data/rs'+fmt+'.csv')
sim_df = pd.read_csv('data/simple_sim'+fmt+'_method_'+method+'.csv')

directory = 'figures/'

plt.figure()
plt.errorbar(sim_df['alphas'],sim_df['w_mean'],yerr =sim_df['w_std'],fmt = 'ko', capsize = 3)
plt.plot(rs_df['alphas'],rs_df['w'],'r-')
plt.ylabel(r'$w$')
plt.xlabel(r'$\alpha$')
plt.savefig(directory+'simple_sim_w'+fmt+'method_'+method+'.jpg')

plt.figure()
plt.errorbar(sim_df['alphas'],sim_df['v_mean'],yerr =sim_df['v_std'],fmt = 'ko', capsize = 3)
plt.plot(rs_df['alphas'],rs_df['v'],'r-')
plt.ylabel(r'$v$')
plt.xlabel(r'$\alpha$')
plt.savefig(directory+'simple_sim_v'+fmt+'method_'+method+'.jpg')

fig2 = plt.figure()
ax1 = fig2.add_subplot(211)
ax2 = fig2.add_subplot(212)

ax1.errorbar(sim_df['alphas'],sim_df['w_mean'],yerr =sim_df['w_std'],fmt = 'k.', capsize = 3)
ax1.plot(rs_df['alphas'],rs_df['w'],'r-')
ax1.set_ylabel(r'$w_n$')
ax1.set_xlim(left = min(rs_df['alphas']), right = 3.0)
# ax1.set_xlabel(r'$\alpha$')

ax2.errorbar(sim_df['alphas'],sim_df['v_mean'],yerr =sim_df['v_std'],fmt = 'k.', capsize = 3)
ax2.plot(rs_df['alphas'],rs_df['v'],'r-')
ax2.set_ylabel(r'$v_n$')
ax2.set_xlim(left = min(rs_df['alphas']), right = 3.0)
# ax2.set_xlabel(r'$\alpha$')

plt.savefig(directory+'simple_sim_v_and_w' + fmt +'method_'+method+'.png')

plt.show()
