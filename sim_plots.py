import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


zeta = 2.0
theta0 = 1.0
ratio = 0.75
delta = 0.01
fmt = '_zeta'+"{:.2f}".format(zeta) +'_l1_ratio'+"{:.2f}".format(ratio)+ '_delta' + "{:.3f}".format(delta)
method = 'amp'

rs_df = pd.read_csv('data/rs'+fmt+'.csv')
sim_df = pd.read_csv('data/sim'+fmt+'_method_'+method+'.csv')

directory = 'figures/'


fig1 = plt.figure()
ax1 = fig1.add_subplot(311)
ax2 = fig1.add_subplot(312)
ax3 = fig1.add_subplot(313)

ax1.errorbar(sim_df['alphas'],sim_df['hat_w_mean'],yerr =sim_df['hat_w_std'],fmt = 'k.', capsize = 3)
ax1.plot(rs_df['alphas'],rs_df['hat_w'],'r-')
ax1.set_ylabel(r'$\hat{w}_n$')
ax1.set_xlim(left = min(rs_df['alphas']), right = 3.0)
# ax1.set_xlabel(r'$\alpha$')

ax2.errorbar(sim_df['alphas'],sim_df['hat_v_mean'],yerr =sim_df['hat_v_std'],fmt = 'k.', capsize = 3)
ax2.plot(rs_df['alphas'],rs_df['hat_v'],'r-')
ax2.set_ylabel(r'$\hat{v}_n$')
ax2.set_xlim(left = min(rs_df['alphas']), right = 3.0)
# ax2.set_xlabel(r'$\alpha$')

ax3.errorbar(sim_df['alphas'],sim_df['hat_tau_mean'],yerr =sim_df['hat_tau_std'],fmt = 'k.', capsize = 3)
ax3.plot(rs_df['alphas'],rs_df['hat_tau'],'r-')
ax3.set_ylabel(r'$\hat{\tau}_n$')
ax3.set_xlabel(r'$\alpha$')
ax3.set_xlim(left = min(rs_df['alphas']), right = 3.0)

plt.savefig(directory+'hat_order_parameters' + fmt +'method_'+method+'.png')

fig2 = plt.figure()
ax1 = fig2.add_subplot(311)
ax2 = fig2.add_subplot(312)
ax3 = fig2.add_subplot(313)

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

ax3.errorbar(sim_df['alphas'],sim_df['tau_mean'],yerr =sim_df['tau_std'],fmt = 'k.', capsize = 3)
ax3.plot(rs_df['alphas'],rs_df['tau'],'r-')
ax3.set_ylabel(r'$\tau_n$')
ax3.set_xlabel(r'$\alpha$')
ax3.set_xlim(left = min(rs_df['alphas']), right = 3.0)

plt.savefig(directory+'order_parameters' + fmt +'method_'+method+'.png')


fig3 = plt.figure()
ax1 = fig3.add_subplot(611)
ax2 = fig3.add_subplot(612)
ax3 = fig3.add_subplot(613)
ax4 = fig3.add_subplot(614)
ax5 = fig3.add_subplot(615)
ax6 = fig3.add_subplot(616)

ax1.errorbar(sim_df['alphas'],sim_df['hat_w_mean'],yerr =sim_df['hat_w_std'],fmt = 'k.', capsize = 3)
ax1.plot(rs_df['alphas'],rs_df['hat_w'],'r-')
ax1.set_ylabel(r'$\hat{w}_n$', fontsize = 10)
ax1.set_xlim(left = min(rs_df['alphas']), right = 3.0)
ax1.tick_params(labelbottom = False, bottom = False )  


ax2.errorbar(sim_df['alphas'],sim_df['hat_v_mean'],yerr =sim_df['hat_v_std'],fmt = 'k.', capsize = 3)
ax2.plot(rs_df['alphas'],rs_df['hat_v'],'r-')
ax2.set_ylabel(r'$\hat{v}_n$', fontsize = 10)
ax2.set_xlim(left = min(rs_df['alphas']), right = 3.0)
ax2.tick_params(labelbottom = False, bottom = False )  

ax3.errorbar(sim_df['alphas'],sim_df['hat_tau_mean'],yerr =sim_df['hat_tau_std'],fmt = 'k.', capsize = 3)
ax3.plot(rs_df['alphas'],rs_df['hat_tau'],'r-')
ax3.set_ylabel(r'$\hat{\tau}_n$', fontsize = 10)
ax3.set_xlim(left = min(rs_df['alphas']), right = 3.0)
ax3.tick_params(labelbottom = False, bottom = False )  

ax4.errorbar(sim_df['alphas'],sim_df['w_mean'],yerr =sim_df['w_std'],fmt = 'k.', capsize = 3)
ax4.plot(rs_df['alphas'],rs_df['w'],'r-')
ax4.set_ylabel(r'$w_n$', fontsize = 10)
ax4.set_xlim(left = min(rs_df['alphas']), right = 3.0)
ax4.tick_params(labelbottom = False, bottom = False )  

ax5.errorbar(sim_df['alphas'],sim_df['v_mean'],yerr =sim_df['v_std'],fmt = 'k.', capsize = 3)
ax5.plot(rs_df['alphas'],rs_df['v'],'r-')
ax5.set_ylabel(r'$v_n$', fontsize = 10)
ax5.set_xlim(left = min(rs_df['alphas']), right = 3.0)
ax5.tick_params(labelbottom = False, bottom = False )  

ax6.errorbar(sim_df['alphas'],sim_df['tau_mean'],yerr =sim_df['tau_std'],fmt = 'k.', capsize = 3)
ax6.plot(rs_df['alphas'],rs_df['tau'],'r-')
ax6.set_ylabel(r'$\tau_n$', fontsize = 10)
ax6.set_xlim(left = min(rs_df['alphas']), right = 3.0)

ax6.set_xlabel(r'$\alpha$', fontsize = 10)

plt.savefig(directory+'all_order_parameters' + fmt +'method_'+method+'.png')


plt.show()
