# %%
import numpy as np
import xarray as xr





##--##--##--##--##--##--   HadISST  link  number  VS  Mestimated  (6 months)   --##--##--##--##--##--##
ds2 = xr.open_dataset("1_networks/part3_ersst/ersstv5_Mestimated6_2x2_sign_correct_wrong.nc")
link_number_all6 = ds2.link_number_all.data
relay_correct6 = ds2.link_number_relay_correct.data
relay_wrong6 = ds2.link_number_relay_wrong.data

for mfold in range(6): 
  relay_correct6[mfold,:] = relay_correct6[mfold,:]/link_number_all6
  relay_wrong6[mfold,:] = relay_wrong6[mfold,:]/link_number_all6

print(" 1 ",relay_correct6[0,6]," 2 ",relay_correct6[1,12]," 3 ",relay_correct6[2,18]," 4 ",relay_correct6[3,24]," 5 ",relay_correct6[4,30]," 6 ",relay_correct6[5,36])



##--##--##--##--##--##--   HadISST  link  number  VS  Mestimated  (4 months)   --##--##--##--##--##--##
ds2 = xr.open_dataset("1_networks/part3_ersst/ersstv5_Mestimated4_2x2_sign_correct_wrong.nc")
link_number_all4 = ds2.link_number_all.data
relay_correct4 = ds2.link_number_relay_correct.data
relay_wrong4 = ds2.link_number_relay_wrong.data

for mfold in range(6): 
  relay_correct4[mfold,:] = relay_correct4[mfold,:]/link_number_all4
  relay_wrong4[mfold,:] = relay_wrong4[mfold,:]/link_number_all4

print(" 1 ",relay_correct4[0,4]," 2 ",relay_correct4[1,8]," 3 ",relay_correct4[2,12]," 4 ",relay_correct4[3,16]," 5 ",relay_correct4[4,20]," 6 ",relay_correct4[5,24])




# %%
## Plotting Import
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('agg')



##--##--##--  create figure  --##--##--##
width_in_inches = 210 / 25.4  ## A4 page
height_in_inches = 297 / 25.4

fig = plt.figure(dpi=400,figsize=(width_in_inches,height_in_inches))





def create_map(ax):
  # plt.axes( xscale="log" )
  plt.xlim(-0.5,37.0)
  plt.ylim(0.0,1.25)
  ax.tick_params(axis='both', length=6.0, width=1.0, labelsize=9.5)
  ax.spines['bottom'].set_linewidth(1.0)
  ax.spines['top'].set_linewidth(1.0)
  ax.spines['left'].set_linewidth(1.0)
  ax.spines['right'].set_linewidth(1.0)
  ax.set_xticks([0,3,6,9,12,15,18,21,24,27,30,33,36]); ax.set_xticklabels(["","","","","","","","","","","","",""], color='black')
  ax.set_yticks(np.linspace(0.0,1.0,5)); ax.set_yticklabels(["","","","",""], color='black')
  ax.spines['left'].set_color('black')
  ax.tick_params(axis='y', right=False, length=7.0, width=1.0, labelsize=11.5, color='black')
  ax.tick_params(axis='x', right=False, length=7.0, width=1.0, labelsize=13.0, color='black')
  ax.set_xlabel('', size=10.5)
  ax.set_ylabel('', size=13.0)
  return ax

  


x = np.linspace(0,36,37)
y0 = np.linspace(0.0,0.0,37)
y7 = np.linspace(0.6,0.6,37)



ax3 = plt.axes([0.52,0.72, 0.47,0.12])
create_map(ax3)  ## C93735
ax3.plot(np.linspace(0,36,37), relay_correct6[0,:], label='same', color='#E5637B', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.3,  markersize=4.0, linestyle='-')
ax3.plot(np.linspace(0,36,37), relay_wrong6[0,:], label='opposite', color='#E5637B', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.4,  markersize=4.0, linestyle='--')
ax3.plot([6], [relay_correct6[0,6]+0.05], color='#E5637B', markerfacecolor='#E5637B', markeredgecolor='#E5637B', marker='v', markeredgewidth=1.0,  markersize=4.0)
ax3.plot([6], [0.05], color='#E5637B', markerfacecolor='#E5637B', markeredgecolor='#E5637B', marker='v', markeredgewidth=1.0,  markersize=4.0)
# ax3.plot([-1,37], [0,0], color='#E5637B', linewidth=0.6,  linestyle='-')
ax3.fill_between(x,y0, relay_correct6[0,:], where=(relay_correct6[0,:]>=y7), alpha=0.2, color='#E5637B')
# ax3.legend(loc=(0.48, 0.43), fontsize=8.5, frameon=False)
ax3.text(4.35, relay_correct6[0,6]-0.60, s='100%', fontsize=12.0, color='#E5637B')
ax3.text(-0.5, 1.45, s='b', fontsize=16.0, color='black', fontweight='bold')
ax3.text(31.5, 0.95, s='('+r"$A_6$"+")"r"$^1$", fontsize=13.0, color='black')


ax4 = plt.axes([0.52,0.6, 0.47,0.12])
create_map(ax4)
ax4.plot(np.linspace(0,36,37), relay_correct6[1,:], label='same', color='#D2AA3A', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.3,  markersize=4.0, linestyle='-')
ax4.plot(np.linspace(0,36,37), relay_wrong6[1,:], label='opposite', color='#D2AA3A', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.4,  markersize=4.0, linestyle='--')
ax4.plot([12], [relay_correct6[1,12]+0.05], color='#D2AA3A', markerfacecolor='#D2AA3A', markeredgecolor='#D2AA3A', marker='v', markeredgewidth=1.0,  markersize=4.0)
ax4.plot([12], [0.05], color='#D2AA3A', markerfacecolor='#D2AA3A', markeredgecolor='#D2AA3A', marker='v', markeredgewidth=1.0,  markersize=4.0)
ax4.fill_between(x,y0, relay_correct6[1,:], where=(relay_correct6[1,:]>=y7), alpha=0.2, color='#D2AA3A')
ax4.text(9.8, relay_correct6[1,12]-0.30, s='96.3%', fontsize=12.0, color='#D2AA3A')
ax4.text(31.5, 0.95, s='('+r"$A_6$"+")"r"$^2$", fontsize=13.0, color='black')


ax5 = plt.axes([0.52,0.48, 0.47,0.12])
create_map(ax5)
ax5.plot(np.linspace(0,36,37), relay_correct6[2,:], label='same', color='#009E74', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.3,  markersize=4.0, linestyle='-')
ax5.plot(np.linspace(0,36,37), relay_wrong6[2,:], label='opposite', color='#009E74', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.4,  markersize=4.0, linestyle='--')
ax5.plot([18], [relay_correct6[2,18]+0.05], color='#009E74', markerfacecolor='#009E74', markeredgecolor='#009E74', marker='v', markeredgewidth=1.0,  markersize=4.0)
ax5.plot([18], [0.05], color='#009E74', markerfacecolor='#009E74', markeredgecolor='#009E74', marker='v', markeredgewidth=1.0,  markersize=4.0)
ax5.fill_between(x,y0, relay_correct6[2,:], where=(relay_correct6[2,:]>=y7), alpha=0.17, color='#009E74')
ax5.text(15.8, relay_correct6[2,18]-0.30, s='93.7%', fontsize=12.0, color='#009E74')
ax5.text(31.5, 0.95, s='('+r"$A_6$"+")"r"$^3$", fontsize=13.0, color='black')


ax6 = plt.axes([0.52,0.36, 0.47,0.12])
create_map(ax6)
ax6.plot(np.linspace(0,36,37), relay_correct6[3,:], label='same', color='#57B4E9', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.3,  markersize=4.0, linestyle='-')
ax6.plot(np.linspace(0,36,37), relay_wrong6[3,:], label='opposite', color='#57B4E9', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.4,  markersize=4.0, linestyle='--')
ax6.plot([24], [relay_correct6[3,24]+0.05], color='#57B4E9', markerfacecolor='#57B4E9', markeredgecolor='#57B4E9', marker='v', markeredgewidth=1.0,  markersize=4.0)
ax6.plot([24], [0.05], color='#57B4E9', markerfacecolor='#57B4E9', markeredgecolor='#57B4E9', marker='v', markeredgewidth=1.0,  markersize=4.0)
ax6.fill_between(x,y0, relay_correct6[3,:], where=(relay_correct6[3,:]>=y7), alpha=0.22, color='#57B4E9')
ax6.text(21.8, relay_correct6[3,24]-0.30, s='87.2%', fontsize=12.0, color='#57B4E9')
ax6.text(31.5, 0.95, s='('+r"$A_6$"+")"r"$^4$", fontsize=13.0, color='black')


ax7 = plt.axes([0.52,0.24, 0.47,0.12])
create_map(ax7)
ax7.set_xticks([0,3,6,9,12,15,18,21,24,27,30,33,36]); ax7.set_xticklabels([r"$A_0$","",r"$A_6$","",r"$A_{12}$","",r"$A_{18}$","",r"$A_{24}$","",r"$A_{30}$","",r"$A_{36}$"], color='black')
ax7.plot(np.linspace(0,36,37), relay_correct6[4,:], label='same', color='#333A8C', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.3,  markersize=4.0, linestyle='-')
ax7.plot(np.linspace(0,36,37), relay_wrong6[4,:], label='opposite', color='#333A8C', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.4,  markersize=4.0, linestyle='--')
ax7.plot([30], [relay_correct6[4,30]+0.05], color='#333A8C', markerfacecolor='#333A8C', markeredgecolor='#333A8C', marker='v', markeredgewidth=1.0,  markersize=4.0)
ax7.plot([30], [0.05], color='#333A8C', markerfacecolor='#333A8C', markeredgecolor='#333A8C', marker='v', markeredgewidth=1.0,  markersize=4.0)
ax7.fill_between(x,y0, relay_correct6[4,:], where=(relay_correct6[4,:]>=y7), alpha=0.2, color='#333A8C')
ax7.text(27.8, relay_correct6[4,30]-0.30, s='68.2%', fontsize=12.0, color='#333A8C')
ax7.text(31.5, 0.95, s='('+r"$A_6$"+")"r"$^5$", fontsize=13.0, color='black')










def create_map(ax):
  # plt.axes( xscale="log" )
  plt.xlim(-0.5,37.0)
  plt.ylim(0.0,1.25)
  ax.tick_params(axis='both', length=6.0, width=1.0, labelsize=9.5)
  ax.spines['bottom'].set_linewidth(1.0)
  ax.spines['top'].set_linewidth(1.0)
  ax.spines['left'].set_linewidth(1.0)
  ax.spines['right'].set_linewidth(1.0)
  ax.set_xticks([0,3,6,9,12,15,18,21,24,27,30,33,36]); ax.set_xticklabels(["","","","","","","","","","","","",""], color='black')
  ax.set_yticks(np.linspace(0.0,1.0,5)); ax.set_yticklabels([0,"","50%","","100%"], color='black')
  ax.spines['left'].set_color('black')
  ax.tick_params(axis='y', right=False, length=7.0, width=1.0, labelsize=11.5, color='black')
  ax.tick_params(axis='x', right=False, length=7.0, width=1.0, labelsize=13.0, color='black')
  ax.set_xlabel('', size=10.5)
  ax.set_ylabel('', size=13.0)
  return ax




ax13 = plt.axes([0.01,0.72, 0.47,0.12])
create_map(ax13)
ax13.plot(np.linspace(0,36,37), relay_correct4[0,:], label='same', color='#E5637B', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.3,  markersize=4.0, linestyle='-')
ax13.plot(np.linspace(0,36,37), relay_wrong4[0,:], label='opposite', color='#E5637B', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.4,  markersize=4.0, linestyle='--')
ax13.plot([4], [relay_correct4[0,4]+0.05], color='#E5637B', markerfacecolor='#E5637B', markeredgecolor='#E5637B', marker='v', markeredgewidth=1.0,  markersize=4.0)
ax13.plot([4], [0.05], color='#E5637B', markerfacecolor='#E5637B', markeredgecolor='#E5637B', marker='v', markeredgewidth=1.0,  markersize=4.0)
# ax13.plot([-1,37], [0,0], color='#E5637B', linewidth=0.6,  linestyle='-')
ax13.fill_between(x,y0, relay_correct4[0,:], where=(relay_correct4[0,:]>=y7), alpha=0.2, color='#E5637B')
ax13.text(2.35, relay_correct4[0,4]-0.60, s='100%', fontsize=12.0, color='#E5637B')
ax13.text(-0.5, 1.45, s='a', fontsize=16.0, color='black', fontweight='bold')
ax13.text(31.5, 0.95, s='('+r"$A_4$"+")"r"$^1$", fontsize=13.0, color='black')
ax13.text(3.7, 1.85, s="Similarity   between   short-delayed   and   long-delayed   networks", fontsize=15.0, color='black')



ax14 = plt.axes([0.01,0.6, 0.47,0.12])
create_map(ax14)
ax14.plot(np.linspace(0,36,37), relay_correct4[1,:], label='same', color='#D2AA3A', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.3,  markersize=4.0, linestyle='-')
ax14.plot(np.linspace(0,36,37), relay_wrong4[1,:], label='opposite', color='#D2AA3A', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.4,  markersize=4.0, linestyle='--')
ax14.plot([8], [relay_correct4[1,8]+0.05], color='#D2AA3A', markerfacecolor='#D2AA3A', markeredgecolor='#D2AA3A', marker='v', markeredgewidth=1.0,  markersize=4.0)
ax14.plot([8], [0.05], color='#D2AA3A', markerfacecolor='#D2AA3A', markeredgecolor='#D2AA3A', marker='v', markeredgewidth=1.0,  markersize=4.0)
ax14.fill_between(x,y0, relay_correct4[1,:], where=(relay_correct4[1,:]>=y7), alpha=0.2, color='#D2AA3A')
ax14.text(5.8, relay_correct4[1,8]-0.30, s='83.4%', fontsize=12.0, color='#D2AA3A')
ax14.text(31.5, 0.95, s='('+r"$A_4$"+")"r"$^2$", fontsize=13.0, color='black')


ax15 = plt.axes([0.01,0.48, 0.47,0.12])
create_map(ax15)
ax15.plot(np.linspace(0,36,37), relay_correct4[2,:], label='same', color='#009E74', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.3,  markersize=4.0, linestyle='-')
ax15.plot(np.linspace(0,36,37), relay_wrong4[2,:], label='opposite', color='#009E74', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.4,  markersize=4.0, linestyle='--')
ax15.plot([12], [relay_correct4[2,12]+0.05], color='#009E74', markerfacecolor='#009E74', markeredgecolor='#009E74', marker='v', markeredgewidth=1.0,  markersize=4.0)
ax15.plot([12], [0.05], color='#009E74', markerfacecolor='#009E74', markeredgecolor='#009E74', marker='v', markeredgewidth=1.0,  markersize=4.0)
ax15.fill_between(x,y0, relay_correct4[2,:], where=(relay_correct4[2,:]>=y7), alpha=0.17, color='#009E74')
ax15.text(9.8, relay_correct4[2,12]-0.30, s='74.0%', fontsize=12.0, color='#009E74')
ax15.text(31.5, 0.95, s='('+r"$A_4$"+")"r"$^3$", fontsize=13.0, color='black')


ax16 = plt.axes([0.01,0.36, 0.47,0.12])
create_map(ax16)
ax16.plot(np.linspace(0,36,37), relay_correct4[3,:], label='same', color='#57B4E9', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.3,  markersize=4.0, linestyle='-')
ax16.plot(np.linspace(0,36,37), relay_wrong4[3,:], label='opposite', color='#57B4E9', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.4,  markersize=4.0, linestyle='--')
ax16.plot([16], [relay_correct4[3,16]+0.05], color='#57B4E9', markerfacecolor='#57B4E9', markeredgecolor='#57B4E9', marker='v', markeredgewidth=1.0,  markersize=4.0)
ax16.plot([16], [0.05], color='#57B4E9', markerfacecolor='#57B4E9', markeredgecolor='#57B4E9', marker='v', markeredgewidth=1.0,  markersize=4.0)
ax16.fill_between(x,y0, relay_correct4[3,:], where=(relay_correct4[3,:]>=y7), alpha=0.22, color='#57B4E9')
ax16.text(13.8, relay_correct4[3,16]-0.30, s='65.0%', fontsize=12.0, color='#57B4E9')
ax16.text(31.5, 0.95, s='('+r"$A_4$"+")"r"$^4$", fontsize=13.0, color='black')


ax17 = plt.axes([0.01,0.24, 0.47,0.12])
create_map(ax17)
ax17.set_xticks([0,3,6,9,12,15,18,21,24,27,30,33,36]); ax17.set_xticklabels([r"$A_0$","",r"$A_6$","",r"$A_{12}$","",r"$A_{18}$","",r"$A_{24}$","",r"$A_{30}$","",r"$A_{36}$"], color='black')
ax17.plot(np.linspace(0,36,37), relay_correct4[4,:], label='same', color='#333A8C', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.3,  markersize=4.0, linestyle='-')
ax17.plot(np.linspace(0,36,37), relay_wrong4[4,:], label='opposite', color='#333A8C', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.4,  markersize=4.0, linestyle='--')
ax17.plot([20], [relay_correct4[4,20]+0.05], color='#333A8C', markerfacecolor='#333A8C', markeredgecolor='#333A8C', marker='v', markeredgewidth=1.0,  markersize=4.0)
ax17.plot([20], [0.05], color='#333A8C', markerfacecolor='#333A8C', markeredgecolor='#333A8C', marker='v', markeredgewidth=1.0,  markersize=4.0)
ax17.fill_between(x,y0, relay_correct4[4,:], where=(relay_correct4[4,:]>=y7), alpha=0.2, color='#333A8C')
ax17.text(17.6, relay_correct4[4,20]-0.30, s='70.4%', fontsize=12.0, color='#333A8C')
ax17.text(31.5, 0.95, s='('+r"$A_4$"+")"r"$^5$", fontsize=13.0, color='black')









fig.savefig('3_extended_data_fig/FigureS4_relay_transport_ERSSTv5.png', bbox_inches = 'tight')
plt.show()


