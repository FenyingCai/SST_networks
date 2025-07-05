# %%
import numpy as np
import xarray as xr



X = np.zeros([13,2], dtype=float)  ## linear fit
X[:,0] = 1
X[:,1] = np.linspace(1,13,13)
x_mean = 6.0
x1 = x_mean - 6
x2 = x_mean + 18
x3 = x_mean + 30



##--##--##--##--##--##--   Link  number  (CPC Tmax)   --##--##--##--##--##--##
ds1 = xr.open_dataset("1_networks/part11_HadISST_CPCTmax/Link2x2_CPCTmax_number_lag0_36.nc")
link_number_CPCTmax = np.log10(ds1.link_number.data)

beta, resids, rank, s = np.linalg.lstsq(X, link_number_CPCTmax[0:13],  )  ## linear fit
print(" reg (CPC Tmax,  12 months) = ", beta)

y_mean_CPCTmax = np.nanmean(link_number_CPCTmax[0:13])
reg_xy_CPCTmax = beta[1]  ## log10(y) = ax + b,   y = 10**(ax+b)

y1_CPCTmax = y_mean_CPCTmax - 6*reg_xy_CPCTmax
y2_CPCTmax = y_mean_CPCTmax + 18*reg_xy_CPCTmax
y3_CPCTmax = y_mean_CPCTmax + (18+12)*reg_xy_CPCTmax



ds2 = xr.open_dataset("2_analysis/fig4_sign2x2/networks012_CPCTmax_sign_correct_wrong.nc")
relay_CPCTmax_correct = np.log10(ds2.link_number_relay_correct.data)
relay_CPCTmax_notsame = np.log10(ds2.link_number_relay_notsame.data)

beta, resids, rank, s = np.linalg.lstsq(X, relay_CPCTmax_correct[0:13],  )  ## linear fit
print(" reg (CPC Tmax,  12 months) = ", beta)

y_mean_CPCTmax2 = np.nanmean(relay_CPCTmax_correct[0:13])
reg_xy_CPCTmax2 = beta[1]  ## log10(y) = ax + b,   y = 10**(ax+b)

y1_CPCTmax2 = y_mean_CPCTmax2 - 6*reg_xy_CPCTmax2
y2_CPCTmax2 = y_mean_CPCTmax2 + 18*reg_xy_CPCTmax2
y3_CPCTmax2 = y_mean_CPCTmax2 + (18+12)*reg_xy_CPCTmax2






##--##--##--##--##--##--   Link  number  (CPC Rain)   --##--##--##--##--##--##
ds3 = xr.open_dataset("1_networks/part15_HadISST_CPCRain90_2x2/Link2x2_CPCRain90_number_lag0_36.nc")
link_number_CPCRain = np.log10(ds3.link_number.data)

beta, resids, rank, s = np.linalg.lstsq(X, link_number_CPCRain[0:13],  )  ## linear fit
print(" reg (CPC Rain,  12 months) = ", beta)

y_mean_CPCRain = np.nanmean(link_number_CPCRain[0:13])
reg_xy_CPCRain = beta[1]  ## log10(y) = ax + b,   y = 10**(ax+b)

y1_CPCRain = y_mean_CPCRain - 6*reg_xy_CPCRain
y2_CPCRain = y_mean_CPCRain + 18*reg_xy_CPCRain
y3_CPCRain = y_mean_CPCRain + (18+12)*reg_xy_CPCRain



ds4 = xr.open_dataset("2_analysis/fig4_sign2x2/networks012_CPCRain90_sign_correct_wrong.nc")
relay_CPCRain_correct = np.log10(ds4.link_number_relay_correct.data)
relay_CPCRain_notsame = np.log10(ds4.link_number_relay_notsame.data)

beta, resids, rank, s = np.linalg.lstsq(X, relay_CPCRain_correct[0:13],  )  ## linear fit
print(" reg (CPC Rain,  12 months) = ", beta)

y_mean_CPCRain2 = np.nanmean(relay_CPCRain_correct[0:13])
reg_xy_CPCRain2 = beta[1]  ## log10(y) = ax + b,   y = 10**(ax+b)

y1_CPCRain2 = y_mean_CPCRain2 - 6*reg_xy_CPCRain2
y2_CPCRain2 = y_mean_CPCRain2 + 18*reg_xy_CPCRain2
y3_CPCRain2 = y_mean_CPCRain2 + (18+12)*reg_xy_CPCRain2









##--##--##--##--##--##--   HadISST  link  number  VS  Mestimated  (CPC Rain)   --##--##--##--##--##--##
ds2 = xr.open_dataset("2_analysis/fig4_relay/Mestimated6_CPCRain90_sign_correct_wrong.nc")
link_number_all6_Rain = ds2.link_number_all.data
relay_correct6_Rain = ds2.link_number_relay_correct.data
relay_wrong6_Rain = ds2.link_number_relay_wrong.data

for mfold in range(6): 
  relay_correct6_Rain[mfold,:] = relay_correct6_Rain[mfold,:]/link_number_all6_Rain
  relay_wrong6_Rain[mfold,:] = relay_wrong6_Rain[mfold,:]/link_number_all6_Rain

relay_correct6_Rain_moving3 = np.zeros_like(relay_correct6_Rain)
for imon in range(1,36): 
  relay_correct6_Rain_moving3[:,imon] = np.nanmean(relay_correct6_Rain[:,imon-1:imon+2], axis=1)
relay_correct6_Rain_moving3 = relay_correct6_Rain

print(" Rain  1 ",relay_correct6_Rain[0,6]," 2 ",relay_correct6_Rain[1,12]," 3 ",relay_correct6_Rain[2,18]," 4 ",relay_correct6_Rain[3,24]," 5 ",relay_correct6_Rain[4,30]," 6 ",relay_correct6_Rain[5,36])



##--##--##--##--##--##--   HadISST  link  number  VS  Mestimated  (CPC Tmax)   --##--##--##--##--##--##
ds2 = xr.open_dataset("2_analysis/fig4_relay/Mestimated6_CPCTmax_sign_correct_wrong.nc")
link_number_all6_Tmax = ds2.link_number_all.data
relay_correct6_Tmax = ds2.link_number_relay_correct.data
relay_wrong6_Tmax = ds2.link_number_relay_wrong.data

for mfold in range(6): 
  relay_correct6_Tmax[mfold,:] = relay_correct6_Tmax[mfold,:]/link_number_all6_Tmax
  relay_wrong6_Tmax[mfold,:] = relay_wrong6_Tmax[mfold,:]/link_number_all6_Tmax

print(" Tmax  1 ",relay_correct6_Tmax[0,6]," 2 ",relay_correct6_Tmax[1,12]," 3 ",relay_correct6_Tmax[2,18]," 4 ",relay_correct6_Tmax[3,24]," 5 ",relay_correct6_Tmax[4,30]," 6 ",relay_correct6_Tmax[5,36])
















# %%
## Plotting Import
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('agg')

## Define a Map Function for plotting
def create_map(ax):
  # plt.axes( xscale="log" )
  plt.xlim(-1.0,37.0)
  plt.ylim(2.0,5.5)
  ax.tick_params(axis='both', length=6.0, width=1.0, labelsize=9.0)
  ax.spines['bottom'].set_linewidth(1.0)
  ax.spines['top'].set_linewidth(1.0)
  ax.spines['left'].set_linewidth(1.0)
  ax.spines['right'].set_linewidth(1.0)
  ax.spines['left'].set_color('black')
  ax.tick_params(axis='y', right=False, length=6.0, width=1.0, labelsize=9.4, color='black')
  return ax


##--##--##--  create figure  --##--##--##
width_in_inches = 210 / 25.4  ## A4 page
height_in_inches = 297 / 25.4

fig = plt.figure(dpi=400,figsize=(width_in_inches,height_in_inches))







##------------------------------------------------------------------------------------------------------------------------------##
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
  ax.set_yticks(np.linspace(0.0,1.0,5)); ax.set_yticklabels([0,"","50%","","100%"], color='black')
  ax.spines['left'].set_color('black')
  ax.tick_params(axis='y', right=False, length=7.0, width=1.0, labelsize=9.5, color='black')
  ax.tick_params(axis='x', right=False, length=7.0, width=1.0, labelsize=11.0, color='black')
  ax.set_xlabel('', size=10.5)
  ax.set_ylabel('', size=13.0)
  return ax

  


x = np.linspace(0,36,37)
y0 = np.linspace(0.0,0.0,37)
y7 = np.linspace(0.75,0.75,37)

ax11 = plt.axes([0.1,0.4, 0.3,0.09])
create_map(ax11)  ## C93735
ax11.plot(np.linspace(0,36,37), relay_correct6_Tmax[0,:], label='same', color='#E5637B', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.3,  markersize=4.0, linestyle='-')
ax11.plot(np.linspace(0,36,37), relay_wrong6_Tmax[0,:], label='opposite', color='#E5637B', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.4,  markersize=4.0, linestyle='--')
ax11.plot([6], [relay_correct6_Tmax[0,6]+0.05], color='#E5637B', markerfacecolor='#E5637B', markeredgecolor='#E5637B', marker='v', markeredgewidth=1.0,  markersize=4.0)
ax11.plot([6], [0.05], color='#E5637B', markerfacecolor='#E5637B', markeredgecolor='#E5637B', marker='v', markeredgewidth=1.0,  markersize=4.0)
# ax11.plot([-1,37], [0,0], color='#E5637B', linewidth=0.6,  linestyle='-')
ax11.fill_between(x,y0, relay_correct6_Tmax[0,:], where=(relay_correct6_Tmax[0,:]>=y7), alpha=0.2, color='#E5637B')
# ax11.legend(loc=(0.48, 0.43), fontsize=8.5, frameon=False)
ax11.text(4.35, relay_correct6_Tmax[0,6]-0.75, s='100%', fontsize=10.5, color='#E5637B')
ax11.text(-0.5, 1.55, s='e', fontsize=17.5, color='black', fontweight='bold')
ax11.text(30.2, 0.93, s='('+r"$B_6$"+")", fontsize=11.0, color='black')
plt.title("SST   &   Tmax", size=11.5)


ax12 = plt.axes([0.1,0.31, 0.3,0.09])
create_map(ax12)
ax12.plot(np.linspace(0,36,37), relay_correct6_Tmax[1,:], label='same', color='#D2AA3A', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.3,  markersize=4.0, linestyle='-')
ax12.plot(np.linspace(0,36,37), relay_wrong6_Tmax[1,:], label='opposite', color='#D2AA3A', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.4,  markersize=4.0, linestyle='--')
ax12.plot([12], [relay_correct6_Tmax[1,12]+0.05], color='#D2AA3A', markerfacecolor='#D2AA3A', markeredgecolor='#D2AA3A', marker='v', markeredgewidth=1.0,  markersize=4.0)
ax12.plot([12], [0.05], color='#D2AA3A', markerfacecolor='#D2AA3A', markeredgecolor='#D2AA3A', marker='v', markeredgewidth=1.0,  markersize=4.0)
ax12.fill_between(x,y0, relay_correct6_Tmax[1,:], where=(relay_correct6_Tmax[1,:]>=y7), alpha=0.2, color='#D2AA3A')
ax12.text(10.0, relay_correct6_Tmax[1,12]-0.27, s='81.2%', fontsize=10.5, color='#D2AA3A')
ax12.text(24.5, 0.93, s='('+r"$A_6$"+")"r"$^1$"+'*'+'('+r"$B_6$"+")", fontsize=11.0, color='black')


ax13 = plt.axes([0.1,0.22, 0.3,0.09])
create_map(ax13)
ax13.set_xticks([0,3,6,9,12,15,18,21,24,27,30,33,36]); ax13.set_xticklabels([r"$B_0$","",r"$B_6$","",r"$B_{12}$","",r"$B_{18}$","",r"$B_{24}$","",r"$B_{30}$","",r"$B_{36}$"], color='black')
ax13.plot(np.linspace(0,36,37), relay_correct6_Tmax[2,:], label='same', color='#009E74', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.3,  markersize=4.0, linestyle='-')
ax13.plot(np.linspace(0,36,37), relay_wrong6_Tmax[2,:], label='opposite', color='#009E74', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.4,  markersize=4.0, linestyle='--')
ax13.plot([18], [relay_correct6_Tmax[2,18]+0.05], color='#009E74', markerfacecolor='#009E74', markeredgecolor='#009E74', marker='v', markeredgewidth=1.0,  markersize=4.0)
ax13.plot([18], [0.05], color='#009E74', markerfacecolor='#009E74', markeredgecolor='#009E74', marker='v', markeredgewidth=1.0,  markersize=4.0)
ax13.fill_between(x,y0, relay_correct6_Tmax[2,:], where=(relay_correct6_Tmax[2,:]>=y7), alpha=0.17, color='#009E74')
ax13.text(16.0, relay_correct6_Tmax[2,18]-0.27, s='87.2%', fontsize=10.5, color='#009E74')
ax13.text(24.5, 0.93, s='('+r"$A_6$"+")"r"$^2$"+'*'+'('+r"$B_6$"+")", fontsize=11.0, color='black')




y7 = np.linspace(0.60,0.60,37)

ax21 = plt.axes([0.5,0.4, 0.3,0.09])
create_map(ax21)  ## C93735
ax21.plot(np.linspace(0,36,37), relay_correct6_Rain[0,:], label='same', color='#E5637B', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.3,  markersize=4.0, linestyle='-')
ax21.plot(np.linspace(0,36,37), relay_wrong6_Rain[0,:], label='opposite', color='#E5637B', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.4,  markersize=4.0, linestyle='--')
ax21.plot([6], [relay_correct6_Rain[0,6]+0.05], color='#E5637B', markerfacecolor='#E5637B', markeredgecolor='#E5637B', marker='v', markeredgewidth=1.0,  markersize=4.0)
ax21.plot([6], [0.05], color='#E5637B', markerfacecolor='#E5637B', markeredgecolor='#E5637B', marker='v', markeredgewidth=1.0,  markersize=4.0)
# ax21.plot([-1,37], [0,0], color='#E5637B', linewidth=0.6,  linestyle='-')
ax21.fill_between(x,y0, relay_correct6_Rain[0,:], where=(relay_correct6_Rain[0,:]>=y7), alpha=0.2, color='#E5637B')
# ax21.legend(loc=(0.48, 0.43), fontsize=8.5, frameon=False)
ax21.text(4.32, relay_correct6_Rain[0,6]-0.85, s='100%', fontsize=10.5, color='#E5637B')
ax21.text(-0.5, 1.55, s='f', fontsize=17.5, color='black', fontweight='bold')
ax21.text(30.2, 0.93, s='('+r"$B_6$"+")", fontsize=11.0, color='black')
plt.title("SST   &   Precip", size=11.5)


ax22 = plt.axes([0.5,0.31, 0.3,0.09])
create_map(ax22)
ax22.plot(np.linspace(0,36,37), relay_correct6_Rain[1,:], label='same', color='#D2AA3A', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.3,  markersize=4.0, linestyle='-')
ax22.plot(np.linspace(0,36,37), relay_wrong6_Rain[1,:], label='opposite', color='#D2AA3A', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.4,  markersize=4.0, linestyle='--')
ax22.plot([12], [relay_correct6_Rain[1,12]+0.05], color='#D2AA3A', markerfacecolor='#D2AA3A', markeredgecolor='#D2AA3A', marker='v', markeredgewidth=1.0,  markersize=4.0)
ax22.plot([12], [0.05], color='#D2AA3A', markerfacecolor='#D2AA3A', markeredgecolor='#D2AA3A', marker='v', markeredgewidth=1.0,  markersize=4.0)
ax22.fill_between(x,y0, relay_correct6_Rain[1,:], where=(relay_correct6_Rain[1,:]>=y7), alpha=0.2, color='#D2AA3A')
ax22.text(10.0, relay_correct6_Rain[1,12]-0.27, s='67.6%', fontsize=10.5, color='#D2AA3A')
ax22.text(24.5, 0.93, s='('+r"$A_6$"+")"r"$^1$"+'*'+'('+r"$B_6$"+")", fontsize=11.0, color='black')


ax23 = plt.axes([0.5,0.22, 0.3,0.09])
create_map(ax23)
ax23.set_xticks([0,3,6,9,12,15,18,21,24,27,30,33,36]); ax23.set_xticklabels([r"$B_0$","",r"$B_6$","",r"$B_{12}$","",r"$B_{18}$","",r"$B_{24}$","",r"$B_{30}$","",r"$B_{36}$"], color='black')
ax23.plot(np.linspace(0,36,37), relay_correct6_Rain[2,:], label='same', color='#009E74', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.3,  markersize=4.0, linestyle='-')
ax23.plot(np.linspace(0,36,37), relay_wrong6_Rain[2,:], label='opposite', color='#009E74', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.4,  markersize=4.0, linestyle='--')
ax23.plot([18], [relay_correct6_Rain[2,18]+0.05], color='#009E74', markerfacecolor='#009E74', markeredgecolor='#009E74', marker='v', markeredgewidth=1.0,  markersize=4.0)
ax23.plot([18], [0.05], color='#009E74', markerfacecolor='#009E74', markeredgecolor='#009E74', marker='v', markeredgewidth=1.0,  markersize=4.0)
ax23.fill_between(x,y0, relay_correct6_Rain[2,:], where=(relay_correct6_Rain[2,:]>=y7), alpha=0.17, color='#009E74')
ax23.text(16.0, relay_correct6_Rain[2,18]-0.27, s='79.6%', fontsize=10.5, color='#009E74')
ax23.text(24.5, 0.93, s='('+r"$A_6$"+")"r"$^2$"+'*'+'('+r"$B_6$"+")", fontsize=11.0, color='black')






fig.savefig('2_analysis/Figure4ef.png', bbox_inches = 'tight')
plt.show()


