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
ds3 = xr.open_dataset("1_networks/part12_HadISST_CPCRain90/Link2x2_CPCRain90_number_lag0_36.nc")
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
  ax.tick_params(axis='both', length=6.0, width=1.0, labelsize=8.5)
  ax.spines['bottom'].set_linewidth(1.0)
  ax.spines['top'].set_linewidth(1.0)
  ax.spines['left'].set_linewidth(1.0)
  ax.spines['right'].set_linewidth(1.0)
  ax.spines['left'].set_color('black')
  ax.tick_params(axis='y', right=False, length=6.0, width=1.0, labelsize=8.6, color='black')
  return ax


##--##--##--  create figure  --##--##--##
width_in_inches = 210 / 25.4  ## A4 page
height_in_inches = 297 / 25.4

fig = plt.figure(dpi=400,figsize=(width_in_inches,height_in_inches))



ax1 = plt.axes([0.02,0.6, 0.25,0.15])
create_map(ax1)
ax1.set_xlabel('Lagged   months', size=10)
ax1.set_ylabel('Link   number', size=10)
plt.rcParams['text.usetex'] = False
plt.ylim(3.5,6.0)
ax1.set_yticks([3.5,4.0,4.5,5.0,5.5,6.0]); ax1.set_yticklabels(["",r'$10^4$',"",r'$10^5$',"",r'$10^6$'], color='black')
ax1.set_xticks([0,3,6,9,12,15,18,21,24,27,30,33,36]); ax1.set_xticklabels([0,"",6,"",12,"",18,"",24,"",30,"",36], color='black')
ax1.plot(np.linspace(0,12,13), link_number_CPCTmax[0:13], label='0-12  months', markerfacecolor='#DB9BA1', markeredgecolor='#B83945', marker='o', markeredgewidth=0.60, linewidth=0.0,  markersize=3.8, linestyle='-')#, alpha=0.5)
ax1.plot(np.linspace(13,36,24), link_number_CPCTmax[13:37], label='13-36  months', markerfacecolor='#9AB9C0', markeredgecolor='#377483', marker='*', markeredgewidth=0.55, linewidth=0.0,  markersize=4.7, linestyle='-')#, alpha=0.5)
ax1.plot([x1, x_mean, x2, x3], [y1_CPCTmax, y_mean_CPCTmax, y2_CPCTmax, y3_CPCTmax], label='linear  fit  (0-12 month)', color="#e89a10", linewidth=0.6,  linestyle='-', alpha=0.6)
ax1.legend(loc=(0.04, 0.07), fontsize=8.5, frameon=False)
plt.title("SST   &   Tmax", size=10.5)
ax1.text(-1.0, 6.3, s='a', fontsize=14.5, color='black', fontweight='bold')


ax2 = plt.axes([0.35,0.6, 0.25,0.15])
create_map(ax2)
ax2.set_xlabel('Lagged   months', size=10)
ax2.set_ylabel('', size=10)
plt.rcParams['text.usetex'] = False
plt.ylim(4.5,5.5)
ax2.set_yticks([4.5,4.75,5.0,5.25,5.5]); ax2.set_yticklabels([r'$10^{4.5}$',"",r'$10^5$',"",r'$10^{5.5}$'], color='black')
ax2.set_xticks([0,3,6,9,12,15,18,21,24,27,30,33,36]); ax2.set_xticklabels([0,"",6,"",12,"",18,"",24,"",30,"",36], color='black')
ax2.plot(np.linspace(0,12,13), link_number_CPCRain[0:13], label='0-12  months', markerfacecolor='#DB9BA1', markeredgecolor='#B83945', marker='o', markeredgewidth=0.60, linewidth=0.0,  markersize=3.8, linestyle='-')#, alpha=0.5)
ax2.plot(np.linspace(13,36,24), link_number_CPCRain[13:37], label='13-36  months', markerfacecolor='#9AB9C0', markeredgecolor='#377483', marker='*', markeredgewidth=0.55, linewidth=0.0,  markersize=4.7, linestyle='-')#, alpha=0.5)
ax2.plot([x1, x_mean, x2, x3], [y1_CPCRain, y_mean_CPCRain, y2_CPCRain, y3_CPCRain], label='linear  fit  (0-12 month)', color="#e89a10", linewidth=0.6,  linestyle='-', alpha=0.6)
# ax2.legend(loc=(0.04, 0.07), fontsize=8.5, frameon=False)
plt.title("SST   &   Precip", size=10.5)
ax1.text(49.0, 6.3, s='b', fontsize=14.5, color='black', fontweight='bold')





ax3 = plt.axes([0.02,0.37, 0.25,0.15])
create_map(ax3)
ax3.set_xlabel('Lagged   months', size=10)
ax3.set_ylabel('Link   number', size=10)
plt.rcParams['text.usetex'] = False
plt.ylim(3.5,6.0)
ax3.set_yticks([3.5,4.0,4.5,5.0,5.5,6.0]); ax3.set_yticklabels(["",r'$10^4$',"",r'$10^5$',"",r'$10^6$'], color='black')
ax3.set_xticks([0,3,6,9,12,15,18,21,24,27,30,33,36]); ax3.set_xticklabels([0,"",6,"",12,"",18,"",24,"",30,"",36], color='black')
ax3.plot(np.linspace(0,36,37), relay_CPCTmax_correct[:], label='same  signs', markerfacecolor='#DB9BA1', markeredgecolor='#B83945', marker='o', markeredgewidth=0.60, linewidth=0.0,  markersize=3.8, linestyle='-')#, alpha=0.5)
ax3.plot(np.linspace(0,36,37), relay_CPCTmax_notsame[:], label='different  signs', markerfacecolor='#9AB9C0', markeredgecolor='#377483', marker='*', markeredgewidth=0.55, linewidth=0.0,  markersize=4.7, linestyle='-')#, alpha=0.5)
ax3.plot([x1, x_mean, x2, x3], [y1_CPCTmax2, y_mean_CPCTmax2, y2_CPCTmax2, y3_CPCTmax2], label='', color="#e89a10", linewidth=0.6,  linestyle='-', alpha=0.6)
ax3.legend(loc=(0.04, 0.07), fontsize=8.5, frameon=False)
plt.title("SST   &   Tmax", size=10.5)
ax3.text(-1.0, 6.3, s='c', fontsize=14.5, color='black', fontweight='bold')




ax4 = plt.axes([0.35,0.37, 0.25,0.15])
create_map(ax4)
ax4.set_xlabel('Lagged   months', size=10)
ax4.set_ylabel('', size=10)
plt.rcParams['text.usetex'] = False
plt.ylim(4.25,5.0)
ax4.set_yticks([4.25,4.5,4.75,5.0,5.25,5.5]); ax4.set_yticklabels(["",r'$10^{4.5}$',"",r'$10^5$',"",r'$10^{5.5}$'], color='black')
ax4.set_xticks([0,3,6,9,12,15,18,21,24,27,30,33,36]); ax4.set_xticklabels([0,"",6,"",12,"",18,"",24,"",30,"",36], color='black')
ax4.plot(np.linspace(0,36,37), relay_CPCRain_correct[:], label='same  signs', markerfacecolor='#DB9BA1', markeredgecolor='#B83945', marker='o', markeredgewidth=0.60, linewidth=0.0,  markersize=3.8, linestyle='-')#, alpha=0.5)
ax4.plot(np.linspace(0,36,37), relay_CPCRain_notsame[:], label='different  signs', markerfacecolor='#9AB9C0', markeredgecolor='#377483', marker='*', markeredgewidth=0.55, linewidth=0.0,  markersize=4.7, linestyle='-')#, alpha=0.5)
ax4.plot([x1, x_mean, x2, x3], [y1_CPCRain2, y_mean_CPCRain2, y2_CPCRain2, y3_CPCRain2], label='', color="#e89a10", linewidth=0.6,  linestyle='-', alpha=0.6)
# ax4.legend(loc=(0.04, 0.07), fontsize=8.5, frameon=False)
plt.title("SST   &   Precip", size=10.5)
ax3.text(49.0, 6.3, s='d', fontsize=14.5, color='black', fontweight='bold')







fig.savefig('2_analysis/Figure4abcd.png', bbox_inches = 'tight')
plt.show()


