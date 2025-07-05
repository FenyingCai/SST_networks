# %%
import numpy as np
import xarray as xr




##--##--##--##--##--##--   HadISST  link  number  (0-36 months)   --##--##--##--##--##--##
ds1 = xr.open_dataset("1_networks/part1_HadISST/Link2x2_number_lag0_36.nc")
link_number = np.log10(ds1.link_number.data)
print(" max = ",np.nanmax(link_number))


X = np.zeros([13,2], dtype=float)  ## linear fit
X[:,0] = 1
X[:,1] = np.linspace(1,13,13)
beta, resids, rank, s = np.linalg.lstsq(X, link_number[0:13],  )  ## linear fit
print(" reg (HadISST,  12 months) = ", beta)

x_mean = 6.0
y_mean = np.nanmean(link_number[0:13])
reg_xy = beta[1]  ## log10(y) = ax + b,   y = 10**(ax+b)

x1 = x_mean - 6
x2 = x_mean + 18
x3 = x_mean + 30
y1 = y_mean - 6*reg_xy
y2 = y_mean + 18*reg_xy
y3 = y_mean + (18+12)*reg_xy

print(" decreasing ratio (1 month) = ", 10**reg_xy)
print(" decreasing ratio (6 month) = ", 10**(reg_xy*6))
print(" decreasing ratio (12 month) = ", 10**(reg_xy*12))
print(" y0 (HadISST) = ", y1,"  ",10**y1)






##--##--##--##--##--##--   Estimated  VS  Real  (13-36 months)   --##--##--##--##--##--##
estimate_monthly_link = np.zeros([37],float)
for imonth in range(37): 
  estimate_monthly_link[imonth] = y_mean - 6*reg_xy + (imonth)*reg_xy   ## imonth=0, lag=0

estimate_year2 = np.zeros(12,float)
estimate_year2_lin = np.zeros(12,float)
for imonth in range(12): 
  estimate_year2[imonth] = y_mean + 7*reg_xy + (imonth)*reg_xy   ## imonth=0, lag=13
  estimate_year2_lin[imonth] = 10**estimate_year2[imonth]

estimate_year3 = np.zeros(12,float)
estimate_year3_lin = np.zeros(12,float)
for imonth in range(12): 
  estimate_year3[imonth] = y_mean + 19*reg_xy + (imonth)*reg_xy   ## imonth=0, lag=25
  estimate_year3_lin[imonth] = 10**estimate_year3[imonth]

estimate_year2_link = (np.nansum(estimate_year2_lin))
estimate_year3_link = (np.nansum(estimate_year3_lin))
estimate_year23_link = (np.nansum(estimate_year2_lin)+np.nansum(estimate_year3_lin))

relay_notsame_1336 = (np.nansum(ds1.link_number[13:36+1].data, axis=0))
relay_notsame_1324 = (np.nansum(ds1.link_number[13:24+1].data, axis=0))
relay_notsame_2536 = (np.nansum(ds1.link_number[25:36+1].data, axis=0))
print(" 13-36 month number = ", relay_notsame_1336,"  ",estimate_year23_link,"  ",relay_notsame_1336/estimate_year23_link)
print(" 13-36 month number = ", np.log10(relay_notsame_1336),"  ",np.log10(estimate_year23_link))
print(" 13-24 month number = ", relay_notsame_1324,"  ",estimate_year2_link,"  ",relay_notsame_1324/estimate_year2_link)
print(" 13-24 month number = ", np.log10(relay_notsame_1324),"  ",np.log10(estimate_year2_link))
print(" 25-36 month number = ", relay_notsame_2536,"  ",estimate_year3_link,"  ",relay_notsame_2536/estimate_year3_link)
print(" 25-36 month number = ", np.log10(relay_notsame_2536),"  ",np.log10(estimate_year3_link))









##--##--##--##--##--##--   HadISST  same-sign  VS  different-sign  (0-12 months)   --##--##--##--##--##--##
ds2 = xr.open_dataset("2_analysis/fig1_sign/networks012_2x2_sign_correct_wrong.nc")
relay_correct0 = np.log10(ds2.link_number_relay_correct.data)
relay_wrong0 = np.log10(ds2.link_number_relay_wrong.data)
relay_notsame0 = np.log10(ds2.link_number_relay_notsame.data)

XX = np.zeros([13,2], dtype=float)  ## linear fit
XX[:,0] = 1
XX[:,1] = np.linspace(1,13,13)
beta, resids, rank, s = np.linalg.lstsq(XX, relay_correct0[0:13],  )  ## linear fit
print(" reg (HadISST correct,  12 months) = ", beta)

xx_mean = 6.0
yy_mean = np.nanmean(relay_correct0[0:13])
reg_xxyy = beta[1]  ## log10(yy) = axx + b,   yy = 10**(axx+b)

xx1 = xx_mean - 6
xx2 = xx_mean + 18
xx3 = xx_mean + 30
yy1 = yy_mean - 6*reg_xxyy
yy2 = yy_mean + 18*reg_xxyy
yy3 = yy_mean + (18+12)*reg_xxyy

print(" decreasing ratio (1 month, correct) = ", 10**reg_xxyy)
print(" decreasing ratio (6 month, correct) = ", 10**(reg_xxyy*6))
print(" decreasing ratio (12 month, correct) = ", 10**(reg_xxyy*12))
print(" yy0 (HadISST, correct) = ", yy1,"  ",10**yy1)












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
  ax.tick_params(axis='both', length=6.0, width=1.0, labelsize=9.5)
  ax.spines['bottom'].set_linewidth(0.9)
  ax.spines['top'].set_linewidth(0.9)
  ax.spines['left'].set_linewidth(0.9)
  ax.spines['right'].set_linewidth(0.9)
  ax.spines['left'].set_color('black')
  ax.tick_params(axis='y', right=False, length=6.0, width=1.0, labelsize=9.8, color='black')
  return ax


##--##--##--  create figure  --##--##--##
width_in_inches = 210 / 25.4  ## A4 page
height_in_inches = 297 / 25.4

fig = plt.figure(dpi=400,figsize=(width_in_inches,height_in_inches))



ax1 = plt.axes([0.02,0.6, 0.443,0.20])
create_map(ax1)
ax1.set_xlabel('Lagged   months', size=10)
ax1.set_ylabel('Link   number', size=10)
plt.rcParams['text.usetex'] = False
plt.ylim(3.5,7.0)
ax1.set_yticks([3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0]); ax1.set_yticklabels(["",r'$10^4$',"",r'$10^5$',"",r'$10^6$',"",r'$10^7$'], color='black')
ax1.set_xticks([0,3,6,9,12,15,18,21,24,27,30,33,36]); ax1.set_xticklabels([0,"",6,"",12,"",18,"",24,"",30,"",36], color='black')
ax1.plot(np.linspace(0,12,13), link_number[0:13], label='0-12  months', markerfacecolor='#DB9BA1', markeredgecolor='#B83945', marker='o', markeredgewidth=0.60, linewidth=0.0,  markersize=5.5, linestyle='-')#, alpha=0.5)
ax1.plot(np.linspace(13,36,24), link_number[13:37], label='13-36  months', markerfacecolor='#9AB9C0', markeredgecolor='#377483', marker='*', markeredgewidth=0.55, linewidth=0.0,  markersize=7.0, linestyle='-')#, alpha=0.5)
ax1.plot([x1, x_mean, x2, x3], [y1, y_mean, y2, y3], label='linear  fit  (0-12 month)', color="#e89a10", linewidth=0.6,  linestyle='-', alpha=0.6)

ax1.legend(loc=(0.04, 0.07), fontsize=9.5, frameon=False)
plt.title("Observed   links", size=11.5)
ax1.text(1.8, 5.45, s='decrease  18%', fontsize=9.5, color='#e89a10')
ax1.text(2.25, 5.19, s='each  month', fontsize=9.5, color='#e89a10')

# ax1.text(20.0, 4.5, s=r'$y=10^{6.90-0.0875x}$', fontsize=10.0, color='#e89a10',bbox=dict(facecolor='#e89a10',alpha=0.1,edgecolor='#e89a10'))
ax1.text(-1.2, 7.50, s='a', fontsize=14.5, color='black', fontweight='bold')






ax2 = plt.axes([0.54,0.6, 0.443,0.20])
create_map(ax2)
ax2.set_xlabel('Lagged   months', size=10)
ax2.set_ylabel('', size=9)
plt.rcParams['text.usetex'] = False
plt.ylim(3.5,7.0)
ax2.set_yticks([3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0]); ax2.set_yticklabels(["",r'$10^4$',"",r'$10^5$',"",r'$10^6$',"",r'$10^7$'], color='black')
ax2.set_xticks([0,3,6,9,12,15,18,21,24,27,30,33,36]); ax2.set_xticklabels([0,"",6,"",12,"",18,"",24,"",30,"",36], color='black')
line1, = ax2.plot([xx1, xx_mean, xx2, xx3], [yy1, yy_mean, yy2, yy3], label='linear  fit  (0-12 month)', color="#e89a10", linewidth=0.6,  linestyle='-', alpha=0.6)
line2, = ax2.plot(np.linspace(0,36,37), relay_correct0[0:37], label='same  signs  with  0-12 month', markerfacecolor='#DB9BA1', markeredgecolor='#B83945', marker='o', markeredgewidth=0.60, linewidth=0.0,  markersize=5.5, linestyle='-')
line3, = ax2.plot(np.linspace(0,36,37), relay_notsame0[0:37], label='different  from  0-12 month', markerfacecolor='#9AB9C0', markeredgecolor='#377483', marker='*', markeredgewidth=0.55, linewidth=0.0,  markersize=7.0, linestyle='-')

# ax2.legend(loc=(0.38, 0.63), fontsize=8.5, frameon=False)
legend1 = ax2.legend(handles=[line2,line3], loc=(0.30, 0.75), fontsize=9.5, frameon=False)
ax2.add_artist(legend1)

plt.title("Maintained   &   non-maintained   links", size=11.5)
ax2.text(1.3, 5.50, s='decrease  18%', fontsize=9.5, color='#e89a10')
ax2.text(1.8, 5.24, s='each  month', fontsize=9.5, color='#e89a10')

# ax2.text(22.5, 4.8, s=r'$y=10^{6.90-0.0881x}$', fontsize=10.0, color='#e89a10',bbox=dict(facecolor='#e89a10',alpha=0.1,edgecolor='#e89a10'))
ax2.text(-1.2, 7.50, s='b', fontsize=14.5, color='black', fontweight='bold')







fig.savefig('2_analysis/Figure1.png', bbox_inches = 'tight')
plt.show()


