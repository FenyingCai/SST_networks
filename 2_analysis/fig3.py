# %%
import numpy as np
import xarray as xr




## ------------------------------------------------------------------------------------------------------------------------------------------- ##
ds1 = xr.open_dataset("2_analysis/fig3_CMIP6/Link2x2_Historical_number_lag0_36.nc")
link_number = np.log10(np.nanmean(ds1.link_number_23models[:,:].data,axis=0))
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






ds1 = xr.open_dataset("2_analysis/fig3_CMIP6/historicalpure_2x2_sign_same_wrong_012.nc")
relay_same0 = np.log10(np.nanmean(ds1.link_number_relay_same[:,:].data,axis=0))
relay_wrong0 = np.log10(np.nanmean(ds1.link_number_relay_wrong[:,:].data,axis=0))
relay_mid0 = np.log10(np.nanmean(ds1.link_number_relay_mid[:,:].data,axis=0))
relay_notsame0 = np.log10(np.nanmean(ds1.link_number_relay_notsame[:,:].data,axis=0))
print(" ------------- mid ------------- ")
print(relay_mid0)

beta, resids, rank, s = np.linalg.lstsq(X, relay_same0[0:13],  )  ## linear fit
print(" reg (HadISST same,  12 months) = ", beta)

xx_mean = 6.0
yy_mean = np.nanmean(relay_same0[0:13])
reg_xxyy = beta[1]  ## log10(yy) = axx + b,   yy = 10**(axx+b)

xx1 = xx_mean - 6
xx2 = xx_mean + 18
xx3 = xx_mean + 30
yy1 = yy_mean - 6*reg_xxyy
yy2 = yy_mean + 18*reg_xxyy
yy3 = yy_mean + (18+12)*reg_xxyy

print(" decreasing ratio (1 month, same) = ", 10**reg_xxyy)
print(" decreasing ratio (6 month, same) = ", 10**(reg_xxyy*6))
print(" decreasing ratio (12 month, same) = ", 10**(reg_xxyy*12))
print(" yy0 (HadISST, same) = ", yy1,"  ",10**yy1)










## ------------------------------------------------------------------------------------------------------------------------------------------------------------ ##
ds1 = xr.open_dataset("2_analysis/fig3_CMIP6/historicalpure2x2_real_estimated_link_number_34models.nc")
estimate_year2_link = (ds1.estimate_year23_link)
real_year2_link = (ds1.real_year23_link)

diff = real_year2_link - estimate_year2_link  
print(diff)

sorted_indices = sorted(range(len(diff)), key=lambda i: diff[i])
print(sorted_indices)

High_2 = [12,1,10,22, 19,24,13,28]  ## 8
Low_2  = [5,27,31,20, 18,7,8,33]  ## 8

ds1 = xr.open_dataset("2_analysis/fig3_CMIP6/Link2x2_Historical_number_lag0_36.nc")
link_number_Low2 = np.log10(np.nanmean(ds1.link_number_23models[Low_2,:].data,axis=0))
link_number_High2 = np.log10(np.nanmean(ds1.link_number_23models[High_2,:].data,axis=0))

# ds1 = xr.open_dataset("2_analysis/fig3_CMIP6/historicalpure_2x2_sign_same_wrong_012.nc")
# relay_same_Low2 = np.log10(np.nanmean(ds1.link_number_relay_same[Low_2,:].data,axis=0))
# relay_wrong_Low2 = np.log10(np.nanmean(ds1.link_number_relay_wrong[Low_2,:].data,axis=0))
# relay_same_High2 = np.log10(np.nanmean(ds1.link_number_relay_same[High_2,:].data,axis=0))
# relay_wrong_High2 = np.log10(np.nanmean(ds1.link_number_relay_wrong[High_2,:].data,axis=0))


ds1 = xr.open_dataset("2_analysis/fig3_CMIP6/historicalpure_2x2_sign_same_wrong_012.nc")
relay_same_year2 = np.log10(np.nanmean(ds1.link_number_relay_same[:,13:25].data,axis=1))
relay_wrong_year2 = np.log10(np.nanmean(ds1.link_number_relay_wrong[:,13:25].data,axis=1))
relay_wrong_per_year2 = relay_wrong_year2 / (relay_same_year2 + relay_wrong_year2)




beta, resids, rank, s = np.linalg.lstsq(X, link_number_Low2[0:13],  )  ## linear fit
# print(" reg (HadISST,  12 months) = ", beta)
x_mean_Low2 = 6.0
y_mean_Low2 = np.nanmean(link_number_Low2[0:13])
reg_xy_Low2 = beta[1]  ## log10(y) = ax + b,   y = 10**(ax+b)

x1_Low2 = x_mean_Low2 - 6
x2_Low2 = x_mean_Low2 + 18
x3_Low2 = x_mean_Low2 + 30
y1_Low2 = y_mean_Low2 - 6*reg_xy_Low2
y2_Low2 = y_mean_Low2 + 18*reg_xy_Low2
y3_Low2 = y_mean_Low2 + (18+12)*reg_xy_Low2

beta, resids, rank, s = np.linalg.lstsq(X, link_number_High2[0:13],  )  ## linear fit
# print(" reg (HadISST,  12 months) = ", beta)
x_mean_High2 = 6.0
y_mean_High2 = np.nanmean(link_number_High2[0:13])
reg_xy_High2 = beta[1]  ## log10(y) = ax + b,   y = 10**(ax+b)

x1_High2 = x_mean_High2 - 6
x2_High2 = x_mean_High2 + 18
x3_High2 = x_mean_High2 + 30
y1_High2 = y_mean_High2 - 6*reg_xy_High2
y2_High2 = y_mean_High2 + 18*reg_xy_High2
y3_High2 = y_mean_High2 + (18+12)*reg_xy_High2













##--##--##--##--   34 models   --##--##--##--##
ds3 = xr.open_dataset("2_analysis/fig3_CMIP6/historicalpure2x2_real_estimated_link_number_34models.nc")
estimate_year2_link = ds3.estimate_year2_link
estimate_year3_link = ds3.estimate_year3_link
estimate_year23_link = ds3.estimate_year23_link
real_year2_link = ds3.real_year2_link
real_year3_link = ds3.real_year3_link
real_year23_link = ds3.real_year23_link
diff = real_year23_link - estimate_year23_link

estimate_year23_link_35 = np.zeros(35)
real_year23_link_35 = np.zeros(35)
estimate_year23_link_35[0] = np.nanmean(estimate_year23_link)
estimate_year23_link_35[1:] = estimate_year23_link
real_year23_link_35[0] = np.nanmean(real_year23_link)
real_year23_link_35[1:] = real_year23_link



##--##--##--##--   34 models   --##--##--##--##
ds1 = xr.open_dataset("2_analysis/fig3_CMIP6/historicalpure_2x2_sign_same_wrong_012.nc")
relay_same_year23 = (np.nanmean(ds1.link_number_relay_same[:,13:37].data,axis=1))
relay_wrong_year23 = (np.nanmean(ds1.link_number_relay_wrong[:,13:37].data,axis=1))
relay_notsame_year23 = (np.nanmean(ds1.link_number_relay_wrong[:,13:37].data+ds1.link_number_relay_mid[:,13:37].data,axis=1))
relay_wrong_per_year23 = relay_wrong_year23 / (relay_wrong_year23 + relay_same_year23)
relay_notsame_per_year23 = relay_notsame_year23 / (relay_notsame_year23 + relay_same_year23)

# diff = np.nanlog10(10**real_year23_link - 10**estimate_year23_link)
diff = (real_year23_link - estimate_year23_link)

print("                         ")
print(" Cor = ",np.corrcoef(diff, relay_wrong_year23)[0,1])
print(" Cor = ",np.corrcoef(diff, relay_same_year23)[0,1])
print(" Cor = ",np.corrcoef(diff, relay_wrong_per_year23)[0,1])
print(" Cor = ",np.corrcoef(diff, relay_notsame_per_year23)[0,1])







##-----------------------------------------------------------------##
##--##--##--##--##--##--   relay transport   --##--##--##--##--##--##
ds2 = xr.open_dataset("2_analysis/fig3_CMIP6_relay46/historical_2x2_sign_lag6_M1.nc")
link_number_all_lag4_M1 = np.nanmean(ds2.link_number_all.data, axis=0)
relay_same_lag4_M1 = np.nanmean(ds2.link_number_relay_same.data, axis=0)
relay_same_lag4_M1_top = np.nanpercentile(ds2.link_number_relay_same.data, 90, axis=0)
relay_same_lag4_M1_bottom = np.nanpercentile(ds2.link_number_relay_same.data, 10, axis=0)
relay_wrong_lag4_M1 = np.nanmean(ds2.link_number_relay_wrong.data, axis=0)

relay_same_lag4_M1 = relay_same_lag4_M1/link_number_all_lag4_M1
relay_same_lag4_M1_top = relay_same_lag4_M1_top/link_number_all_lag4_M1
relay_same_lag4_M1_bottom = relay_same_lag4_M1_bottom/link_number_all_lag4_M1
relay_wrong_lag4_M1 = relay_wrong_lag4_M1/link_number_all_lag4_M1


ds2 = xr.open_dataset("2_analysis/fig3_CMIP6_relay46/historical_2x2_sign_lag6_M2.nc")
link_number_all_lag4_M2 = np.nanmean(ds2.link_number_all.data, axis=0)
relay_same_lag4_M2 = np.nanmean(ds2.link_number_relay_same.data, axis=0)
relay_same_lag4_M2_top = np.nanpercentile(ds2.link_number_relay_same.data, 90, axis=0)
relay_same_lag4_M2_bottom = np.nanpercentile(ds2.link_number_relay_same.data, 10, axis=0)
relay_wrong_lag4_M2 = np.nanmean(ds2.link_number_relay_wrong.data, axis=0)

relay_same_lag4_M2 = relay_same_lag4_M2/link_number_all_lag4_M2
relay_same_lag4_M2_top = relay_same_lag4_M2_top/link_number_all_lag4_M2
relay_same_lag4_M2_bottom = relay_same_lag4_M2_bottom/link_number_all_lag4_M2
relay_wrong_lag4_M2 = relay_wrong_lag4_M2/link_number_all_lag4_M2


ds2 = xr.open_dataset("2_analysis/fig3_CMIP6_relay46/historical_2x2_sign_lag6_M3.nc")
link_number_all_lag4_M3 = np.nanmean(ds2.link_number_all.data, axis=0)
relay_same_lag4_M3 = np.nanmean(ds2.link_number_relay_same.data, axis=0)
relay_same_lag4_M3_top = np.nanpercentile(ds2.link_number_relay_same.data, 90, axis=0)
relay_same_lag4_M3_bottom = np.nanpercentile(ds2.link_number_relay_same.data, 10, axis=0)
relay_wrong_lag4_M3 = np.nanmean(ds2.link_number_relay_wrong.data, axis=0)

relay_same_lag4_M3 = relay_same_lag4_M3/link_number_all_lag4_M3
relay_same_lag4_M3_top = relay_same_lag4_M3_top/link_number_all_lag4_M3
relay_same_lag4_M3_bottom = relay_same_lag4_M3_bottom/link_number_all_lag4_M3
relay_wrong_lag4_M3 = relay_wrong_lag4_M3/link_number_all_lag4_M3


ds2 = xr.open_dataset("2_analysis/fig3_CMIP6_relay46/historical_2x2_sign_lag6_M4.nc")
link_number_all_lag4_M4 = np.nanmean(ds2.link_number_all.data, axis=0)
relay_same_lag4_M4 = np.nanmean(ds2.link_number_relay_same.data, axis=0)
relay_same_lag4_M4_top = np.nanpercentile(ds2.link_number_relay_same.data, 90, axis=0)
relay_same_lag4_M4_bottom = np.nanpercentile(ds2.link_number_relay_same.data, 10, axis=0)
relay_wrong_lag4_M4 = np.nanmean(ds2.link_number_relay_wrong.data, axis=0)

relay_same_lag4_M4 = relay_same_lag4_M4/link_number_all_lag4_M4
relay_same_lag4_M4_top = relay_same_lag4_M4_top/link_number_all_lag4_M4
relay_same_lag4_M4_bottom = relay_same_lag4_M4_bottom/link_number_all_lag4_M4
relay_wrong_lag4_M4 = relay_wrong_lag4_M4/link_number_all_lag4_M4





ds2 = xr.open_dataset("2_analysis/fig3_CMIP6_relay46/historical_2x2_sign_lag6_M1.nc")
link_number_all_lag6_M1 = np.nanmean(ds2.link_number_all.data, axis=0)
relay_same_lag6_M1 = np.nanmean(ds2.link_number_relay_same.data, axis=0)
relay_same_lag6_M1_top = np.nanpercentile(ds2.link_number_relay_same.data, 90, axis=0)
relay_same_lag6_M1_bottom = np.nanpercentile(ds2.link_number_relay_same.data, 10, axis=0)
relay_wrong_lag6_M1 = np.nanmean(ds2.link_number_relay_wrong.data, axis=0)

relay_same_lag6_M1 = relay_same_lag6_M1/link_number_all_lag6_M1
relay_same_lag6_M1_top = relay_same_lag6_M1_top/link_number_all_lag6_M1
relay_same_lag6_M1_bottom = relay_same_lag6_M1_bottom/link_number_all_lag6_M1
relay_wrong_lag6_M1 = relay_wrong_lag6_M1/link_number_all_lag6_M1


ds2 = xr.open_dataset("2_analysis/fig3_CMIP6_relay46/historical_2x2_sign_lag6_M2.nc")
link_number_all_lag6_M2 = np.nanmean(ds2.link_number_all.data, axis=0)
relay_same_lag6_M2 = np.nanmean(ds2.link_number_relay_same.data, axis=0)
relay_same_lag6_M2_top = np.nanpercentile(ds2.link_number_relay_same.data, 90, axis=0)
relay_same_lag6_M2_bottom = np.nanpercentile(ds2.link_number_relay_same.data, 10, axis=0)
relay_wrong_lag6_M2 = np.nanmean(ds2.link_number_relay_wrong.data, axis=0)

relay_same_lag6_M2 = relay_same_lag6_M2/link_number_all_lag6_M2
relay_same_lag6_M2_top = relay_same_lag6_M2_top/link_number_all_lag6_M2
relay_same_lag6_M2_bottom = relay_same_lag6_M2_bottom/link_number_all_lag6_M2
relay_wrong_lag6_M2 = relay_wrong_lag6_M2/link_number_all_lag6_M2


ds2 = xr.open_dataset("2_analysis/fig3_CMIP6_relay46/historical_2x2_sign_lag6_M3.nc")
link_number_all_lag6_M3 = np.nanmean(ds2.link_number_all.data, axis=0)
relay_same_lag6_M3 = np.nanmean(ds2.link_number_relay_same.data, axis=0)
relay_same_lag6_M3_top = np.nanpercentile(ds2.link_number_relay_same.data, 90, axis=0)
relay_same_lag6_M3_bottom = np.nanpercentile(ds2.link_number_relay_same.data, 10, axis=0)
relay_wrong_lag6_M3 = np.nanmean(ds2.link_number_relay_wrong.data, axis=0)

relay_same_lag6_M3 = relay_same_lag6_M3/link_number_all_lag6_M3
relay_same_lag6_M3_top = relay_same_lag6_M3_top/link_number_all_lag6_M3
relay_same_lag6_M3_bottom = relay_same_lag6_M3_bottom/link_number_all_lag6_M3
relay_wrong_lag6_M3 = relay_wrong_lag6_M3/link_number_all_lag6_M3


ds2 = xr.open_dataset("2_analysis/fig3_CMIP6_relay46/historical_2x2_sign_lag6_M4.nc")
link_number_all_lag6_M4 = np.nanmean(ds2.link_number_all.data, axis=0)
relay_same_lag6_M4 = np.nanmean(ds2.link_number_relay_same.data, axis=0)
relay_same_lag6_M4_top = np.nanpercentile(ds2.link_number_relay_same.data, 90, axis=0)
relay_same_lag6_M4_bottom = np.nanpercentile(ds2.link_number_relay_same.data, 10, axis=0)
relay_wrong_lag6_M4 = np.nanmean(ds2.link_number_relay_wrong.data, axis=0)

relay_same_lag6_M4 = relay_same_lag6_M4/link_number_all_lag6_M4
relay_same_lag6_M4_top = relay_same_lag6_M4_top/link_number_all_lag6_M4
relay_same_lag6_M4_bottom = relay_same_lag6_M4_bottom/link_number_all_lag6_M4
relay_wrong_lag6_M4 = relay_wrong_lag6_M4/link_number_all_lag6_M4






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
  ax.tick_params(axis='both', length=6.0, width=1.0, labelsize=10)
  ax.spines['bottom'].set_linewidth(1.0)
  ax.spines['top'].set_linewidth(1.0)
  ax.spines['left'].set_linewidth(1.0)
  ax.spines['right'].set_linewidth(1.0)
  ax.spines['left'].set_color('black')
  ax.tick_params(axis='y', right=False, length=6.0, width=1.0, labelsize=10.1, color='black')
  return ax


##--##--##--  create figure  --##--##--##
width_in_inches = 210 / 25.4  ## A4 page
height_in_inches = 297 / 25.4

fig = plt.figure(dpi=400,figsize=(width_in_inches,height_in_inches))



ax1 = plt.axes([0.02,0.815, 0.44,0.14])
create_map(ax1)
ax1.set_xlabel('', size=10)
ax1.set_ylabel('Link   number', size=12)
plt.rcParams['text.usetex'] = False
plt.ylim(3.5,7.0)
ax1.set_yticks([3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0]); ax1.set_yticklabels(["",r'$10^4$',"",r'$10^5$',"",r'$10^6$',"",r'$10^7$'], color='black')
ax1.set_xticks([0,3,6,9,12,15,18,21,24,27,30,33,36]); ax1.set_xticklabels([0,"",6,"",12,"",18,"",24,"",30,"",36], color='black')
ax1.plot(np.linspace(0,12,13), link_number[0:13], label='0-12  months', markerfacecolor='#DB9BA1', markeredgecolor='#B83945', marker='o', markeredgewidth=0.60, linewidth=0.0,  markersize=5.0, linestyle='-')
ax1.plot(np.linspace(13,36,24), link_number[13:37], label='13-36  months', markerfacecolor='#9AB9C0', markeredgecolor='#377483', marker='*', markeredgewidth=0.55, linewidth=0.0,  markersize=6.4, linestyle='-')

ax1.plot([x1, x_mean, x2, x3], [y1, y_mean, y2, y3], label='linear  fit  (0-12 month)', color="#e89a10", linewidth=0.6,  linestyle='-', alpha=0.6)
ax1.legend(loc=(0.035, 0.065), fontsize=10.0, frameon=False)
plt.title("Simulated   links", size=13)
# ax1.text(1.9, 5.55, s='decrease  18%', fontsize=8.5, color='#e89a10')
# ax1.text(2.3, 5.25, s='each  month', fontsize=8.5, color='#e89a10')

# ax1.text(20.0, 4.5, s=r'$y=10^{6.90-0.0875x}$', fontsize=9.0, color='#e89a10',bbox=dict(facecolor='#e89a10',alpha=0.1,edgecolor='#e89a10'))
ax1.text(-2.1, 7.55, s='a', fontsize=16, color='black', fontweight='bold')






ax2 = plt.axes([0.53,0.815, 0.44,0.14])
create_map(ax2)
ax2.set_xlabel('', size=10)
ax2.set_ylabel('', size=9)
plt.rcParams['text.usetex'] = False
plt.ylim(3.5,7.0)
ax2.set_yticks([3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0]); ax2.set_yticklabels(["",r'$10^4$',"",r'$10^5$',"",r'$10^6$',"",r'$10^7$'], color='black')
ax2.set_xticks([0,3,6,9,12,15,18,21,24,27,30,33,36]); ax2.set_xticklabels([0,"",6,"",12,"",18,"",24,"",30,"",36], color='black')
line1, = ax2.plot([xx1, xx_mean, xx2, xx3], [yy1, yy_mean, yy2, yy3], label='linear  fit  (0-12 month)', color="#e89a10", linewidth=0.6,  linestyle='-', alpha=0.6)
line2, = ax2.plot(np.linspace(0,36,37), relay_same0[0:37], label='same  signs  with  0-12 month', markerfacecolor='#DB9BA1', markeredgecolor='#B83945', marker='o', markeredgewidth=0.60, linewidth=0.0,  markersize=5.0, linestyle='-')
line3, = ax2.plot(np.linspace(0,36,37), relay_notsame0[0:37], label='different  from  0-12 month', markerfacecolor='#9AB9C0', markeredgecolor='#377483', marker='*', markeredgewidth=0.55, linewidth=0.0,  markersize=6.4, linestyle='-')
# line4, = ax2.plot(np.linspace(0,36,37), relay_mid0[0:37], label='opposite  signs  with  0-12 month', markerfacecolor='none', markeredgecolor='green', marker='^', markeredgewidth=0.55, linewidth=0.0,  markersize=5.0, linestyle='-')

legend1 = ax2.legend(handles=[line2,line3], loc=(0.26, 0.67), fontsize=10.0, frameon=False)
# legend1 = ax2.legend(handles=[line2], loc=(0.26, 0.68), fontsize=10.0, frameon=False)
# legend2 = ax2.legend(handles=[line3], loc=(0.26, 0.15), fontsize=10.0, frameon=False)
ax2.add_artist(legend1)
# ax2.add_artist(legend2)
# legend2 = ax2.legend(handles=[line3], loc=(0.03, 0.25), fontsize=8.5, frameon=False)
# ax2.add_artist(legend2)
plt.title("Maintained  &  non-maintained  links", size=13)
# ax2.text(2.0, 5.55, s='decrease  18%', fontsize=8.5, color='#e89a10')
# ax2.text(2.4, 5.25, s='each  month', fontsize=8.5, color='#e89a10')

# ax2.text(24.0, 4.1, s=r'$y=10^{6.90-0.0881x}$', fontsize=9.0, color='#e89a10',bbox=dict(facecolor='#e89a10',alpha=0.1,edgecolor='#e89a10'))
ax2.text(-2.1, 7.55, s='b', fontsize=16, color='black', fontweight='bold')






##------------------------------------------##
##--##--##--##--  fig 3. (c)  --##--##--##--##
model_iname = ['M','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20',\
              '21','22','23','24','25','26','27','28','29','30','31','32','33','34']
# model_name = ['      ACCESS-CM2 ', '     BCC-CSM2-MR ', '     CAMS-CSM1-0 ', '     CESM2-WACCM ', '           CIESM ', \
#             '    CMCC-CM2-SR5 ', '       CMCC-ESM2 ', '       CanESM5-1 ', '         CanESM5 ', 'EC-Earth3-Veg-LR ', \
#             '   EC-Earth3-Veg ', '       EC-Earth3 ', '     FGOALS-f3-L ', '       FGOALS-g3 ', '     FIO-ESM-2-0 ', \
#             '        GFDL-CM4 ', '       GFDL-ESM4 ', ' HadGEM3-GC31-LL ', ' HadGEM3-GC31-MM ', '        IITM-ESM ', \
#             '       INM-CM4-8 ', '       INM-CM5-0 ', '    IPSL-CM6A-LR ', '      KACE-1-0-G ', '       KIOST-ESM ', \
#             '          MIROC6 ', '   MPI-ESM1-2-HR ', '   MPI-ESM1-2-LR ', '      MRI-ESM2-0 ', '           NESM3 ', \
#             '      NorESM2-LM ', '      NorESM2-MM ', '         TaiESM1 ', '     UKESM1-0-LL ']



ax3 = plt.axes([0.02,0.60, 0.95,0.14])
create_map(ax3)
ax3.set_xlim(-0.3,69.3)
ax3.set_ylim(5.0,7.5)
ax3.set_yticks([5.0,5.5,6.0,6.5,7.0,7.5,8.0]); ax3.set_yticklabels([r'$10^5$',"",r'$10^6$',"",r'$10^7$',"",r'$10^8$'], color='black')
ax3.set_xticks(np.linspace(1.0,68.0,35)); ax3.set_xticklabels(model_iname, color='black',rotation=0)
ax3.tick_params(axis='y', right=False, length=6.0, width=1.0, labelsize=10.1, color='black')
ax3.tick_params(axis='x', right=False, length=6.0, width=1.0, labelsize=10.0, color='black')#, which='major',pad=-10)
bar1 = ax3.bar(np.linspace(0.7,67.7,35), real_year23_link_35, label='simulated  link  number', color='#e89a10', edgecolor='#e89a10', width = 0.45)
bar2 = ax3.bar(np.linspace(1.3,68.3,35), estimate_year23_link_35, label='expectation  from  0-12-month  exponential  decay', color='#377483', edgecolor='#377483', width = 0.45)
ax3.plot([64.0], [estimate_year23_link[31]+0.16], markerfacecolor='red', markeredgecolor='purple', marker='x', markeredgewidth=0.85, linewidth=0.0,  markersize=6.0)


ax3.set_title('Long-delayed   (13-36  months)   significant   links   simulated   by   models', fontsize=13)
ax3.set_xlabel(' ', fontsize=14)
ax3.set_ylabel(' Link   number ', fontsize=12)


# ax3.legend(loc=(0.015, 0.695), fontsize=10.5, frameon=False)
legend3 = ax3.legend(handles=[bar1], loc=(0.03, 0.76), fontsize=10.5, frameon=False)
legend4 = ax3.legend(handles=[bar2], loc=(0.39, 0.76), fontsize=10.5, frameon=False)
ax3.add_artist(legend3)
ax3.add_artist(legend4)
ax3.text(-1.1, 8.45, s='c', fontsize=16, color='black', fontweight='bold')





##------------------------------------------##
##--##--##--##--  fig 3. (d)  --##--##--##--##
print(" relay_notsame_per_year23 = ",np.nanmin(relay_notsame_per_year23),"  ",np.nanmax(relay_notsame_per_year23))
print(" diff = ",np.nanmin(diff),"  ",np.nanmax(diff))
ax4 = plt.axes([0.022,0.34, 0.285,0.170])
create_map(ax4)
ax4.set_xlim(0.45,0.95)
ax4.set_ylim(-0.1,1.0)
ax4.set_xticks(np.linspace(0.45,0.95,11)); ax4.set_xticklabels(['','50%','','60%','','70%','','80%','','90%',''], color='black',rotation=0)
ax4.set_yticks(np.linspace(-0.1,1.0,12)); ax4.set_yticklabels(['','0','','0.2','','0.4','','0.6','','0.8','','1.0'], color='black',rotation=0)


model_name=['1. ACCESS-CM2','2. BCC-CSM2-MR','3. CAMS-CSM1-0','4. CESM2-WACCM','5. CIESM',\
'6. CMCC-CM2-SR5','7. CMCC-ESM2','8. CanESM5-1','9. CanESM5','10. EC-Earth3-Veg-LR',\
'11. EC-Earth3-Veg','12. EC-Earth3','13. FGOALS-f3-L','14. FGOALS-g3','15. FIO-ESM-2-0',\
'16. GFDL-CM4','17. GFDL-ESM4','18. HadGEM3-GC31-LL','19. HadGEM3-GC31-MM','20. IITM-ESM',\
'21. INM-CM4-8','22. INM-CM5-0','23. IPSL-CM6A-LR','24. KACE-1-0-G','25. KIOST-ESM',\
'26. MIROC6','27. MPI-ESM1-2-HR','28. MPI-ESM1-2-LR','29. MRI-ESM2-0','30. NESM3',\
'31. NorESM2-LM','32. NorESM2-MM','33. TaiESM1','34. UKESM1-0-LL']


model_marker = ['^','o','s','D','*','x','2', 
                '^','x','s','*','D','2','o', \
                '^','o','s','D','*','x','2', \
                '^','o','s','D','*','x','2', \
                '^','o','s','x','*','D','2']
model_color = ['red','red','red','red','red','red','red',\
               'blue','blue','blue','blue','blue','blue','blue',\
               'orange','orange','orange','orange','orange','orange','orange',\
                'green','green','green','green','green','green','green',\
               'purple','purple','purple','purple','purple','purple','purple']
model_size =   [6.5,6.5,6.3,5.5,6.8,6.5,7.0, 
                6.5,6.5,6.3,6.8,5.5,7.0,6.5, \
                6.5,6.5,6.3,5.5,6.8,6.5,7.0, \
                6.5,6.5,6.3,5.5,6.8,6.5,7.0, \
                6.5,6.5,6.3,6.5,6.8,5.5,7.0]
for imodel in range(34):
  ax4.plot(relay_notsame_per_year23[imodel], diff[imodel], label=model_name[imodel], marker=model_marker[imodel], markerfacecolor='none', markeredgecolor=model_color[imodel], markeredgewidth=1.0, linewidth=0.0,  markersize=model_size[imodel])

ax4.set_ylabel('log(simulated)  -  log(estimated)', fontsize=12)
ax4.set_xlabel('different-sign   proportion', fontsize=12)
ax4.set_title('Inter-model   dependence', fontsize=13)
ax4.text(0.415, 1.18, s='d', fontsize=16, color='black', fontweight='bold')

ax4.legend(loc=(1.055, -0.16), ncol=3, columnspacing=0.45, handletextpad=0.16, fontsize=9.6, frameon=False)
ax4.text(0.535, 0.79, s='R = 0.75', fontsize=11.0, color='black')
ax4.text(0.520, 0.67, s='(p<0.0001)', fontsize=11.0, color='black')
print(" different sign (%) ",np.nanmin(relay_notsame_per_year23), np.nanmax(relay_notsame_per_year23))













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
  ax.tick_params(axis='y', right=False, length=5.0, width=1.0, labelsize=10.0, color='black')
  ax.tick_params(axis='x', right=False, length=5.0, width=1.0, labelsize=11.5, color='black')
  ax.set_xlabel('', size=10.5)
  ax.set_ylabel('', size=13.0)
  return ax


x = np.linspace(0,36,37)
y0 = np.linspace(0.0,0.0,37)
y7 = np.linspace(0.7,0.7,37)


ax33 = plt.axes([0.018,0.15, 0.458,0.09])
create_map(ax33)  ## C93735
ax33.plot(np.linspace(0,36,37), relay_same_lag6_M1, label='same', color='#E5637B', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.3,  markersize=4.0, linestyle='-')
ax33.plot(np.linspace(0,36,37), relay_wrong_lag6_M1, label='opposite', color='#E5637B', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.4,  markersize=4.0, linestyle='--')
ax33.plot([6], [relay_same_lag6_M1[6]+0.05], color='#E5637B', markerfacecolor='#E5637B', markeredgecolor='#E5637B', marker='v', markeredgewidth=1.0,  markersize=4.0)
ax33.plot([6], [0.05], color='#E5637B', markerfacecolor='#E5637B', markeredgecolor='#E5637B', marker='v', markeredgewidth=1.0,  markersize=4.0)
ax33.fill_between(x,y0, relay_same_lag6_M1, where=(relay_same_lag6_M1>=y7), alpha=0.2, color='#E5637B')
# ax33.legend(loc=(0.48, 0.43), fontsize=8.5, frameon=False)
ax33.text(4.35, relay_same_lag6_M1[6]-0.65, s='100%', fontsize=11.0, color='#E5637B')
ax33.text(-1.5, 1.55, s='e', fontsize=16.0, color='black', fontweight='bold')
ax33.text(16.5, 1.40, s='Signal   relay   transport   simulated   by   models', fontsize=13.0, color='black')
ax33.text(31.5, 0.90, s='('+r"$A_6$"+")"r"$^1$", fontsize=12.0, color='black')


ax34 = plt.axes([0.018,0.06, 0.458,0.09])
create_map(ax34)
ax34.set_xticks([0,3,6,9,12,15,18,21,24,27,30,33,36]); ax34.set_xticklabels([r"$A_0$","",r"$A_6$","",r"$A_{12}$","",r"$A_{18}$","",r"$A_{24}$","",r"$A_{30}$","",r"$A_{36}$"], color='black')
ax34.plot(np.linspace(0,36,37), relay_same_lag6_M2, label='same', color='#D2AA3A', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.3,  markersize=4.0, linestyle='-')
ax34.plot(np.linspace(0,36,37), relay_wrong_lag6_M2, label='opposite', color='#D2AA3A', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.4,  markersize=4.0, linestyle='--')
ax34.plot([12], [relay_same_lag6_M2[12]+0.05], color='#D2AA3A', markerfacecolor='#D2AA3A', markeredgecolor='#D2AA3A', marker='v', markeredgewidth=1.0,  markersize=4.0)
ax34.plot([12], [0.05], color='#D2AA3A', markerfacecolor='#D2AA3A', markeredgecolor='#D2AA3A', marker='v', markeredgewidth=1.0,  markersize=4.0)
ax34.fill_between(x,y0, relay_same_lag6_M2, where=(relay_same_lag6_M2>=y7), alpha=0.2, color='#D2AA3A')
ax34.text(9.8, relay_same_lag6_M2[12]-0.35, s='84.9%', fontsize=11.0, color='#D2AA3A')
ax34.text(31.5, 0.90, s='('+r"$A_6$"+")"r"$^2$", fontsize=12.0, color='black')


ax35 = plt.axes([0.505,0.15, 0.458,0.09])
create_map(ax35)
ax35.set_yticks(np.linspace(0.0,1.0,5)); ax35.set_yticklabels(["","","","",""], color='black')
ax35.set_xticks([0,3,6,9,12,15,18,21,24,27,30,33,36]); ax35.set_xticklabels(["","","","","","","","","","","","",""], color='black')
ax35.plot(np.linspace(0,36,37), relay_same_lag6_M3, label='same', color='#009E74', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.3,  markersize=4.0, linestyle='-')
ax35.plot(np.linspace(0,36,37), relay_wrong_lag6_M3, label='opposite', color='#009E74', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.4,  markersize=4.0, linestyle='--')
ax35.plot([18], [relay_same_lag6_M3[18]+0.05], color='#009E74', markerfacecolor='#009E74', markeredgecolor='#009E74', marker='v', markeredgewidth=1.0,  markersize=4.0)
ax35.plot([18], [0.05], color='#009E74', markerfacecolor='#009E74', markeredgecolor='#009E74', marker='v', markeredgewidth=1.0,  markersize=4.0)
ax35.fill_between(x,y0, relay_same_lag6_M3, where=(relay_same_lag6_M3>=y7), alpha=0.17, color='#009E74')
ax35.text(15.8, relay_same_lag6_M3[18]-0.35, s='86.8%', fontsize=11.0, color='#009E74')
ax35.text(31.5, 0.90, s='('+r"$A_6$"+")"r"$^3$", fontsize=12.0, color='black')


ax36 = plt.axes([0.505,0.06, 0.458,0.09])
create_map(ax36)
ax36.set_yticks(np.linspace(0.0,1.0,5)); ax36.set_yticklabels(["","","","",""], color='black')
ax36.set_xticks([0,3,6,9,12,15,18,21,24,27,30,33,36]); ax36.set_xticklabels([r"$A_0$","",r"$A_6$","",r"$A_{12}$","",r"$A_{18}$","",r"$A_{24}$","",r"$A_{30}$","",r"$A_{36}$"], color='black')
ax36.plot(np.linspace(0,36,37), relay_same_lag6_M4, label='same', color='#57B4E9', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.3,  markersize=4.0, linestyle='-')
ax36.plot(np.linspace(0,36,37), relay_wrong_lag6_M4, label='opposite', color='#57B4E9', markerfacecolor='none', markeredgecolor='black', marker='o', markeredgewidth=0.0,  linewidth=1.4,  markersize=4.0, linestyle='--')
ax36.plot([24], [relay_same_lag6_M4[24]+0.05], color='#57B4E9', markerfacecolor='#57B4E9', markeredgecolor='#57B4E9', marker='v', markeredgewidth=1.0,  markersize=4.0)
ax36.plot([24], [0.05], color='#57B4E9', markerfacecolor='#57B4E9', markeredgecolor='#57B4E9', marker='v', markeredgewidth=1.0,  markersize=4.0)
ax36.fill_between(x,y0, relay_same_lag6_M4, where=(relay_same_lag6_M4>=y7), alpha=0.17, color='#57B4E9')
ax36.text(21.8, relay_same_lag6_M4[24]-0.32, s='74.2%', fontsize=11.0, color='#57B4E9')
ax36.text(31.5, 0.90, s='('+r"$A_6$"+")"r"$^4$", fontsize=12.0, color='black')


print(" model percentage = ",relay_same_lag6_M1[6]," ",relay_same_lag6_M2[12]," ",relay_same_lag6_M3[18]," ",relay_same_lag6_M4[24])






fig.savefig('2_analysis/Figure3_historicalpure_new.png', bbox_inches = 'tight')
plt.show()


