# %%
import numpy as np
import xarray as xr




##--##--##--##--##--##--   HadISST  link  number  (0-36 months)   --##--##--##--##--##--##
ds1 = xr.open_dataset("1_networks/part1_HadISST/Link2x2_number_lag0_36.nc")
link_number = np.log10(ds1.link_number.data)
self_cor_strength = ds1.self_correlated_strength.data


ds8 = xr.open_dataset("1_networks/part4_HadISST_removelowfreq/Link2x2_number_lag0_36_removelow60.nc")
link_number_remove60 = np.log10(ds8.link_number.data)
self_cor_strength_remove60 = ds8.self_correlated_strength.data

ds8 = xr.open_dataset("1_networks/part4_HadISST_removelowfreq/Link2x2_number_lag0_36_removelow72.nc")
link_number_remove72 = np.log10(ds8.link_number.data)
self_cor_strength_remove72 = ds8.self_correlated_strength.data

ds8 = xr.open_dataset("1_networks/part4_HadISST_removelowfreq/Link2x2_number_lag0_36_removelow84.nc")
link_number_remove84 = np.log10(ds8.link_number.data)
self_cor_strength_remove84 = ds8.self_correlated_strength.data



link_number_avg = (link_number_remove30 + link_number_remove36 + link_number_remove42) / 3.0
X = np.zeros([13,2], dtype=float)  ## linear fit
X[:,0] = 1
X[:,1] = np.linspace(1,13,13)
beta, resids, rank, s = np.linalg.lstsq(X, link_number_remove84[0:13],  )  ## linear fit
print(" reg (remove 30-42,   12 months) = ", beta)

x_mean = 6.0
y_mean = np.nanmean(link_number_remove84[0:13])
reg_xy = beta[1]  ## log10(y) = ax + b,   y = 10**(ax+b)

x1 = x_mean - 6
x2 = x_mean + 18
x3 = x_mean + 30
y1 = y_mean - 6*reg_xy
y2 = y_mean + 18*reg_xy
y3 = y_mean + (18+12)*reg_xy



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



ax1 = plt.axes([0.02,0.6, 0.42,0.18])
create_map(ax1)
ax1.set_xlabel('Lagged   months', size=10)
ax1.set_ylabel('Number', size=10)
plt.rcParams['text.usetex'] = False
plt.ylim(4.5,7.1)
ax1.set_yticks([4.5,5.0,5.5,6.0,6.5,7.0]); ax1.set_yticklabels(["",r'$10^5$',"",r'$10^6$',"",r'$10^7$'], color='black')
ax1.set_xticks([0,3,6,9,12,15,18,21,24,27,30,33,36]); ax1.set_xticklabels([0,"",6,"",12,"",18,"",24,"",30,"",36], color='black')
line1, = ax1.plot(np.linspace(0,36,37), link_number[:], label='original  data', markeredgewidth=0.60, linewidth=1.0,  markersize=5.5, linestyle='-')
line2, = ax1.plot(np.linspace(0,36,37), link_number_remove60[:], label='periods < 60 months', markeredgewidth=0.60, linewidth=1.0,  markersize=5.5, linestyle='-')
line3, = ax1.plot(np.linspace(0,36,37), link_number_remove72[:], label='periods < 72 months', markeredgewidth=0.60, linewidth=1.0,  markersize=5.5, linestyle='-')
line4, = ax1.plot(np.linspace(0,36,37), link_number_remove84[:], label='periods < 84 months', markeredgewidth=0.60, linewidth=1.0,  markersize=5.5, linestyle='-')

line5, = ax1.plot([x1, x_mean, x2, x3], [y1, y_mean, y2, y3], label='linear  fit  (0-12 mons,  periods < 84 mons)', color="red", linewidth=1.0,  linestyle='--', alpha=0.6)

legend1 = ax1.legend(handles=[line2, line3], loc=(0.45, 0.73), fontsize=8.5, frameon=False)
legend2 = ax1.legend(handles=[line1, line4, line5], loc=(0.035, 0.04), fontsize=8.5, frameon=False)
ax1.add_artist(legend1)
ax1.text(-1.2, 7.38, s='a', fontsize=13, color='black', fontweight='bold')
plt.title("Link  number", size=10.5)




ax2 = plt.axes([0.55,0.6, 0.42,0.18])
create_map(ax2)
ax2.set_xlabel('Lagged   months', size=10)
ax2.set_ylabel('Strengths', size=10)
plt.rcParams['text.usetex'] = False
plt.ylim(-2000.0,7000.0)
ax2.set_yticks([-2000,-1000,0.0,1000,2000,3000,4000,5000,6000,7000]); ax2.set_yticklabels([-2000,"",0,"",2000,"",4000,"",6000,""], color='black')
ax2.set_xticks([0,3,6,9,12,15,18,21,24,27,30,33,36]); ax2.set_xticklabels([0,"",6,"",12,"",18,"",24,"",30,"",36], color='black')
ax2.plot(np.linspace(0,36,37), self_cor_strength[:], label='original  data', markeredgewidth=0.60, linewidth=1.0,  markersize=5.5, linestyle='-')
ax2.plot(np.linspace(0,36,37), self_cor_strength_remove60[:], label='period < 60 months', markeredgewidth=0.60, linewidth=1.0,  markersize=5.5, linestyle='-')
ax2.plot(np.linspace(0,36,37), self_cor_strength_remove72[:], label='period < 72 months', markeredgewidth=0.60, linewidth=1.0,  markersize=5.5, linestyle='-')
ax2.plot(np.linspace(0,36,37), self_cor_strength_remove84[:], label='period < 84 months', markeredgewidth=0.60, linewidth=1.0,  markersize=5.5, linestyle='-')
ax2.plot([-1,37], [0.0, 0.0], label='', color="gray", linewidth=1.0,  linestyle='--', alpha=0.6)


ax2.legend(loc=(0.42, 0.51), fontsize=8.5, frameon=False)
ax2.text(-1.2, 7900, s='b', fontsize=13, color='black', fontweight='bold')
plt.title("Autocorrelation  strenghts", size=10.5)






fig.savefig('3_extended_data_fig/FigureS9_remove_lowfreq.png', bbox_inches = 'tight')
plt.show()


