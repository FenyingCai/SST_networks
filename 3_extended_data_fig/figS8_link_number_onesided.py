# %%
import numpy as np
import xarray as xr




##--##--##--##--##--##--   HadISST  link  number  (0-36 months)   --##--##--##--##--##--##
ds1 = xr.open_dataset("2_analysis/fig1_sign2x2/Oneside_Link2x2_number_lag0_36.nc")
link_number = np.log10(ds1.link_number.data)
# print(" max = ",np.nanmax(link_number))



ds2 = xr.open_dataset("1_networks/part2_HadISST/Link2x2_number_lag0_36.nc")
link_number_all = np.log10(ds2.link_number.data)
percentage = ds1.link_number.data / (ds2.link_number.data)

print(percentage)







# %%
## Plotting Import
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('agg')

## Define a Map Function for plotting
def create_map(ax):
  # plt.axes( xscale="log" )
  plt.xlim(-0.5,18.0)
  plt.ylim(2.0,5.5)
  ax.tick_params(axis='both', length=6.0, width=1.0, labelsize=9.0)
  ax.spines['bottom'].set_linewidth(1.0)
  ax.spines['top'].set_linewidth(1.0)
  ax.spines['left'].set_linewidth(1.0)
  ax.spines['right'].set_linewidth(1.0)
  ax.spines['left'].set_color('black')
  ax.tick_params(axis='y', right=False, length=6.0, width=1.0, labelsize=9.1, color='black')
  return ax


##--##--##--  create figure  --##--##--##
width_in_inches = 210 / 25.4  ## A4 page
height_in_inches = 297 / 25.4

fig = plt.figure(dpi=400,figsize=(width_in_inches,height_in_inches))



ax1 = plt.axes([0.02,0.6, 0.44,0.18])
create_map(ax1)
ax1.set_xlabel('Lagged   months', size=10)
ax1.set_ylabel('Link   number', size=10)
plt.rcParams['text.usetex'] = False
plt.ylim(5.4,6.65)
ax1.set_yticks([5.5,5.75,6.0,6.25,6.5]); ax1.set_yticklabels([r'$10^{5.5}$',"",r'$10^6$',"",r'$10^{6.5}$'], color='black')
ax1.set_xticks([0,2,4,6,8,10,12,14,16,18]); ax1.set_xticklabels([0,"",4,"",8,"",12,"",16,""], color='black')
ax1.plot(np.linspace(0,18,19), link_number[0:19], label='0-12  months', markerfacecolor='#DB9BA1', markeredgecolor='#B83945', marker='o', markeredgewidth=0.60, linewidth=0.0,  markersize=5.5, linestyle='-')#, alpha=0.5)
ax1.plot([4,4], [1.0,link_number[4]], label='', color="#e89a10", linewidth=1.5,  linestyle='-', alpha=0.6)

# ax1.legend(loc=(0.42, 0.59), fontsize=8.5, frameon=False)
plt.title("One-sided   link   number", size=11.0)
ax1.text(-0.8, 6.8, s='a', fontsize=14, color='black', fontweight='bold')
ax1.text(21.0, 6.8, s='b', fontsize=14, color='black', fontweight='bold')






ax2 = plt.axes([0.54,0.6, 0.44,0.18])
create_map(ax2)
ax2.set_xlabel('Lagged   months', size=10)
ax2.set_ylabel('', size=9)
plt.rcParams['text.usetex'] = False
plt.ylim(0.0,1.0)
ax2.set_yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]) ; ax2.set_yticklabels(["0","","20%","","40%","","60%","","80%","","100%"], color='black')
ax2.set_xticks([0,2,4,6,8,10,12,14,16,18]); ax2.set_xticklabels([0,"",4,"",8,"",12,"",16,""], color='black')
ax2.plot(np.linspace(0,18,19), percentage[0:19], label='', markerfacecolor='#DB9BA1', markeredgecolor='#B83945', marker='o', markeredgewidth=0.60, linewidth=0.0,  markersize=5.5, linestyle='-')#, alpha=0.5)
ax2.plot([6,6], [0.0,percentage[6]], label='', color="#e89a10", linewidth=1.5,  linestyle='-', alpha=0.6)
ax2.plot([4,4], [0.0,percentage[4]], label='', color="#e89a10", linewidth=1.5,  linestyle='-', alpha=0.6)
print(percentage[4])
print(percentage[6])

plt.title("One-sided   link   percentage", size=11.0)
ax2.text(2.65, 0.74, s='69.3%', fontsize=8.5, color='black')
ax2.text(4.65, 0.90, s='85.8%', fontsize=8.5, color='black')






fig.savefig('3_extended_data_fig/FigureS8_link_number_onesided.png', bbox_inches = 'tight')
plt.show()


