# %%
import numpy as np
import xarray as xr



estimate_year2_link = np.zeros(34,float)
estimate_year3_link = np.zeros(34,float)
estimate_year23_link = np.zeros(34,float)
estimate_monthly_link = np.zeros([34,37],float)

for imodel in range(34):
  print(" imodel  =  ", imodel)

  ##--##--##--##--  link number of 6 networks  --##--##--##--##
  ds1 = xr.open_dataset("2_analysis/fig3_CMIP6_sign/Link2x2_Historical_number_lag0_36.nc")
  link_number0 = ds1.link_number_23models.data[imodel,:]
  link_number = np.log10(link_number0)
  # print(" max = ",np.nanmax(link_number))


  ##--##--##--##--  linear fit  --##--##--##--##
  X = np.zeros([13,2], dtype=float)
  X[:,0] = 1
  X[:,1] = np.linspace(1,13,13)
  beta, resids, rank, s = np.linalg.lstsq(X, link_number[0:13],  )
  print("imodel =  ", imodel,"    reg (12 months) = ", beta)


  
  x_mean = 6.0
  y_mean = np.nanmean(link_number[0:13])
  reg_xy = beta[1]  ## log10(y) = ax + b,   y = 10**(ax+b)


  for imonth in range(37): 
    estimate_monthly_link[imodel,imonth] = y_mean - 6*reg_xy + (imonth)*reg_xy   ## imonth=0, lag=0


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

  estimate_year2_link[imodel] = np.log10(np.nansum(estimate_year2_lin))
  estimate_year3_link[imodel] = np.log10(np.nansum(estimate_year3_lin))
  estimate_year23_link[imodel] = np.log10(np.nansum(estimate_year2_lin)+np.nansum(estimate_year3_lin))






ds1 = xr.open_dataset("2_analysis/fig3_CMIP6_sign/Link2x2_Historical_number_lag0_36.nc")
link_number_34models = ds1.link_number_23models.data[:,:]

print("  ")
print(estimate_year2_link-np.log10(np.nansum(link_number_34models[:,13:25],axis=1)))
print("  ")
print(estimate_year3_link-np.log10(np.nansum(link_number_34models[:,25:37],axis=1)))
print("  ")
print(estimate_year23_link-np.log10(np.nansum(link_number_34models[:,13:37],axis=1)))
print("  ")




real_year2_link = np.log10(np.nansum(link_number_34models[:,13:25],axis=1))
real_year3_link = np.log10(np.nansum(link_number_34models[:,25:37],axis=1))
real_year23_link = np.log10(np.nansum(link_number_34models[:,13:37],axis=1))



##--##--##--##--  to nc file  --##--##--##--##
estimate_year2_link_array0 = xr.DataArray(data=estimate_year2_link, dims=['model'], coords={'model':np.linspace(1,34,34)})
estimate_year3_link_array0 = xr.DataArray(data=estimate_year3_link, dims=['model'], coords={'model':np.linspace(1,34,34)})
estimate_year23_link_array0 = xr.DataArray(data=estimate_year23_link, dims=['model'], coords={'model':np.linspace(1,34,34)})
real_year2_link_array0 = xr.DataArray(data=real_year2_link, dims=['model'], coords={'model':np.linspace(1,34,34)})
real_year3_link_array0 = xr.DataArray(data=real_year3_link, dims=['model'], coords={'model':np.linspace(1,34,34)})
real_year23_link_array0 = xr.DataArray(data=real_year23_link, dims=['model'], coords={'model':np.linspace(1,34,34)})
ds0 = xr.Dataset(data_vars=dict(estimate_year2_link=estimate_year2_link_array0, estimate_year3_link=estimate_year3_link_array0, estimate_year23_link=estimate_year23_link_array0, \
                                real_year2_link=real_year2_link_array0, real_year3_link=real_year3_link_array0, real_year23_link=real_year23_link_array0))
ds0.to_netcdf("2_analysis/fig3_CMIP6_sign/historicalpure2x2_real_estimated_link_number_34models.nc")
ds0.close()




