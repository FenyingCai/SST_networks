# %%
import numpy as np
import xarray as xr



##--##--##--##--    read  network  matrix  1    --##--##--##--##
ds1 = xr.open_dataset("2_analysis/fig2_relay46/Network_2x2_lag_46810.nc")
network = ds1.network6_weight
Size = np.shape(network)
print(" network = ",np.nanmin(network),"  ",np.nanmax(network))

##--##--##--##--    4D -> 2D    --##--##--##--##
network_3D = np.zeros([Size[0],Size[1], Size[2]*Size[3]], dtype=np.float32)
for ilat in range(Size[2]):
    network_3D[:,:,ilat*Size[3]:ilat*Size[3]+Size[3]] = network[:,:,ilat,:]
# del network
print(" network 3D = ",np.nanmin(network_3D),"  ",np.nanmax(network_3D))

network_2D = np.zeros([Size[0]*Size[1], Size[2]*Size[3]], dtype=np.float32)
for ilat in range(Size[0]):
    network_2D[ilat*Size[1]:ilat*Size[1]+Size[1],:] = network_3D[ilat,:,:]
del network_3D
print(" network 2D = ",np.nanmin(network_2D),"  ",np.nanmax(network_2D))





##--##--##--##--    read  network  matrix  2    --##--##--##--##
ds1 = xr.open_dataset("2_analysis/fig4_relay46/Network_CPCTmax_2x2_lag_46810.nc")
networkTmax = ds1.network6_weight
Size = np.shape(networkTmax)
print(" networkTmax = ",np.nanmin(networkTmax),"  ",np.nanmax(networkTmax))

##--##--##--##--    4D -> 2D    --##--##--##--##
networkTmax_3D = np.zeros([Size[0],Size[1], Size[2]*Size[3]], dtype=np.float32)
for ilat in range(Size[2]):
    networkTmax_3D[:,:,ilat*Size[3]:ilat*Size[3]+Size[3]] = networkTmax[:,:,ilat,:]
# del networkTmax
print(" networkTmax 3D = ",np.nanmin(networkTmax_3D),"  ",np.nanmax(networkTmax_3D))

networkTmax_2D = np.zeros([Size[0]*Size[1], Size[2]*Size[3]], dtype=np.float32)
for ilat in range(Size[0]):
    networkTmax_2D[ilat*Size[1]:ilat*Size[1]+Size[1],:] = networkTmax_3D[ilat,:,:]
del networkTmax_3D
print(" networkTmax 2D = ",np.nanmin(networkTmax_2D),"  ",np.nanmax(networkTmax_2D))





##--##--##--##--    Matrix multiply    --##--##--##--##
print(np.shape(network_2D))
print(np.shape(networkTmax_2D))
M20 = np.dot(network_2D, networkTmax_2D)

M20_notfinish = np.dot(network_2D, network_2D)
M30_notfinish = np.dot(M20_notfinish, network_2D)
M40_notfinish = np.dot(M30_notfinish, network_2D)

M30 = np.dot(M20_notfinish, networkTmax_2D)
del M20_notfinish

M40 = np.dot(M30_notfinish, networkTmax_2D)
del M30_notfinish

M50 = np.dot(M40_notfinish, networkTmax_2D)
del M40_notfinish
del network_2D, networkTmax_2D
print(" network (M2) = ",np.nanmin(M20),"  ",np.nanmax(M20))
print(" network (M3) = ",np.nanmin(M30),"  ",np.nanmax(M30))
print(" network (M4) = ",np.nanmin(M40),"  ",np.nanmax(M40))
print(" network (M5) = ",np.nanmin(M50),"  ",np.nanmax(M50))






##--##--##--##--##--##--       2D -> 4D      --##--##--##--##--##--##
M2_3D = np.zeros([Size[0],Size[1], Size[2]*Size[3]], dtype=np.float32)
for ilat in range(Size[0]):
      M2_3D[ilat,:,:] = M20[ilat*Size[1]:ilat*Size[1]+Size[1],:]
del M20
print(" network (M2) 3D = ",np.nanmin(M2_3D),"  ",np.nanmax(M2_3D))
M2_4D = np.zeros([Size[0],Size[1], Size[2],Size[3]], dtype=np.float32)
for ilat in range(Size[2]):
     M2_4D[:,:,ilat,:] = M2_3D[:,:,ilat*Size[1]:ilat*Size[1]+Size[1]]
del M2_3D
print(" network (M2) = ",np.nanmin(M2_4D),"  ",np.nanmax(M2_4D))


M3_3D = np.zeros([Size[0],Size[1], Size[2]*Size[3]], dtype=np.float32)
for ilat in range(Size[0]):
      M3_3D[ilat,:,:] = M30[ilat*Size[1]:ilat*Size[1]+Size[1],:]
del M30
print(" network (M3) 3D = ",np.nanmin(M3_3D),"  ",np.nanmax(M3_3D))
M3_4D = np.zeros([Size[0],Size[1], Size[2],Size[3]], dtype=np.float32)
for ilat in range(Size[2]):
     M3_4D[:,:,ilat,:] = M3_3D[:,:,ilat*Size[1]:ilat*Size[1]+Size[1]]
del M3_3D
print(" network (M3) = ",np.nanmin(M3_4D),"  ",np.nanmax(M3_4D))


M4_3D = np.zeros([Size[0],Size[1], Size[2]*Size[3]], dtype=np.float32)
for ilat in range(Size[0]):
      M4_3D[ilat,:,:] = M40[ilat*Size[1]:ilat*Size[1]+Size[1],:]
del M40
print(" network (M4) 3D = ",np.nanmin(M4_3D),"  ",np.nanmax(M4_3D))
M4_4D = np.zeros([Size[0],Size[1], Size[2],Size[3]], dtype=np.float32)
for ilat in range(Size[2]):
     M4_4D[:,:,ilat,:] = M4_3D[:,:,ilat*Size[1]:ilat*Size[1]+Size[1]]
del M4_3D
print(" network (M4) = ",np.nanmin(M4_4D),"  ",np.nanmax(M4_4D))


M5_3D = np.zeros([Size[0],Size[1], Size[2]*Size[3]], dtype=np.float32)
for ilat in range(Size[0]):
      M5_3D[ilat,:,:] = M50[ilat*Size[1]:ilat*Size[1]+Size[1],:]
del M50
print(" network (M5) 3D = ",np.nanmin(M5_3D),"  ",np.nanmax(M5_3D))
M5_4D = np.zeros([Size[0],Size[1], Size[2],Size[3]], dtype=np.float32)
for ilat in range(Size[2]):
     M5_4D[:,:,ilat,:] = M5_3D[:,:,ilat*Size[1]:ilat*Size[1]+Size[1]]
del M5_3D
print(" network (M5) = ",np.nanmin(M5_4D),"  ",np.nanmax(M5_4D))





##--##--##--##--    to nc file    --##--##--##--##
M1_array0= xr.DataArray(data=networkTmax, dims=['lat1', 'lon1', 'lat2', 'lon2'],
                                coords={'lat1':np.linspace(-59.0,59.0,60),'lon1':np.linspace(-179.0,179.0,180),
                                        'lat2':np.linspace(-79.0,79.0,80),'lon2':np.linspace(-179.0,179.0,180)})
M2_array0= xr.DataArray(data=M2_4D, dims=['lat1', 'lon1', 'lat2', 'lon2'],
                                coords={'lat1':np.linspace(-59.0,59.0,60),'lon1':np.linspace(-179.0,179.0,180),
                                        'lat2':np.linspace(-79.0,79.0,80),'lon2':np.linspace(-179.0,179.0,180)})
M3_array0= xr.DataArray(data=M3_4D, dims=['lat1', 'lon1', 'lat2', 'lon2'],
                                coords={'lat1':np.linspace(-59.0,59.0,60),'lon1':np.linspace(-179.0,179.0,180),
                                        'lat2':np.linspace(-79.0,79.0,80),'lon2':np.linspace(-179.0,179.0,180)})
M4_array0= xr.DataArray(data=M4_4D, dims=['lat1', 'lon1', 'lat2', 'lon2'],
                                coords={'lat1':np.linspace(-59.0,59.0,60),'lon1':np.linspace(-179.0,179.0,180),
                                        'lat2':np.linspace(-79.0,79.0,80),'lon2':np.linspace(-179.0,179.0,180)})
M5_array0= xr.DataArray(data=M5_4D, dims=['lat1', 'lon1', 'lat2', 'lon2'],
                                coords={'lat1':np.linspace(-59.0,59.0,60),'lon1':np.linspace(-179.0,179.0,180),
                                        'lat2':np.linspace(-79.0,79.0,80),'lon2':np.linspace(-179.0,179.0,180)})
ds0 = xr.Dataset(data_vars=dict(M1=M1_array0, M2=M2_array0, M3=M3_array0, M4=M4_array0, M5=M5_array0))
ds0.to_netcdf("2_analysis/fig4_relay46/Network_CPCTmax_Mestimated6.nc")
ds0.close()





