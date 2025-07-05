# %%
import numpy as np
import xarray as xr



##--##--##--##--    read  matrix    --##--##--##--##
ds1 = xr.open_dataset("2_analysis/fig2_relay46/Network_2x2_lag46.nc")
network = ds1.network4_weight
Size = np.shape(network)
print(" network = ",np.nanmin(network),"  ",np.nanmax(network))




##--##--##--##--    4D -> 2D    --##--##--##--##
network_3D = np.zeros([Size[0],Size[1], Size[0]*Size[1]], dtype=np.float32)
for ilat in range(Size[0]):
    network_3D[:,:,ilat*Size[1]:ilat*Size[1]+Size[1]] = network[:,:,ilat,:]
# del network
print(" network 3D = ",np.nanmin(network_3D),"  ",np.nanmax(network_3D))

network_2D = np.zeros([Size[0]*Size[1], Size[0]*Size[1]], dtype=np.float32)
for ilat in range(Size[0]):
    network_2D[ilat*Size[1]:ilat*Size[1]+Size[1],:] = network_3D[ilat,:,:]
del network_3D
print(" network 2D = ",np.nanmin(network_2D),"  ",np.nanmax(network_2D))



##--##--##--##--    Matrix multiply    --##--##--##--##
M20 = np.dot(network_2D, network_2D)
M30 = np.dot(M20, network_2D)
M40 = np.dot(M30, network_2D)
M50 = np.dot(M40, network_2D)
M60 = np.dot(M50, network_2D)
del network_2D
print(" network (M2) = ",np.nanmin(M20),"  ",np.nanmax(M20))
print(" network (M3) = ",np.nanmin(M30),"  ",np.nanmax(M30))
print(" network (M4) = ",np.nanmin(M40),"  ",np.nanmax(M40))
print(" network (M5) = ",np.nanmin(M50),"  ",np.nanmax(M50))
print(" network (M6) = ",np.nanmin(M60),"  ",np.nanmax(M60))




##--##--##--##--##--##--       2D -> 4D      --##--##--##--##--##--##
M2_3D = np.zeros([Size[0],Size[1], Size[0]*Size[1]], dtype=np.float32)
for ilat in range(Size[0]):
      M2_3D[ilat,:,:] = M20[ilat*Size[1]:ilat*Size[1]+Size[1],:]
del M20
print(" network (M2) 3D = ",np.nanmin(M2_3D),"  ",np.nanmax(M2_3D))
M2_4D = np.zeros([Size[0],Size[1], Size[0],Size[1]], dtype=np.float32)
for ilat in range(Size[0]):
     M2_4D[:,:,ilat,:] = M2_3D[:,:,ilat*Size[1]:ilat*Size[1]+Size[1]]
del M2_3D
print(" network (M2) = ",np.nanmin(M2_4D),"  ",np.nanmax(M2_4D))



M3_3D = np.zeros([Size[0],Size[1], Size[0]*Size[1]], dtype=np.float32)
for ilat in range(Size[0]):
      M3_3D[ilat,:,:] = M30[ilat*Size[1]:ilat*Size[1]+Size[1],:]
del M30
print(" network (M3) 3D = ",np.nanmin(M3_3D),"  ",np.nanmax(M3_3D))
M3_4D = np.zeros([Size[0],Size[1], Size[0],Size[1]], dtype=np.float32)
for ilat in range(Size[0]):
     M3_4D[:,:,ilat,:] = M3_3D[:,:,ilat*Size[1]:ilat*Size[1]+Size[1]]
del M3_3D
print(" network (M3) = ",np.nanmin(M3_4D),"  ",np.nanmax(M3_4D))



M4_3D = np.zeros([Size[0],Size[1], Size[0]*Size[1]], dtype=np.float32)
for ilat in range(Size[0]):
      M4_3D[ilat,:,:] = M40[ilat*Size[1]:ilat*Size[1]+Size[1],:]
del M40
print(" network (M4) 3D = ",np.nanmin(M4_3D),"  ",np.nanmax(M4_3D))
M4_4D = np.zeros([Size[0],Size[1], Size[0],Size[1]], dtype=np.float32)
for ilat in range(Size[0]):
     M4_4D[:,:,ilat,:] = M4_3D[:,:,ilat*Size[1]:ilat*Size[1]+Size[1]]
del M4_3D
print(" network (M4) = ",np.nanmin(M4_4D),"  ",np.nanmax(M4_4D))



M5_3D = np.zeros([Size[0],Size[1], Size[0]*Size[1]], dtype=np.float32)
for ilat in range(Size[0]):
      M5_3D[ilat,:,:] = M50[ilat*Size[1]:ilat*Size[1]+Size[1],:]
del M50
print(" network (M5) 3D = ",np.nanmin(M5_3D),"  ",np.nanmax(M5_3D))
M5_4D = np.zeros([Size[0],Size[1], Size[0],Size[1]], dtype=np.float32)
for ilat in range(Size[0]):
     M5_4D[:,:,ilat,:] = M5_3D[:,:,ilat*Size[1]:ilat*Size[1]+Size[1]]
del M5_3D
print(" network (M5) = ",np.nanmin(M5_4D),"  ",np.nanmax(M5_4D))



M6_3D = np.zeros([Size[0],Size[1], Size[0]*Size[1]], dtype=np.float32)
for ilat in range(Size[0]):
      M6_3D[ilat,:,:] = M60[ilat*Size[1]:ilat*Size[1]+Size[1],:]
del M60
print(" network (M6) 3D = ",np.nanmin(M6_3D),"  ",np.nanmax(M6_3D))
M6_4D = np.zeros([Size[0],Size[1], Size[0],Size[1]], dtype=np.float32)
for ilat in range(Size[0]):
     M6_4D[:,:,ilat,:] = M6_3D[:,:,ilat*Size[1]:ilat*Size[1]+Size[1]]
del M6_3D
print(" network (M6) = ",np.nanmin(M6_4D),"  ",np.nanmax(M6_4D))




##--##--##--##--    to nc file    --##--##--##--##
M1_array0= xr.DataArray(data=network, dims=['lat1', 'lon1', 'lat2', 'lon2'],
                                coords={'lat1':np.linspace(-59.0,59.0,60),'lon1':np.linspace(-179.0,179.0,180),
                                        'lat2':np.linspace(-59.0,59.0,60),'lon2':np.linspace(-179.0,179.0,180)})
M2_array0= xr.DataArray(data=M2_4D, dims=['lat1', 'lon1', 'lat2', 'lon2'],
                                coords={'lat1':np.linspace(-59.0,59.0,60),'lon1':np.linspace(-179.0,179.0,180),
                                        'lat2':np.linspace(-59.0,59.0,60),'lon2':np.linspace(-179.0,179.0,180)})
M3_array0= xr.DataArray(data=M3_4D, dims=['lat1', 'lon1', 'lat2', 'lon2'],
                                coords={'lat1':np.linspace(-59.0,59.0,60),'lon1':np.linspace(-179.0,179.0,180),
                                        'lat2':np.linspace(-59.0,59.0,60),'lon2':np.linspace(-179.0,179.0,180)})
M4_array0= xr.DataArray(data=M4_4D, dims=['lat1', 'lon1', 'lat2', 'lon2'],
                                coords={'lat1':np.linspace(-59.0,59.0,60),'lon1':np.linspace(-179.0,179.0,180),
                                        'lat2':np.linspace(-59.0,59.0,60),'lon2':np.linspace(-179.0,179.0,180)})
M5_array0= xr.DataArray(data=M5_4D, dims=['lat1', 'lon1', 'lat2', 'lon2'],
                                coords={'lat1':np.linspace(-59.0,59.0,60),'lon1':np.linspace(-179.0,179.0,180),
                                        'lat2':np.linspace(-59.0,59.0,60),'lon2':np.linspace(-179.0,179.0,180)})
M6_array0= xr.DataArray(data=M6_4D, dims=['lat1', 'lon1', 'lat2', 'lon2'],
                                coords={'lat1':np.linspace(-59.0,59.0,60),'lon1':np.linspace(-179.0,179.0,180),
                                        'lat2':np.linspace(-59.0,59.0,60),'lon2':np.linspace(-179.0,179.0,180)})
ds0 = xr.Dataset(data_vars=dict(M1=M1_array0, M2=M2_array0, M3=M3_array0, M4=M4_array0, M5=M5_array0, M6=M6_array0))
ds0.to_netcdf("2_analysis/fig2_relay46/Network_2x2_Mestimated4.nc")
ds0.close()





