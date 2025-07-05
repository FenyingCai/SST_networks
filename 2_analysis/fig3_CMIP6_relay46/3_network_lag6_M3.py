# %%
import numpy as np
import xarray as xr



network_34models_M3 = np.zeros([34, 60,180, 60,180], dtype=np.float32)

for imodel in range(34):
      print(" imodel ", imodel)

      ##--##--##--##--    read  matrix    --##--##--##--##
      ds1 = xr.open_dataset("2_analysis/fig3_CMIP6_relay46/Network_2x2_historical_lag6.nc")
      network = ds1.network6_weight[imodel,:,:,:,:]
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
      del network_2D
      print(" network (M3) = ",np.nanmin(M30),"  ",np.nanmax(M30))





      ##--##--##--##--##--##--       2D -> 4D      --##--##--##--##--##--##
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


      network_34models_M3[imodel,:,:,:,:] = M3_4D.copy()




##--##--##--##--    to nc file    --##--##--##--##
M3_array0= xr.DataArray(data=network_34models_M3, dims=['model', 'lat1', 'lon1', 'lat2', 'lon2'],
                                coords={'model':np.linspace(1.0,34.0,34),
                                        'lat1':np.linspace(-59.0,59.0,60),'lon1':np.linspace(-179.0,179.0,180),
                                        'lat2':np.linspace(-59.0,59.0,60),'lon2':np.linspace(-179.0,179.0,180)})
ds0 = xr.Dataset(data_vars=dict(M3=M3_array0))
ds0.to_netcdf("2_analysis/fig3_CMIP6_relay46/Network_2x2_historical_lag6_M3.nc")
ds0.close()





