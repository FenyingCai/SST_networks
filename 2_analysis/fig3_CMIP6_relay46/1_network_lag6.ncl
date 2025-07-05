load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
begin



dim_0 = (/60,180, 60,180/)
latt=fspan(-59.0,59.0,60)   ;; 2*2 resolution
lonn=fspan(-179.0,179.0,180)
latt@units="degrees_north"
lonn@units="degrees_east"


lat_weight = new((/dim_0(0),dim_0(1), dim_0(0),dim_0(1)/), float)
do ilat = 0,dim_0(0)-1
do jlat = 0,dim_0(0)-1
lat_weight(ilat,:,jlat,:) = cos(abs((latt(ilat))+abs(latt(jlat))) /2.0 / 180.0 * 3.14159)
end do
end do
printMinMax(lat_weight, 1)






;;---------------------------------------------  lag from 0 to 36 months  --------------------------------------------;;
network6_weight = new((/34,dim_0(0),dim_0(1), dim_0(0),dim_0(1)/), float)
do imodel = 0,33
print("imodel = "+imodel)

f1 = addfile("1_networks/part3_historicalpure/Networks2x2_Historicalpure_SST_lag6.nc","r")
network6 = (f1->networks0(imodel,:,:,:,:))
network6 = where(network6.gt.0.0, 1.0, network6)
network6 = where(network6.lt.0.0, -1.0, network6)
network6 = where(ismissing(network6), 0.0, network6)

network6_weight(imodel,:,:,:,:) = network6*lat_weight
printMinMax(network6_weight, 1)
end do

delete(network6)
delete(lat_weight)




network6_weight!1="lat"
network6_weight!2="lon"
network6_weight!3="lat2"
network6_weight!4="lon2"
network6_weight&lat = latt
network6_weight&lon = lonn
network6_weight&lat2 = latt
network6_weight&lon2 = lonn



;;------------------------------------- to nc file --------------------------------------;;
output="2_analysis/fig3_CMIP6_relay46/"
name1="Network_2x2_historical_lag6"
system("/bin/rm -f "+output+name1+".nc")
ncdf1=addfile(output+name1+".nc","c")
ncdf1->network6_weight=network6_weight


end





