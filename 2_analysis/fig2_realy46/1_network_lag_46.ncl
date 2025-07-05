load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
begin



dim_0 = (/60,180, 60,180/)
latt=fspan(-59.0,59.0,60)   ;; 2*2 resolution
lonn=fspan(-179.0,179.0,180)
latt@units="degrees_north"
lonn@units="degrees_east"


lat_weight = new((/dim_0(0),dim_0(1), dim_0(0),dim_0(1)/), float)
lat_weight1 = new((/dim_0(0),dim_0(1), dim_0(0),dim_0(1)/), float)
lat_weight2 = new((/dim_0(0),dim_0(1), dim_0(0),dim_0(1)/), float)
do ilat = 0,dim_0(0)-1
do jlat = 0,dim_0(0)-1
lat_weight(ilat,:,jlat,:) = cos(abs((latt(ilat))+abs(latt(jlat))) /2.0 / 180.0 * 3.14159)
end do
end do
printMinMax(lat_weight, 1)






;;--------------------  lag from 0 to 36 months  -------------------;;
f1 = addfile("1_networks/part1_HadISST/Networks2x2_SST_lag6.nc","r")
network6 = (f1->networks0)
network6 = where(network6.gt.0.0, 1.0, network6)
network6 = where(network6.lt.0.0, -1.0, network6)
network6 = where(ismissing(network6), 0.0, network6)
network6_weight = network6*lat_weight
printMinMax(network6_weight, 1)
delete(network6)




f1 = addfile("1_networks/part1_HadISST/Networks2x2_SST_lag4.nc","r")
network4 = (f1->networks0)
network4 = where(network4.gt.0.0, 1.0, network4)
network4 = where(network4.lt.0.0, -1.0, network4)
network4 = where(ismissing(network4), 0.0, network4)
network4_weight = network4*lat_weight
printMinMax(network4_weight, 1)
delete(network4)






network6_weight!0="lat"
network6_weight!1="lon"
network6_weight!2="lat2"
network6_weight!3="lon2"
network6_weight&lat = latt
network6_weight&lon = lonn
network6_weight&lat2 = latt
network6_weight&lon2 = lonn
copy_VarCoords(network6_weight, network4_weight)






;;------------- to nc file ---------------;;
output="2_analysis/fig2_relay46/"
name1="Network_2x2_lag46"
system("/bin/rm -f "+output+name1+".nc")
ncdf1=addfile(output+name1+".nc","c")
ncdf1->network6_weight=network6_weight
ncdf1->network4_weight=network4_weight



end





