load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
begin



dim_0 = (/60,180, 80,180/)
latt=fspan(-59.0,59.0,60)   ;; 2*2 resolution
lonn=fspan(-179.0,179.0,180)
latt2=fspan(-79.0,79.0,80)   ;; 2*2 resolution
lonn2=fspan(1.0,359.0,180)
latt@units="degrees_north"
latt2@units="degrees_north"
lonn@units="degrees_east"
lonn2@units="degrees_east"


lat_weight = new((/60,180, 80,180/), float)
do ilat = 0,59
do jlat = 0,79
lat_weight(ilat,:,jlat,:) = cos((abs(latt(ilat))+abs(latt2(jlat))) /2 / 180.0 * 3.14159)
end do
end do
printMinMax(lat_weight, 1)




;;---------------------------------------------  SST-Tmax networks  --------------------------------------------;;
network = new((/dim_0(0),dim_0(1), dim_0(2),dim_0(3)/), float)
network = 0.0

do ilag = 1,6
print(" ilag  =  "+ilag)

f1 = addfile("1_networks/part11_HadISST_CPCTmax/Networks2x2_SST_Tmax_lag"+ilag+".nc","r")
networks0 = (f1->networks0)
networks0 = abs(networks0)
networks0 = where(ismissing(networks0), 0.0, networks0)

network = network+networks0
end do
delete(networks0)

network!0 = "lat"
network!1 = "lon"
network&lat = latt
network&lon = lonn
network!2 = "lat2"
network!3 = "lon2"
network&lat2 = latt2
network&lon2 = lonn2


network_weight = network*lat_weight
copy_VarCoords(network, network_weight)


;;---------------------------- to nc file ------------------------------;;
output="2_analysis/fig4_degree/"
name1="Network_Tmax_lag1_6"
system("/bin/rm -f "+output+name1+".nc")
ncdf1=addfile(output+name1+".nc","c")
ncdf1->network=network
ncdf1->network_weight=network_weight











;;---------------------------------------------  SST-Rain networks  --------------------------------------------;;
network = new((/dim_0(0),dim_0(1), dim_0(2),dim_0(3)/), float)
network = 0.0

do ilag = 1,6
print(" ilag  =  "+ilag)

f1 = addfile("1_networks/part12_HadISST_CPCRain90/Networks2x2_SST_Rain90_lag"+ilag+".nc","r")
networks0 = (f1->networks0)
networks0 = abs(networks0)
networks0 = where(ismissing(networks0), 0.0, networks0)

network = network+networks0
end do
delete(networks0)

network!0 = "lat"
network!1 = "lon"
network&lat = latt
network&lon = lonn
network!2 = "lat2"
network!3 = "lon2"
network&lat2 = latt2
network&lon2 = lonn2


network_weight = network*lat_weight
copy_VarCoords(network, network_weight)


;;---------------------------- to nc file ------------------------------;;
output="2_analysis/fig4_degree/"
name1="Network_Rain_lag1_6"
system("/bin/rm -f "+output+name1+".nc")
ncdf1=addfile(output+name1+".nc","c")
ncdf1->network=network
ncdf1->network_weight=network_weight









end





