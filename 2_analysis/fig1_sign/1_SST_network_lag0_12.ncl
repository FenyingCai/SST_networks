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
do ilat = 0,dim_0(0)-1
lat_weight1(ilat,:,:,:) = cos(abs(latt(ilat)) / 180.0 * 3.14159)
end do
do ilat = 0,dim_0(0)-1
lat_weight2(:,:,ilat,:) = cos(abs(latt(ilat)) / 180.0 * 3.14159)
end do
; printMinMax(lat_weight, 1)
; printMinMax(lat_weight1, 1)
; printMinMax(lat_weight2, 1)



;;---------------------------------------------  lag from 0 to 36 months  --------------------------------------------;;
network = new((/dim_0(0),dim_0(1), dim_0(0),dim_0(1)/), float)
network = 0.0

do ilag = 0,12
print(" ilag  =  "+ilag)

f1 = addfile("part1_HadISST/Networks2x2_SST_lag"+ilag+".nc","r")
networks0 = (f1->networks0)
networks0 = where(networks0.gt.0.0, 1.0, networks0)
networks0 = where(networks0.lt.0.0, -1.0, networks0)
networks0 = where(ismissing(networks0), 0.0, networks0)

network = network+networks0
end do
delete(networks0)

printVarSummary(network)
printMinMax(network, 1)
network_weight = network*lat_weight
printMinMax(network_weight, 1)




;;------------------------------ to nc file --------------------------------;;
output="2_analysis/fig1_sign/"
name1="Network_2x2_lag0_12"
system("/bin/rm -f "+output+name1+".nc")
ncdf1=addfile(output+name1+".nc","c")
ncdf1->network=network
ncdf1->network_weight=network_weight
ncdf1->lat_weight=lat_weight
ncdf1->lat_weight1=lat_weight1
ncdf1->lat_weight2=lat_weight2

end





