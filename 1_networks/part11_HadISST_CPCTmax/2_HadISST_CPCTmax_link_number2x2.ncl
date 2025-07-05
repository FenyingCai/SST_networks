load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
begin


dim_0 = (/60,180, 60,180/)
latt=fspan(-59.0,59.0,60)   ;; 2*2 resolution
lonn=fspan(-179.0,179.0,180)
latt2=fspan(-79.0,79.0,80)   ;; 2*2 resolution
latt@units="degrees_north"
lonn@units="degrees_east"

lat_weight = new((/60,180, 80,180/), float)
do ilat = 0,59
do jlat = 0,79
lat_weight(ilat,:,jlat,:) = cos((abs(latt(ilat))+abs(latt2(jlat))) /2 / 180.0 * 3.14159)
end do
end do
printMinMax(lat_weight, 1)


f1 = addfile("0_data/topo_2x2.nc","r")
topo0 = f1->topo({-80:80},:)
topo0_4D = conform(lat_weight, topo0, (/2,3/))




link_number = new(37, float)
do ilag = 0,36
print(" ilag  =  "+ilag)

;;-----------------------------------  read concurrence matrix  ----------------------------------;;
f1 = addfile("1_networks/part11_HadISST_CPCTmax/Networks2x2_SST_Tmax_lag"+ilag+".nc","r")
networks0 = (f1->networks0)
networks0 = abs(networks0)
networks0 = where(networks0.gt.0.0, 1.0, 0.0)
networks0 = where(ismissing(networks0), 0.0, networks0)
networks0 = where(topo0_4D.gt.0.0, networks0, 0.0)


link_number(ilag) = dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(networks0*lat_weight, 3), 2), 1), 0)

end do
print(link_number)



print(" reg = "+regCoef(fspan(0,12,13), log10(link_number(0:12))))




;;---------------------- to nc file -----------------------;;
output="1_networks/part11_HadISST_CPCTmax/"
name1="Link2x2_CPCTmax_number_lag0_36"
system("/bin/rm -f "+output+name1+".nc")
ncdf1=addfile(output+name1+".nc","c")
ncdf1->link_number=link_number



end


