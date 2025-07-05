load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
begin


dim_0 = (/60,180, 60,180/)
latt=fspan(-59.0,59.0,60)   ;; 2*2 resolution
lonn=fspan(-179.0,179.0,180)

latt@units="degrees_north"
lonn@units="degrees_east"

lat_weight = new((/34, 60,180, 60,180/), float)
do ilat = 0,59
do jlat = 0,59
   lat_weight(:,ilat,:,jlat,:) = cos((abs(latt(ilat))+abs(latt(jlat))) / 2 / 180.0 * 3.14159)
end do
end do
printMinMax(lat_weight, 1)




link_number_23models = new((/34,37/), float)
do ilag = 0,36
print(" ilag  =  "+ilag)

;;-----------------------------------  read network matrix  ----------------------------------;;
f1 = addfile("1_networks/part3_historicalpure/Networks2x2_Historicalpure_SST_lag"+ilag+".nc","r")
networks0 = (f1->networks0)
networks0 = abs(networks0)
networks0 = where(networks0.gt.0.0, 1.0, 0.0)
networks0 = where(ismissing(networks0), 0.0, networks0)


link_number_23models(:,ilag) = dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(networks0*lat_weight, 4), 3), 2), 1)

end do
printVarSummary(link_number_23models)
printMinMax(link_number_23models, 1)







;;---------------------- to nc file -----------------------;;
output="2_analysis/fig3_CMIP6_sign/"
name1="Link2x2_Historical_number_lag0_36"
system("/bin/rm -f "+output+name1+".nc")
ncdf1=addfile(output+name1+".nc","c")
ncdf1->link_number_23models=link_number_23models



end


