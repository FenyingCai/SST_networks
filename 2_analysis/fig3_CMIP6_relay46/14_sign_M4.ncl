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
lat_weight(ilat,:,jlat,:) = cos((abs(latt(ilat))+abs(latt(jlat))) /2 / 180.0 * 3.14159)
end do
end do
printMinMax(lat_weight, 1)



;;---------------------------------------------  relay percentage  --------------------------------------------;;
link_number_all = new((/34,37/), float)
link_number_relay_same = new((/34,37/), float)
link_number_relay_wrong = new((/34,37/), float)


do imodel = 0,33
print(" imodel  =  "+imodel)

f2 = addfile("2_analysis/fig3_CMIP6_relay46/Network_2x2_historical_lag6_M4.nc","r")
M4 := (f2->M4(imodel,:,:,:,:))



do ilag = 0,36
print(" ilag  =  "+ilag)

f1 = addfile("1_networks/part3_historicalpure/Networks2x2_Historicalpure_SST_lag"+ilag+".nc","r")
networks0 = (f1->networks0(imodel,:,:,:,:))
networks0 = where(networks0.gt.0.0, 1.0, networks0)
networks0 = where(networks0.lt.0.0, -1.0, networks0)
networks0 = where(ismissing(networks0), 0.0, networks0)

networks_new = abs(networks0)
link_number_all(imodel,ilag) = dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(networks_new*lat_weight, 3), 2), 1), 0)
delete(networks_new)


network_same := where(M4*networks0.gt.0.0, networks0, 0.0)
network_same := abs(network_same)

network_wrong := where(M4*networks0.lt.0.0, networks0, 0.0)
network_wrong := abs(network_wrong)

link_number_relay_same(imodel,ilag) = dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(network_same*lat_weight, 3), 2), 1), 0)
link_number_relay_wrong(imodel,ilag) = dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(network_wrong*lat_weight, 3), 2), 1), 0)

end do
end do





;;-------------------------------------  to nc file  -------------------------------------;;
output="2_analysis/fig3_CMIP6_relay46/"
name1="historical_2x2_sign_same_wrong_M4"
system("/bin/rm -f "+output+name1+".nc")
ncdf1=addfile(output+name1+".nc","c")

ncdf1->link_number_all=link_number_all
ncdf1->link_number_relay_same=link_number_relay_same
ncdf1->link_number_relay_wrong=link_number_relay_wrong



end





