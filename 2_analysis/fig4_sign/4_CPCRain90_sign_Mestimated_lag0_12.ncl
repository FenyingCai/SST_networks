load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
begin



dim_0 = (/60,180, 80,180/)
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







;;---------------------------------------------  Real network: lag 0-36  --------------------------------------------;;
network = new((/37,dim_0(0),dim_0(1), dim_0(2),dim_0(3)/), float)
network = 0.0
do ilag = 0,36
print(" ilag  =  "+ilag)

f1 = addfile("1_networks/part12_HadISST_CPCRain90/Networks2x2_SST_Rain90_lag"+ilag+".nc","r")
networks0 = (f1->networks0)
networks0 = where(networks0.gt.0.0, 1.0, networks0)
networks0 = where(networks0.lt.0.0, -1.0, networks0)
networks0 = where(ismissing(networks0), 0.0, networks0)

network(ilag,:,:,:,:) = networks0
end do
delete(networks0)






f2 = addfile("2_analysis/fig4_sign/Network_CPCRain90_lag0_12.nc","r")
networks012 = (f2->network)







;;---------------------------------------------  relay percentage  --------------------------------------------;;
link_number_all = new(37, float)
link_number_relay_correct = new(37, float)
link_number_relay_wrong = new(37, float)
link_number_relay_mid = new(37, float)

do ilag = 0,36
print(" ilag  =  "+ilag)
networks0 = network(ilag,:,:,:,:)
networks_new = abs(networks0)
link_number_all(ilag) = dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(networks_new*lat_weight, 3), 2), 1), 0)
delete(networks_new)
end do


do ilag = 0,36
print(" ilag  =  "+ilag)
networks0 = network(ilag,:,:,:,:)

network_correct := where(networks012(:,:,:,:)*networks0.gt.0.0, networks0, 0.0)
network_correct := abs(network_correct)

network_wrong := where(networks012(:,:,:,:)*networks0.lt.0.0, networks0, 0.0)
network_wrong := abs(network_wrong)

network_mid := where(networks012(:,:,:,:)*networks0.eq.0.0, networks0, 0.0)
network_mid := abs(network_mid)


link_number_relay_correct(ilag) = dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(network_correct*lat_weight, 3), 2), 1), 0)
link_number_relay_wrong(ilag) = dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(network_wrong*lat_weight, 3), 2), 1), 0)
link_number_relay_mid(ilag) = dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(network_mid*lat_weight, 3), 2), 1), 0)
end do



link_number_relay_notsame = link_number_relay_wrong + link_number_relay_mid





;;------------------------------ to nc file --------------------------------;;
output="2_analysis/fig4_sign/"
name1="networks012_CPCRain90_sign_correct_wrong"
system("/bin/rm -f "+output+name1+".nc")
ncdf1=addfile(output+name1+".nc","c")

ncdf1->link_number_all=link_number_all
ncdf1->link_number_relay_correct=link_number_relay_correct
ncdf1->link_number_relay_wrong=link_number_relay_wrong
ncdf1->link_number_relay_mid=link_number_relay_mid
ncdf1->link_number_relay_notsame=link_number_relay_notsame



end





