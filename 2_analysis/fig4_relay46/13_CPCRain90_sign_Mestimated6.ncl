load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
begin



dim_0 = (/60,180, 80,180/)
latt=fspan(-59.0,59.0,60)   ;; 2*2 resolution
lonn=fspan(-179.0,179.0,180)
latt2=fspan(-79.0,79.0,80)   ;; 2*2 resolution
latt@units="degrees_north"
lonn@units="degrees_east"


lat_weight = new((/dim_0(0),dim_0(1), dim_0(2),dim_0(3)/), float)
do ilat = 0,dim_0(0)-1
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





Mestimated = new((/6,dim_0(0),dim_0(1), dim_0(2),dim_0(3)/), float)
f2 = addfile("2_analysis/fig4_relay46/Network_CPCRain90_Mestimated6.nc","r")
Mestimated(0,:,:,:,:) = (f2->M1)
Mestimated(1,:,:,:,:) = (f2->M2)
Mestimated(2,:,:,:,:) = (f2->M3)
Mestimated(3,:,:,:,:) = (f2->M4)
Mestimated(4,:,:,:,:) = (f2->M5)







;;---------------------------------------------  relay percentage  --------------------------------------------;;
link_number_all = new(37, float)
link_number_relay_correct = new((/6,37/), float)
link_number_relay_wrong = new((/6,37/), float)

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

do m_fold = 0,4
network_correct := where(Mestimated(m_fold,:,:,:,:)*networks0.gt.0.0, networks0, 0.0)
network_correct := abs(network_correct)

network_wrong := where(Mestimated(m_fold,:,:,:,:)*networks0.lt.0.0, networks0, 0.0)
network_wrong := abs(network_wrong)

link_number_relay_correct(m_fold,ilag) = dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(network_correct*lat_weight, 3), 2), 1), 0)
link_number_relay_wrong(m_fold,ilag) = dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(network_wrong*lat_weight, 3), 2), 1), 0)
end do
end do







;;------------------------------ to nc file --------------------------------;;
output="2_analysis/fig4_relay46/"
name1="Mestimated6_CPCRain90_sign_correct_wrong"
system("/bin/rm -f "+output+name1+".nc")
ncdf1=addfile(output+name1+".nc","c")

ncdf1->link_number_all=link_number_all
ncdf1->link_number_relay_correct=link_number_relay_correct
ncdf1->link_number_relay_wrong=link_number_relay_wrong



end





