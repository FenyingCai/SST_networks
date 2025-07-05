load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
begin


dim_0 = (/60,180, 60,180/)
latt=fspan(-59.0,59.0,60)   ;; 2*2 resolution
lonn=fspan(-179.0,179.0,180)
latt2=fspan(-79.0,79.0,80)
latt@units="degrees_north"
lonn@units="degrees_east"

lat_weight = new((/60,180, 80,180/), float)
do ilat = 0,59
do jlat = 0,79
lat_weight(ilat,:,jlat,:) = cos((abs(latt(ilat))+abs(latt2(jlat))) /2 / 180.0 * 3.14159)
end do
end do



;;-----------------------------------  read networks (1-12 months)  -----------------------------------;;
f1 = addfile("2_analysis/fig4_degree/Network_Rain_lag13_36.nc","r")
networks = (f1->network)





;;-----------------------------------  read networks (1-12 months)  -----------------------------------;;
f1 = addfile("0_data/topo_2x2.nc","r")
topo = (f1->topo({-80:80},:))

topo_tropics = topo
topo_tropics = 0.0
topo_tropics({-20:20},:) = 1.0
topo_extra_NH = topo
topo_extra_NH = 0.0
topo_extra_NH({20:80},:) = 1.0
topo_extra_SH = topo
topo_extra_SH = 0.0
topo_extra_SH({-80:-20},:) = 1.0



topo_ocean = (f1->topo({-60:60},:))
topo_tropics_o = topo_ocean
topo_tropics_o = 0.0
topo_tropics_o({-20:20},:) = 1.0
topo_extra_NH_o = topo_ocean
topo_extra_NH_o = 0.0
topo_extra_NH_o({20:60},:) = 1.0
topo_extra_SH_o = topo_ocean
topo_extra_SH_o = 0.0
topo_extra_SH_o({-60:-20},:) = 1.0

topo_tropics_ocean = topo_tropics_o
topo_tropics_ocean(:,0:89) = topo_tropics_o(:,90:179)
topo_tropics_ocean(:,90:179) = topo_tropics_o(:,0:89)
topo_extra_NH_ocean = topo_extra_NH_o
topo_extra_NH_ocean(:,0:89) = topo_extra_NH_o(:,90:179)
topo_extra_NH_ocean(:,90:179) = topo_extra_NH_o(:,0:89)
topo_extra_SH_ocean = topo_extra_SH_o
topo_extra_SH_ocean(:,0:89) = topo_extra_SH_o(:,90:179)
topo_extra_SH_ocean(:,90:179) = topo_extra_SH_o(:,0:89)




to_tropics = conform(networks, topo_tropics, (/2,3/))
to_extra_NH = conform(networks, topo_extra_NH, (/2,3/))
to_extra_SH = conform(networks, topo_extra_SH, (/2,3/))

from_tropics = conform(networks, topo_tropics_ocean, (/0,1/))
from_extra_NH = conform(networks, topo_extra_NH_ocean, (/0,1/))
from_extra_SH = conform(networks, topo_extra_SH_ocean, (/0,1/))







;;-----------------------------------------------------------  flow  --------------------------------------------------------------;;
flow = new((/3,3/), float)
flow(0,0) = dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(networks * lat_weight * from_tropics*to_tropics, 3), 2), 1), 0)
flow(0,1) = dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(networks * lat_weight * from_tropics*to_extra_NH, 3), 2), 1), 0)
flow(0,2) = dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(networks * lat_weight * from_tropics*to_extra_SH, 3), 2), 1), 0)

flow(1,0) = dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(networks * lat_weight * from_extra_NH * to_tropics, 3), 2), 1), 0)
flow(1,1) = dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(networks * lat_weight * from_extra_NH * to_extra_NH, 3), 2), 1), 0)
flow(1,2) = dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(networks * lat_weight * from_extra_NH * to_extra_SH, 3), 2), 1), 0)

flow(2,0) = dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(networks * lat_weight * from_extra_SH * to_tropics, 3), 2), 1), 0)
flow(2,1) = dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(networks * lat_weight * from_extra_SH * to_extra_NH, 3), 2), 1), 0)
flow(2,2) = dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(networks * lat_weight * from_extra_SH * to_extra_SH, 3), 2), 1), 0)

print(flow)


flow_01 = new((/3,3/), float)
flow_01(0,0) = dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(networks * from_tropics*to_tropics, 3), 2), 1), 0)
flow_01(0,1) = dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(networks * from_tropics*to_extra_NH, 3), 2), 1), 0)
flow_01(0,2) = dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(networks * from_tropics*to_extra_SH, 3), 2), 1), 0)

flow_01(1,0) = dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(networks * from_extra_NH * to_tropics, 3), 2), 1), 0)
flow_01(1,1) = dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(networks * from_extra_NH * to_extra_NH, 3), 2), 1), 0)
flow_01(1,2) = dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(networks * from_extra_NH * to_extra_SH, 3), 2), 1), 0)

flow_01(2,0) = dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(networks * from_extra_SH * to_tropics, 3), 2), 1), 0)
flow_01(2,1) = dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(networks * from_extra_SH * to_extra_NH, 3), 2), 1), 0)
flow_01(2,2) = dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(networks * from_extra_SH * to_extra_SH, 3), 2), 1), 0)

flow_weighted = flow / flow_01
print(flow_01)


;;------------------------ to nc file -------------------------;;
output="2_analysis/fig4_degree/"
name1="Flow_hotspots3x3_Rain_tropics"
system("/bin/rm -f "+output+name1+".nc")
ncdf1=addfile(output+name1+".nc","c")
ncdf1->flow=flow
ncdf1->flow_weighted=flow_weighted




end


