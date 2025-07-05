load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
begin



do ilag=0,36
print(" ilag = " + (ilag))



;;-------------------------------  read SST data  -------------------------------;;
f1 = addfile("0_data/HadISST_1979_2023_2x2.nc","r")
sst00 = f1->sst(:,{-60:63},:)
dim_00 = dimsizes(sst00)
printVarSummary(sst00)

sst0_4D = new((/45,12,dim_00(1),dim_00(2)/), float)
do iyear = 0,44
    sst0_4D(iyear,:,:,:) = sst00(iyear*12:iyear*12+11,:,:)
end do
sst0_clim = dim_avg_n_Wrap(sst0_4D(2:31,:,:,:), 0)
do iyear = 0,44
    sst0_4D(iyear,:,:,:) = sst0_4D(iyear,:,:,:) - sst0_clim
end do
sst0_4D = dtrend_msg_n (fspan(1,45,45), sst0_4D, False,False,0)  ;; remove trends

sst0 = new((/45*12,dim_00(1),dim_00(2)/), float)
do iyear = 0,44
    sst0(iyear*12:iyear*12+11,:,:) = sst0_4D(iyear,:,:,:)
end do
delete([/sst00, sst0_4D/])




;------------------------  Correlation Networks  ------------------------;;
start1 = 0
end1 = 42*12-1
start2 = ilag
end2 = 42*12-1+ilag
cor_Networks0 = escorc_n(sst0(start1:end1,{-60:60},:), sst0(start2:end2,{-60:63},:), 0, 0)
printVarSummary(cor_Networks0)
dim_3 = dimsizes(cor_Networks0)
cor_Networks = cor_Networks0(:,:,0:dim_3(0)-1,:)
delete(cor_Networks0)
printVarSummary(cor_Networks)




;;---------------------  Significat R test (p=0.05)  ----------------------;;
sst = sst0(:,{-60:60},:)
delete(sst0)
sst!0="time"
printVarSummary(sst)
dim_0 = dimsizes(sst)
sst_size1 = new((/dim_0(1),dim_0(2)/), integer)
sst_size2 = new((/dim_0(1),dim_0(2)/), integer)
do ilat = 0,dim_0(1)-1
print("ilat = "+ilat)
do ilon = 0,dim_0(2)-1

if (ismissing(sst(0,ilat,ilon)).eq.False) then
    SST_i1 = sst(start1:end1,ilat,ilon)
    sst_size1(ilat,ilon) = equiv_sample_size(SST_i1, 0.05, 0)
    SST_i2 = sst(start2:end2,ilat,ilon)
    sst_size2(ilat,ilon) = equiv_sample_size(SST_i2, 0.05, 0)
end if


end do
end do




Rt_network = new((/dim_0(1),dim_0(2), dim_0(1),dim_0(2)/), float)
do ilat = 0,dim_0(1)-1
print("ilat = "+ilat)
do ilon = 0,dim_0(2)-1

do jlat = 0,dim_0(1)-1
do jlon = 0,dim_0(2)-1

freedom_value = min((/sst_size1(ilat,ilon), sst_size2(jlat,jlon)/))
Rt_network(ilat,ilon,jlat,jlon) = rtest(cor_Networks(ilat,ilon,jlat,jlon), freedom_value, 0)

end do
end do
end do
end do

printMinMax(Rt_network, 1)

cor_Networks = where(Rt_network.le.0.05, cor_Networks, 0.0)
cor_Networks@_FillValue = 1.0e20
printMinMax(cor_Networks, 1)









;;---------------------- to nc file -----------------------;;
output="1_networks/part1_HadISST/"
name1="Networks2x2_SST_lag"+ilag
system("/bin/rm -f "+output+name1+".nc")
ncdf1=addfile(output+name1+".nc","c")
ncdf1->networks0=cor_Networks


end do


end



