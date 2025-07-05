load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
begin



do ilag=0,36
print(" ilag = " + (ilag))



;;-------------------------------  read SST data  -------------------------------;;
f1 = addfile("0_data/HadISST_1979_2023_2x2.nc","r")
sst00 = f1->sst(:,{-60:60},:)
dim_sst = dimsizes(sst00(0,:,:))
printVarSummary(sst00)

sst0_4D = new((/45,12,dim_sst(0),dim_sst(1)/), float)
do iyear = 0,44
    sst0_4D(iyear,:,:,:) = sst00(iyear*12:iyear*12+11,:,:)
end do
sst0_clim = dim_avg_n_Wrap(sst0_4D(2:31,:,:,:), 0)
do iyear = 0,44
    sst0_4D(iyear,:,:,:) = sst0_4D(iyear,:,:,:) - sst0_clim
end do
sst0_4D = dtrend_msg_n (fspan(1,45,45), sst0_4D, False,False,0)  ;; remove trends

sst = new((/45*12,dim_sst(0),dim_sst(1)/), float)
do iyear = 0,44
    sst(iyear*12:iyear*12+11,:,:) = sst0_4D(iyear,:,:,:)
end do
delete([/sst00, sst0_4D/])



;;-------------------------------  read CPC Tmax data  -------------------------------;;
f2 = addfile("0_data/CPC_precip_monthly_1979_2023.nc","r")
precip_ano0 = f2->precip_ano(:,:,{-80:80},:)
dim_precip = dimsizes(precip_ano0(0,0,:,:))
precip = new((/45*12,dim_precip(0),dim_precip(1)/), float)
do iyear = 0,44
    precip(iyear*12:iyear*12+11,:,:) = precip_ano0(iyear,:,:,:)
end do
delete(precip_ano0)





;------------------------  Correlation Networks  ------------------------;;
start1 = 0
end1 = 42*12-1
start2 = ilag
end2 = 42*12-1+ilag
cor_Networks = escorc_n(sst(start1:end1,{-60:60},:), precip(start2:end2,{-80:80},:), 0, 0)
printVarSummary(cor_Networks)





;;---------------------  Significat R test (p=0.10)  ----------------------;;
sst!0="time"
printVarSummary(sst)
dim_0 = dimsizes(sst)
sst_size1 = new((/dim_sst(0),dim_sst(1)/), integer)
do ilat = 0,dim_sst(0)-1
print("ilat = "+ilat)
do ilon = 0,dim_sst(1)-1
if (ismissing(sst(0,ilat,ilon)).eq.False) then
    SST_i1 = sst(start1:end1,ilat,ilon)
    sst_size1(ilat,ilon) = equiv_sample_size(SST_i1, 0.05, 0)
end if
end do
end do

precip_size2 = new((/dim_precip(0),dim_precip(1)/), integer)
do ilat = 0,dim_precip(0)-1
print("ilat = "+ilat)
do ilon = 0,dim_precip(1)-1
if (ismissing(precip(0,ilat,ilon)).eq.False) then
    precip_i2 = precip(start2:end2,ilat,ilon)
    precip_size2(ilat,ilon) = equiv_sample_size(precip_i2, 0.05, 0)
end if
end do
end do




Rt_network = new((/dim_sst(0),dim_sst(1), dim_precip(0),dim_precip(1)/), float)
do ilat = 0,dim_sst(0)-1
print("ilat = "+ilat)
do ilon = 0,dim_sst(1)-1

do jlat = 0,dim_precip(0)-1
do jlon = 0,dim_precip(1)-1

freedom_value = min((/sst_size1(ilat,ilon), precip_size2(jlat,jlon)/))
Rt_network(ilat,ilon,jlat,jlon) = rtest(cor_Networks(ilat,ilon,jlat,jlon), freedom_value, 0)

end do
end do
end do
end do

printMinMax(Rt_network, 1)

cor_Networks = where(Rt_network.le.0.10, cor_Networks, 0.0)  ;; p=0.10
cor_Networks@_FillValue = 1.0e20
printMinMax(cor_Networks, 1)









;;---------------------- to nc file -----------------------;;
output="1_networks/part12_HadISST_CPCRain90/"
name1="Networks2x2_SST_Rain90_lag"+ilag
system("/bin/rm -f "+output+name1+".nc")
ncdf1=addfile(output+name1+".nc","c")
ncdf1->networks0=cor_Networks


end do


end



