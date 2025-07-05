load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
begin



dim_0 = (/60,180, 60,180/)
latt=fspan(-59.0,59.0,60)   ;; 2*2 resolution
lonn=fspan(-179.0,179.0,180)
latt@units="degrees_north"
lonn@units="degrees_east"

lat_weight = new((/60,180, 60,180/), float)
do ilat = 0,59
do jlat = 0,59
lat_weight(ilat,:,jlat,:) = cos((abs(latt(ilat))+abs(latt(jlat))) /2 / 180.0 * 3.14159)
end do
end do
printMinMax(lat_weight, 1)





link_number = new(37, float)
do ilag = 0,36
print(" ilag  =  "+ilag)

;;-----------------------------------  read concurrence matrix  ----------------------------------;;
f1 = addfile("1_networks/part1_HadISST/Networks2x2_SST_lag"+ilag+".nc","r")
networks0 := (f1->networks0)
networks0 = abs(networks0)
networks0 = where(networks0.gt.0.0, 1.0, 0.0)
networks0 = where(ismissing(networks0), 0.0, networks0)


networks0!0 = "lat1"
networks0!1 = "lon1"
networks0!2 = "lat2"
networks0!3 = "lon2"
;; ------------- exclude two-sided links -------------------;;
networks0_TT = networks0*networks0(lat2|:,lon2|:,lat1|:,lon1|:)
networks0 = where(networks0_TT.eq.0.0, networks0, 0.0)
delete(networks0_TT)



link_number(ilag) = dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(dim_sum_n_Wrap(networks0*lat_weight, 3), 2), 1), 0)
print(" ilag  =  "+ilag+"    ,   "+link_number(ilag))
delete(networks0)
end do
print(link_number)

; print(" reg = "+regCoef(fspan(0,12,13), log10(link_number(0:12))))







;;---------------------- to nc file -----------------------;;
output="2_analysis/fig1_sign/"
name1="Onesided_Link2x2_number_lag0_36"
system("/bin/rm -f "+output+name1+".nc")
ncdf1=addfile(output+name1+".nc","c")
ncdf1->link_number=link_number



end


