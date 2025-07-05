load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
begin



;;------------------  read lat, lon  -------------------;;
dim_0 = (/60,180, 60,180/)
latt=fspan(-59.0,59.0,60)   ;; 2*2 resolution
lonn=fspan(-179.0,179.0,180)
latt@units="degrees_north"
lonn@units="degrees_east"








;;---------------------------------------------  links at different lags  --------------------------------------------;;
f1 = addfile("1_networks/part2_HadISST/Networks2x2_SST_lag6.nc","r")
networks6 = (f1->networks0)
networks6 = where(networks6.gt.0.0, 1.0, networks6)
networks6 = where(networks6.lt.0.0, -1.0, networks6)
networks6 = where(ismissing(networks6), 0.0, networks6)

f1 = addfile("1_networks/part2_HadISST/Networks2x2_SST_lag12.nc","r")
networks12 = (f1->networks0)
networks12 = where(networks12.gt.0.0, 1.0, networks12)
networks12 = where(networks12.lt.0.0, -1.0, networks12)
networks12 = where(ismissing(networks12), 0.0, networks12)

networks6!0 = "lat"
networks6!1 = "lon"
networks6&lat = latt
networks6&lon = lonn
networks6!2 = "lat2"
networks6!3 = "lon2"
networks6&lat2 = latt
networks6&lon2 = lonn
copy_VarCoords(networks6, networks12)









;;--------------------------------------------------------------------------------------------------;;
f1 = addfile("0_data/topo_2x2.nc","r")
topo = (f1->topo)
key_region1 = topo
key_region2 = topo

;;------------------------------------------------;;

WNP_lat1 = 2
WNP_lat2 = 20
WNP_lon1 = 120
WNP_lon2 = 145

IO_lat1 = -15
IO_lat2 = 15
IO_lon1 = 50
IO_lon2 = 75

WEA_lat1 = 10
WEA_lat2 = 22
WEA_lon1 = -85
WEA_lon2 = -55



key_region1({IO_lat1:IO_lat2},{IO_lon1:IO_lon2}) = 1.0
key_region1({WNP_lat1:WNP_lat2},{WNP_lon1:WNP_lon2}) = 1.0
key_region1({WEA_lat1:WEA_lat2},{WEA_lon1+360:WEA_lon2+360}) = 1.0
key_region1 = where(topo.le.0.0, key_region1, -1.0)








;;-------------------------------  Lagged correaltion  -------------------------------;;
f1 = addfile("0_data/HadISST_1979_2023_2x2.nc","r")
sst00 = f1->sst(:,{-60:60},:)
lat = sst00&lat
lon = sst00&lon
dim_00 = dimsizes(sst00)
printVarSummary(sst00)

sst0_4D = new((/45,12,dim_00(1),dim_00(2)/), float)
do iyear = 0,44
    sst0_4D(iyear,:,:,:) = sst00(iyear*12:iyear*12+11,:,:)
end do
sst0_clim = dim_avg_n_Wrap(sst0_4D(2:31,:,:,:), 0)
sst0_std = dim_stddev_n_Wrap(sst0_4D(2:31,:,:,:), 0)
sst0_std = where(sst0_std.gt.0.0, sst0_std, 1.0e20)
sst0_std@_FillValue = 1.0e20
do iyear = 0,44
    sst0_4D(iyear,:,:,:) = (sst0_4D(iyear,:,:,:) - sst0_clim); / sst0_std
end do
sst0_4D = dtrend_msg_n (fspan(1,45,45), sst0_4D, False,False,0)   ;; remove trends

sst0 = new((/45*12,dim_00(1),dim_00(2)/), float)   ;; to 1D
do iyear = 0,44
    sst0(iyear*12:iyear*12+11,:,:) = sst0_4D(iyear,:,:,:)
end do
delete([/sst00, sst0_4D/])





WNP_lat1_new = 2
WNP_lat2_new = 20
WNP_lon1_new = 120
WNP_lon2_new = 145

IO_lat1_new = -15
IO_lat2_new = 15
IO_lon1_new = 50
IO_lon2_new = 75

WEA_lat1_new = 10
WEA_lat2_new = 22
WEA_lon1_new = -85
WEA_lon2_new = -55

SST_WNP = dim_avg_n_Wrap(dim_avg_n_Wrap(sst0(:,{WNP_lat1_new:WNP_lat2_new},{WNP_lon1_new:WNP_lon2_new}), 2), 1)
SST_IO = dim_avg_n_Wrap(dim_avg_n_Wrap(sst0(:,{IO_lat1_new:IO_lat2_new},{IO_lon1_new:IO_lon2_new}), 2), 1)
SST_WEA = dim_avg_n_Wrap(dim_avg_n_Wrap(sst0(:,{WEA_lat1_new:WEA_lat2_new},{WEA_lon1_new:WEA_lon2_new}), 2), 1)


cor_WNP_IO = new((/37/), float)
cor_WNP_IO_t = new((/37/), float)
cor_IO_WEA = new((/37/), float)
cor_IO_WEA_t = new((/37/), float)
cor_WNP_WEA = new((/37/), float)
cor_WNP_WEA_t = new((/37/), float)


do ilag = 0,36
print(" ilag = "+ilag)

cor_WNP_IO(ilag) = escorc_n(SST_WNP(0:12*42-1), SST_IO(0+ilag:12*42-1+ilag), 0, 0)
SST_WNP_sample := equiv_sample_size(SST_WNP(0:12*42-1), 0.05, 0)
SST_IO_sample := equiv_sample_size(SST_IO(0+ilag:12*42-1+ilag), 0.05, 0)
cor_WNP_IO_t(ilag) = rtest(cor_WNP_IO(ilag), min((/SST_WNP_sample,SST_IO_sample/)), 0)

cor_IO_WEA(ilag) = escorc_n(SST_IO(0:12*42-1), SST_WEA(0+ilag:12*42-1+ilag), 0, 0)
sample1 := equiv_sample_size(SST_IO(0:12*42-1), 0.05, 0)
sample2 := equiv_sample_size(SST_WEA(0+ilag:12*42-1+ilag), 0.05, 0)
cor_IO_WEA_t(ilag) = rtest(cor_IO_WEA(ilag), min((/sample1,sample2/)), 0)

cor_WNP_WEA(ilag) = escorc_n(SST_WNP(0:12*42-1), SST_WEA(0+ilag:12*42-1+ilag), 0, 0)
sample1 := equiv_sample_size(SST_WNP(0:12*42-1), 0.05, 0)
sample2 := equiv_sample_size(SST_WEA(0+ilag:12*42-1+ilag), 0.05, 0)
cor_WNP_WEA_t(ilag) = rtest(cor_WNP_WEA(ilag), min((/sample1,sample2/)), 0)

end do

cor_WNP_IO_sign95 = where(cor_WNP_IO_t.le.0.05, cor_WNP_IO, 0.0)
cor_IO_WEA_sign95 = where(cor_IO_WEA_t.le.0.05, cor_IO_WEA, 0.0)
cor_WNP_WEA_sign95 = where(cor_WNP_WEA_t.le.0.05, cor_WNP_WEA, 0.0)


print(" ")
print(cor_WNP_IO_sign95)
print(" ")
print(cor_IO_WEA_sign95)
print(" ")
print(cor_WNP_WEA_sign95)
















;;---------------------------------------------------  plots  ---------------------------------------------------
res = True
res@tmBorderThicknessF = 1.4
res@gsnDraw      =  False
res@gsnFrame     =  False
res@gsnAddCyclic =  True
res@gsnRightString       = ""
res@gsnLeftString        = ""
res@gsnLeftStringFontHeightF   = 0.04
res@gsnRightStringFontHeightF  = 0.04
res@gsnCenterStringFontHeightF  = 0.04
res@tmXTLabelsOn  = False
res@tmYRLabelsOn  = False
vcres = res
cnres = res
cnres@gsnAddCyclic =  False
res@tmXTOn        = False
res@tmYROn        = False
res@tmXBOn        = True
res@tmYLOn        = True
; res@tmXBLabelFontHeightF = 0.038
; res@tmYLLabelFontHeightF = 0.038
res@tmXBTickSpacingF = 45.0
res@tmYLTickSpacingF = 20.0





res@mpMinLatF = -60
res@mpMaxLatF = 60
res@mpMinLonF = 0
res@mpMaxLonF = 360
res@mpCenterLonF = 180
res@tmXBMode = "Explicit"
res@tmXBValues = (/-135,-90,-45,0,45,90,135,180,225,270,315,360/)
res@tmXBLabels = (/"135W","90W","45W","0","45E","90E","135E","180","135W","90W","45W","0"/)
res@tmYLMode = "Explicit"
res@tmYLValues = (/-90,-60,-30,0,30,60,90/)
res@tmYLLabels = (/"90S","60S","30S","0","30N","60N","90N"/)
res@tmXBMinorOn = True
res@tmXBMinorValues = fspan(-360, 360+15, 49+1)
res@tmXBLabelsOn = True
res@tmXBLabelFontHeightF = 0.012
res@tmYLLabelsOn = True
res@tmYLLabelFontHeightF = 0.012

res@tmYLMajorThicknessF = 1.5
res@tmXBMajorThicknessF = 1.5
res@tmYLMinorThicknessF = 1.1
res@tmXBMinorThicknessF = 1.1
res@tmYLMajorLengthF = 0.014
res@tmXBMajorLengthF = 0.014
res@tmYLMinorLengthF = 0.012
res@tmXBMinorLengthF = 0.012
res@tmXBLabelDeltaF = -0.35
res@tmYLLabelDeltaF = -0.35
res@tmXBMajorOutwardLengthF = 0.014
res@tmYLMajorOutwardLengthF = 0.014
res@tmXBMinorOutwardLengthF = 0.012
res@tmYLMinorOutwardLengthF = 0.012





res@mpFillOn                    = False
res@mpOutlineOn                 = True
res@mpGeophysicalLineThicknessF = 0.9
res@mpGeophysicalLineColor      = "black";"gray20"
res@cnFillOn             = True
res@cnLinesOn            = False
;res@cnLineThicknessF     = 6.0
;res@cnLineColor          = "red"
res@cnLineLabelsOn       = False
res@lbLabelBarOn         = False
res@lbOrientation        = "Vertical"
res@pmLabelBarWidthF     = 0.045
res@pmLabelBarHeightF    = 0.14
res@pmLabelBarOrthogonalPosF = 0.012
res@lbLabelFontHeightF  = 0.015
res@cnMissingValFillColor = -1
res@cnLevelSelectionMode = "ExplicitLevels"              
res@cnFillPalette        = "precip2_17lev" 



;vcres@mpFillDrawOrder         = "PostDraw"
vcres@vcRefAnnoOrthogonalPosF = -0.255
;vcres@vcRefAnnoSide           = "TopRight"
vcres@vcGlyphStyle            = "LineArrow"
vcres@vcRefAnnoArrowLineColor   = "black"         ; change ref vector color
vcres@vcRefAnnoArrowUseVecColor = False           ; don't use vec color for ref
vcres@vcMinDistanceF          = 0.03             ; thin out vectors
vcres@vcLineArrowColor        = "black"           ; change vector color
vcres@vcRefAnnoOn             = True
vcres@vcRefLengthF            = 0.022            ; ref vec length
vcres@vcRefAnnoFontHeightF    = 0.014   ;参考箭头字体大小
vcres@vcRefAnnoString1On      = True
vcres@vcRefAnnoString2On      = False
vcres@vcLineArrowThicknessF   = 3.0            ; make vectors larger
vcres@vcVectorDrawOrder = "PostDraw"
vcres@vcRefAnnoOrthogonalPosF = -0.26 ;;正向下
vcres@vcRefAnnoParallelPosF   = 1.0  ;;正向右
vcres@vcRefAnnoString1        = "1"
vcres@vcRefMagnitudeF         = 1



sres = True
sres@cnLineLabelsOn   = False
sres@cnConstFLabelOn  = False
sres@cnInfoLabelOn    = False 
sres@gsnDraw          = False                   
sres@gsnFrame         = False
sres@gsnLeftString    = ""
sres@gsnRightString   = ""
sres@lbLabelBarOn = False
sres@cnFillOn  = False
sres@cnLinesOn = True
sres2 = sres
sres3 = sres
sres@cnLineColor = "red"
sres@cnLevelSelectionMode = "ExplicitLevels"
; sres@cnLevels         = (/2700.0/)
; sres@cnFillColors     = (/"red","gray20"/)
sres@cnMissingValFillColor= -1
sres@cnLineThicknessF = 6.0
sres@cnLineDashPattern = 14


sres2@cnLevelSelectionMode = "ExplicitLevels"
sres2@cnLevels         = (/2/)
sres2@cnLineColor = "blue"
sres2@cnLineDashPattern = 14
sres2@cnLineThicknessF = 4




cnres@tmBorderThicknessF = 2.2
cnres@gsnDraw        = False
cnres@gsnFrame       = False
cnres@lgPerimOn = False
cnres@lgLabelFontHeightF = 0.018
cnres@vpWidthF  = 0.42
cnres@vpHeightF = 0.42
cnres@vpXF      = 0.06
cnres@vpYF      = 0.5

cnres@trXMinF = -0.5
cnres@trXMaxF = 18.5
cnres@trYMinF = 0.0
cnres@trYMaxF = 90.0


cnres@tmXTOn        = False
cnres@tmYROn        = False
cnres@tmXBOn        = True
cnres@tmYLOn        = True
cnres@tmXBLabelFontHeightF = 0.014
cnres@tmYLLabelFontHeightF = 0.014
cnres@tmYRLabelFontHeightF = 0.014
cnres@gsnLeftStringFontHeightF   = 0.022
cnres@gsnRightStringFontHeightF  = 0.022
cnres@gsnCenterStringFontHeightF  = 0.022
cnres@tmYRLabelsOn   = False
; cnres@tmYROn         = True

cnres@tmYLMode := "Explicit"
cnres@tmYLValues := fspan(0.0, 80.0, 5)
cnres@tmYLLabels := (/"0","20N","40N","60N","80N"/)
cnres@tmYLMinorOn := True
cnres@tmYLMinorValues := fspan(0.0, 80.0, 9)

cnres@tmXBMode := "Explicit"
cnres@tmXBValues := fspan(0, 18.0, 19)
cnres@tmXBLabels := (/"M","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18"/)
cnres@tmXBMinorOn := False

cnres@cnFillOn       = True
cnres@cnLinesOn      = False
cnres@cnLineLabelsOn = False
cnres@lbLabelBarOn   = False
cnres@cnInfoLabelOn  = False
cnres@cnLevelSelectionMode = "ExplicitLevels"

cnres@tmYLMajorThicknessF = 2.5
cnres@tmXBMajorThicknessF = 2.5
cnres@tmYLMinorThicknessF = 1.5
cnres@tmXBMinorThicknessF = 1.5
cnres@tmYLMajorLengthF = 0.012
cnres@tmXBMajorLengthF = 0.012
cnres@tmYLMinorLengthF = 0.009
cnres@tmXBMinorLengthF = 0.009
cnres@tmXBLabelDeltaF = -0.4
cnres@tmYLLabelDeltaF = -0.4
cnres@tmXBMajorOutwardLengthF = 0.012
cnres@tmYLMajorOutwardLengthF = 0.012
cnres@tmXBMinorOutwardLengthF = 0.009
cnres@tmYLMinorOutwardLengthF = 0.009







lires = True
lires@tmBorderThicknessF = 1.1
lires@gsnDraw        = False
lires@gsnFrame       = False
lires@tiXAxisOn      = False
lires@tiYAxisOn      = False
lires@tmXTLabelsOn   = False
lires@tmXTOn         = False

lires@trYMinF        = 0.0
lires@trYMaxF        = 0.5
lires@trXMinF        = 0.5
lires@trXMaxF        = 12.5

lires@tmXBLabelFontHeightF = 0.072
lires@tmYLLabelFontHeightF = 0.072
lires@tmYRLabelFontHeightF = 0.072
lires@gsnLeftStringFontHeightF   = 0.025
lires@gsnRightStringFontHeightF  = 0.025
lires@tmYRLabelsOn   = False
lires@tmYROn         = False

lires@tmYLOn         = True
lires@tmYLLabelsOn   = True
lires@tmYLMode = "Explicit"
lires@tmXBMode = "Explicit"

lires@xyMarkLineModes = (/"Lines","Lines","Lines"/)

lires@lgPerimOn = False
lires@lgLabelFontHeightF = 0.018
lires@vpWidthF  = 0.9
lires@vpHeightF = 0.5
lires@vpXF      = 0.06
lires@vpYF      = 0.75

lires@gsnLeftString  = ""
lires@gsnRightString = ""
lires@xyMarkLineModes = (/"Lines","Lines","Lines"/)
lires@xyLineThicknesses  = (/3.5,8.0,8.0/)
lires@xyLineColors       = (/"red","blue","orange"/)
lires@xyDashPatterns     = (/0,4,2/)

lires@tmYLMajorThicknessF = 1.5
lires@tmYRMajorThicknessF = 1.5
lires@tmXBMajorThicknessF = 1.5
lires@tmYLMinorThicknessF = 1.0
lires@tmYRMinorThicknessF = 1.0
lires@tmXBMinorThicknessF = 1.0
lires@tmYLMajorLengthF = 0.02
lires@tmXBMajorLengthF = 0.02
lires@tmYLMinorLengthF = 0.016
lires@tmXBMinorLengthF = 0.016
lires@tmXBLabelDeltaF = -0.35
lires@tmYLLabelDeltaF = -0.35
lires@tmXBMajorOutwardLengthF = 0.02
lires@tmYLMajorOutwardLengthF = 0.02
lires@tmXBMinorOutwardLengthF = 0.016
lires@tmYLMinorOutwardLengthF = 0.016






;;*************************打点***************************;;
ores                 = True   
ores@gsnAddCyclic    =  True         
ores@gsnDraw         = False               
ores@gsnFrame        = False                 
ores@cnLineLabelsOn  = False               
ores@cnLinesOn       = False                 
ores@cnInfoLabelOn   = False                                                                                                        
ores@cnFillOn        = True                               
ores@lbLabelBarOn    = False                                                                                                            
ores@cnLevelSelectionMode = "ExplicitLevels"                                                                                                                                                                                
; ores@cnLevels        = (/14.007/)   ;;90%
; ores@cnLevels        = (/-0.312,0.312/)   ;;95%
ores@cnMonoFillPattern    = False            ; want multiple patterns                                                               
; ores@cnFillPatterns       = (/-1,11/)     ; the patterns                                                                         
ores@cnMonoFillScale      = False            ; want different densities                                                             
; ores@cnFillScales         = (/0.6,0.6/)    ; change densities                                                                         
ores@cnMonoFillColor      =True                                                                                                     
ores@cnFillDotSizeF       = 0.003    
ores@cnFillDrawOrder ="postDraw"
ores@cnFillColor = "gray30"

ores@cnLevels        = (/0.05/)   ;;90%
; ores@cnLevels        = (/-0.312,0.312/)   ;;95%
ores@cnMonoFillPattern    = False            ; want multiple patterns                                                               
ores@cnFillPatterns       = (/5,-1/)     ; the patterns                                                                         
ores@cnMonoFillScale      = False            ; want different densities                                                             
ores@cnFillScales         = (/0.8,0.8/)    ; change densities   




;**********************plot**************************
plot = new(14, "graphic")
plot2 = new(20, "graphic")
plot3 = new(20, "graphic")
plot4 = new(20, "graphic")
vector = new(14, "graphic")
contour1 = new(14, "graphic")
contour2 = new(24, "graphic")
contour3 = new(14, "graphic")
contour4 = new(14, "graphic")
contour5 = new(14, "graphic")
contour6 = new(14, "graphic")
contour7 = new(14, "graphic")
contour8 = new(14, "graphic")
topooo = new(14, "graphic") 




;;-------------------------------------------------------------------------------------------
pltType = "eps"
pltName = "3_extended_data_fig/FigureS2_WNP_WNA"
wks = gsn_open_wks(pltType, pltName)



res@tmXBLabelsOn = True
res@cnMissingValFillColor = -1
res@cnFillMode = "CellFill"
res@gsnLeftString = ""


res@tmYLMajorLengthF = 0.016
res@tmXBMajorLengthF = 0.016
res@tmYLMinorLengthF = 0.0115
res@tmXBMinorLengthF = 0.0115
res@tmXBMajorOutwardLengthF = 0.016
res@tmYLMajorOutwardLengthF = 0.016
res@tmXBMinorOutwardLengthF = 0.0115
res@tmYLMinorOutwardLengthF = 0.0115

res@mpMinLatF = -70
res@mpMaxLatF = 85
res@mpMinLonF = -120
res@mpMaxLonF = 180
res@mpCenterLonF = 0
res@tmXBMode = "Explicit"
res@tmXBValues := (/-180,-120,-60,0,60,120,180,240,300,360,420/)
res@tmXBLabels := (/"180","120W","60W","0","60E","120E","180","120W","60W","0","60E"/)
res@tmYLMode = "Explicit"
res@tmYLValues := (/-60,-30,0,30,60/)
res@tmYLLabels := (/"60S","20S","0","30N","60N"/)
res@tmYLMinorValues = fspan(-90, 90, 13)


res@tmYLLabelsOn = True
res@tmXBLabelFontHeightF = 0.018
res@tmYLLabelFontHeightF = 0.018
; plot(0) = gsn_csm_map_ce(wks, res)
res@cnFillPalette = "GMT_gray"
res@cnLevels     := (/0.5/)
res@cnFillColors := (/-1,8/)
plot(0) = gsn_csm_contour_map_ce(wks, key_region1, res)





;;----------------------------- text -----------------------------;;
txres               = True
txres@txFont = "helvetica"
txres@txFontHeightF = 0.019
txres@txPerimOn = True
txres@txBackgroundFillColor = "white"


txres@txPerimColor = "violetred2"
txres@txFontColor = "violetred2"
eum1 = gsn_add_text(wks,plot(0), "lag = 6 months", -5, -40, txres)

txres@txPerimColor = "springgreen3"
txres@txFontColor = "springgreen3"
eum3 = gsn_add_text(wks,plot(0), "lag = 6 months", 105, -40, txres)
txres@txPerimColor = "dodgerblue"
txres@txFontColor = "dodgerblue"
eum4 = gsn_add_text(wks,plot(0), "lag = 12 months", 33, 68, txres)










;;-----------------------------------  Network Links (1.): WNP->IO  ---------------------------------;;
network6_new := networks6({WNP_lat1:WNP_lat2},{WNP_lon1:WNP_lon2}, {IO_lat1:IO_lat2},{IO_lon1:IO_lon2})
; printVarSummary(network6_new)
; printMinMax(network6_new, 1)
dim_new = dimsizes(network6_new)
linelink1 = new(10000, "graphic")
k = 0

do ilat = 0, dim_new(0)-1
print("ilat = "+ilat)
do ilon = 0, dim_new(1)-1
do jlat = 0, dim_new(2)-1,2
do jlon = 0, dim_new(3)-1,2


if (network6_new(ilat,ilon,jlat,jlon).lt.0.0) then
lat1 := WNP_lat1 + 1.0 + ilat*2
lon1 := WNP_lon1 + 1.0 + ilon*2
lat2 := IO_lat1 + 1.0 + jlat*2
lon2 := IO_lon1 + 1.0 + jlon*2

npts = 100
lon := new(npts, "float")
lat := new(npts, "float")
do i=0,npts-1
  t = i / (npts - 1.0)
  lon(i) = lon1 + t * (lon2 - lon1)
  lat(i) = lat1 + t * (lat2 - lat1)
  lat(i) = lat(i) - 26.0 * sin(3.14159 * t)
end do

lnres = True
lnres@gsLineColor="springgreen3"
lnres@gsLineDashPattern=0
lnres@gsLineThicknessF=0.001
linelink1(k) = gsn_add_polyline(wks,plot(0), lon,lat, lnres)
k = k+1
end if

end do
end do
end do
end do




;;-----------------------------------  Network Links (2.): IO->WEA  ---------------------------------;;
network6_new := networks6({IO_lat1:IO_lat2},{IO_lon1:IO_lon2}, {WEA_lat1:WEA_lat2},{WEA_lon1:WEA_lon2})
dim_new = dimsizes(network6_new)
linelink2 = new(10000, "graphic")
k = 0

do ilat = 0, dim_new(0)-1,2
print("ilat = "+ilat)
do ilon = 0, dim_new(1)-1,2
do jlat = 0, dim_new(2)-1,2
do jlon = 0, dim_new(3)-1,2


if (network6_new(ilat,ilon,jlat,jlon).gt.0.0) then
lat1 := IO_lat1 + 1.0 + ilat*2
lon1 := IO_lon1 + 1.0 + ilon*2
lat2 := WEA_lat1 + 1.0 + jlat*2
lon2 := WEA_lon1 + 1.0 + jlon*2

npts = 100
lon := new(npts, "float")
lat := new(npts, "float")
do i=0,npts-1
  t = i / (npts - 1.0)
  lon(i) = lon1 + t * (lon2 - lon1)
  lat(i) = lat1 + t * (lat2 - lat1)
  lat(i) = lat(i) - 30.0 * sin(3.14159 * t)
end do

lnres = True
lnres@gsLineColor="violetred2"
lnres@gsLineDashPattern=0
lnres@gsLineThicknessF=0.001
linelink2(k) = gsn_add_polyline(wks,plot(0), lon,lat, lnres)
k = k+1
end if

end do
end do
end do
end do




;;--------------------------------------  Network Links (3.): WNP->WEA  ------------------------------------;;
network12_new := networks12({WNP_lat1:WNP_lat2},{WNP_lon1:WNP_lon2}, {WEA_lat1:WEA_lat2},{WEA_lon1:WEA_lon2})
dim_new = dimsizes(network12_new)
linelink3 = new(10000, "graphic")
k = 0

do ilat = 0, dim_new(0)-1,2
print("ilat = "+ilat)
do ilon = 0, dim_new(1)-1,2
do jlat = 0, dim_new(2)-1,2
do jlon = 0, dim_new(3)-1,2


if (network12_new(ilat,ilon,jlat,jlon).lt.0.0) then
lat1 := WNP_lat1 + 1.0 + ilat*2
lon1 := WNP_lon1 + 1.0 + ilon*2
lat2 := WEA_lat1 + 1.0 + jlat*2
lon2 := WEA_lon1 + 1.0 + jlon*2

npts = 100
lon := new(npts, "float")
lat := new(npts, "float")
do i=0,npts-1
  t = i / (npts - 1.0)
  lon(i) = lon1 + t * (lon2 - lon1)
  lat(i) = lat1 + t * (lat2 - lat1)
  lat(i) = lat(i) + 40.0 * sin(3.14159 * t)
end do

lnres@gsLineColor="dodgerblue"
lnres@gsLineDashPattern=0
lnres@gsLineThicknessF=0.001
linelink3(k) = gsn_add_polyline(wks,plot(0), lon,lat, lnres)
k = k+1
end if

end do
end do
end do
end do













;;-----------------------------------------------------------------;;
lires@vpWidthF  = 0.93
lires@vpHeightF = 0.48

lires@tmXBLabelFontHeightF = 0.024
lires@tmYLLabelFontHeightF = 0.024
lires@trXMinF := 0.0
lires@trXMaxF := 24.0
lires@trYMinF := -1.0
lires@trYMaxF := 1.0
lires@tmYLMode = "Explicit"
lires@tmYLValues := fspan(-1.0, 1.0, 5)
lires@tmYLLabels := fspan(-1.0, 1.0, 5)
lires@tmYLMinorValues := fspan(-1.0, 1.0, 9)
lires@tmXBMode = "Explicit"
lires@tmXBValues := fspan(0, 36, 7)
lires@tmXBLabels := fspan(0, 36, 7)
lires@tmXBMinorValues := fspan(0, 36, 19)
lires2 = lires


lires@gsnCenterStringFontHeightF = 0.025
lires@pmLegendDisplayMode  = "Always"
lires@xyExplicitLegendLabels = (/"   WNP -> WIO",""/)
lires@pmLegendHeightF = 0.03
lires@pmLegendWidthF  = 0.16
lires@lgLabelFontHeightF = 0.024
lires@pmLegendOrthogonalPosF = -0.21  ;正为向上
lires@pmLegendParallelPosF = 0.67
lires@pmLegendSide    = "Top"

lires@xyLineThicknesses  := (/1.0,8.0,8.0/)
lires@xyLineColors       := (/"springgreen3","violetred2","violetred2"/)
lires@xyDashPatterns     := (/0,4,2/)
lires@gsnCenterString = " "
plot(1) = gsn_csm_xy(wks, fspan(0,36,37), cor_WNP_IO, lires)

lires@xyExplicitLegendLabels = (/"   WIO -> WNA",""/)
lires@pmLegendOrthogonalPosF = -0.35  ;正为向上
lires@pmLegendParallelPosF = 0.67
lires@xyLineColors       := (/"violetred2","violetred2","violetred2"/)
lires@xyDashPatterns     := (/0,4,2/)
lires@xyLineThicknesses  := (/1.0,8.0,8.0/)
contour1(1) = gsn_csm_xy(wks, fspan(0,36,37), cor_IO_WEA, lires)
overlay(plot(1), contour1(1))

lires@xyExplicitLegendLabels = (/"   WIO -> WNA",""/)
lires@pmLegendOrthogonalPosF = -0.87  ;正为向上
lires@pmLegendParallelPosF = 0.67
lires@xyLineColors       := (/"dodgerblue","violetred2","violetred2"/)
lires@xyDashPatterns     := (/0,4,2/)
lires@xyLineThicknesses  := (/1.0,8.0,8.0/)
contour2(1) = gsn_csm_xy(wks, fspan(0,36,37), cor_WNP_WEA, lires)
overlay(plot(1), contour2(1))


dum1 = new(36,graphic)
dum2 = new(36,graphic)
dum3 = new(36,graphic)
pmres = True
do i=0,36
  pmres@gsMarkerColor = "springgreen3"
  pmres@gsMarkerIndex = 4
  if (cor_WNP_IO_sign95(i).lt.0.0) then
    dum1(i) = gsn_add_polymarker(wks, plot(1), i, cor_WNP_IO_sign95(i), pmres)
  end if

  pmres@gsMarkerColor = "violetred2"
  pmres@gsMarkerIndex = 7
  if (cor_IO_WEA_sign95(i).gt.0.0) then
    dum2(i) = gsn_add_polymarker(wks, plot(1), i, cor_IO_WEA_sign95(i), pmres)
  end if

  pmres@gsMarkerColor = "dodgerblue"
  pmres@gsMarkerIndex = 12
  if (cor_WNP_WEA_sign95(i).lt.0.0) then
    dum3(i) = gsn_add_polymarker(wks, plot(1), i, cor_WNP_WEA_sign95(i), pmres)
  end if 
end do






  pres = True
  pres@gsnFrame = False
  pres@gsnPanelLabelBar = True
  pres@lbOrientation        = "Vertical"
  pres@pmLabelBarWidthF     = 0.04
  pres@pmLabelBarHeightF    = 0.15
  pres@pmLabelBarOrthogonalPosF = -0.005
  ; pres@pmLabelBarParallelPosF = 0.01
  pres@lbLabelFontHeightF  = 0.010


  pres@gsnPanelLabelBar = False
  pres@pmLabelBarOrthogonalPosF = 0.0

  pres@gsnPanelTop  = 0.625
  pres@gsnPanelBottom = 0.38
  pres@gsnPanelLeft  = 0.0
  pres@gsnPanelRight = 0.51
  gsn_panel(wks,plot(0),(/1,1/),pres)

  pres@gsnPanelTop  = 0.63
  pres@gsnPanelBottom = 0.38
  pres@gsnPanelLeft  = 0.48
  pres@gsnPanelRight = 0.999
  gsn_panel(wks,plot(1),(/1,1/),pres)




txres               = True
txres@txPerimOn = False
txres@txFontHeightF = 0.014
; txres@txFont = "helvetica"
txres@txFontColor = "black"
gsn_text_ndc(wks, "Two-step   signal   transport",  0.255, 0.64, txres)
gsn_text_ndc(wks, "Lead-lag   correaltions",  0.745, 0.64, txres)


txres@txFontHeightF = 0.0180
txres@txFont = "helvetica-bold"
gsn_text_ndc(wks, "a",  0.069, 0.663, txres)
gsn_text_ndc(wks, "b",  0.555, 0.663, txres)



frame(wks)
   delete(wks)  
   system("convert -geometry 2400x2400 -density 1800x1800 -trim " + pltName + "."+pltType + " " + pltName + ".png")     
  system("rm " + pltName + "." + pltType + " -f")



end


