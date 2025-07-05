load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
begin

;;------------------  read lat, lon  -------------------;;
dim_0 = (/60,180, 60,180/)
latt=fspan(-59.0,59.0,60)   ;; 2*2 resolution
lonn=fspan(-179.0,179.0,180)
latt@units="degrees_north"
lonn@units="degrees_east"

lat=fspan(-79.0,79.0,80)   ;; 2*2 resolution
lon=fspan(1.0,359.0,180)
lat@units="degrees_north"
lon@units="degrees_east"


matrix1 = new((/60,180,60,180/), float)
matrix2 = new((/60,180,80,180/), float)
matrix1 = 0.0
matrix2 = 0.0
do i = 0,59
do j = 0,179
if ((i+j)%2.eq.0.0) then
  matrix1(i,j,:,:) = 1.0
  matrix2(i,j,:,:) = 1.0
end if
end do
end do
do i = 0,59
do j = 0,179
if ((i+j)%2.eq.0.0) then
  matrix1(:,:,i,j) = 1.0
end if
end do
end do
do i = 0,79
do j = 0,179
if ((i+j)%2.eq.0.0) then
  matrix2(:,:,i,j) = 1.0
end if
end do
end do



;;---------------------------------------------  links at different lags  --------------------------------------------;;
f1 = addfile("1_networks/part1_HadISST/Networks2x2_SST_lag4.nc","r")
networks4_SST = (f1->networks0)
networks4_SST = where(networks4_SST.gt.0.0, 1.0, networks4_SST)
networks4_SST = where(networks4_SST.lt.0.0, -1.0, networks4_SST)
networks4_SST = where(ismissing(networks4_SST), 0.0, networks4_SST)
networks4_SST = networks4_SST * matrix1

networks4_SST!0 = "lat"
networks4_SST!1 = "lon"
networks4_SST&lat = latt
networks4_SST&lon = lonn
networks4_SST!2 = "lat2"
networks4_SST!3 = "lon2"
networks4_SST&lat2 = latt
networks4_SST&lon2 = lonn




networks24_Tmax = new((/60,180, 80,180/), float)
networks24_Tmax = 0.0
networks814_Tmax = new((/60,180, 80,180/), float)
networks814_Tmax = 0.0
do ilag = 2,4
print(" ilag = "+ilag)
f2 = addfile("1_networks/part11_HadISST_CPCTmax_2x2/Networks2x2_SST_Tmax_lag"+ilag+".nc","r")
networks0 = (f2->networks0)
networks0 = where(networks0.gt.0.0, 1.0, networks0)
networks24_Tmax = networks24_Tmax + networks0
end do
do ilag = 8,14
print(" ilag = "+ilag)
f2 = addfile("1_networks/part11_HadISST_CPCTmax_2x2/Networks2x2_SST_Tmax_lag"+ilag+".nc","r")
networks0 = (f2->networks0)
networks0 = where(networks0.gt.0.0, 1.0, networks0)
networks0 = where(networks0.lt.0.0, -1.0, networks0)
networks814_Tmax = networks814_Tmax + networks0
end do
networks24_Tmax = networks24_Tmax * matrix2
networks814_Tmax = networks814_Tmax * matrix2


networks24_Tmax!0 = "lat"
networks24_Tmax!1 = "lon"
networks24_Tmax&lat = latt
networks24_Tmax&lon = lonn
networks24_Tmax!2 = "lat2"
networks24_Tmax!3 = "lon2"
networks24_Tmax&lat2 = lat
networks24_Tmax&lon2 = lon
copy_VarCoords(networks24_Tmax, networks814_Tmax)









;;--------------------------------------------------------------------------------------------------;;
f1 = addfile("0_data/topo_2x2.nc","r")
topo = (f1->topo)
key_region1 = topo
key_region2 = topo
key_region3 = topo

;;------------------------------------------------;;
IO_lat1 = -10
IO_lat2 = 5
IO_lon1 = 70
IO_lon2 = 85

WP_lat1 = -10
WP_lat2 = 10
WP_lon1 = 140
WP_lon2 = 155

SWPacific_lat1 = -35
SWPacific_lat2 = -20
SWPacific_lon1 = 160
SWPacific_lon2 = 180

SEAsia_lat1 = 5
SEAsia_lat2 = 22
SEAsia_lon1 = 95
SEAsia_lon2 = 110


key_region1({SWPacific_lat1:SWPacific_lat2},{SWPacific_lon1:SWPacific_lon2}) = 10000.0
key_region1({WP_lat1:WP_lat2},{WP_lon1:WP_lon2}) = 10000.0
key_region1({IO_lat1:IO_lat2},{IO_lon1:IO_lon2}) = 10000.0
key_region1 = where(topo.le.0.0, key_region1, -1.0)
key_region2({SEAsia_lat1:SEAsia_lat2},{SEAsia_lon1:SEAsia_lon2}) = 100000.0
key_region1 = where(topo.ge.0.0.and.key_region2.gt.50000, 10000.0, key_region1)


key_region3({SWPacific_lat1:SWPacific_lat2},{SWPacific_lon1:SWPacific_lon2}) = 10000.0
key_region3 = where(topo.le.0.0, key_region3, -1.0)
key_region2({SEAsia_lat1:SEAsia_lat2},{SEAsia_lon1:SEAsia_lon2}) = 100000.0
key_region3 = where(topo.ge.0.0.and.key_region2.gt.50000, 10000.0, key_region3)
















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
ores@cnMonoFillPattern    = False            ; want multiple patterns                                                                                                                                     
ores@cnMonoFillScale      = False            ; want different densities                                                                                                                                   
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
pltName = "3_extended_data_fig/FigureS5_extra_Asia"
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
res@tmXBMinorOutwardLengthF = 0.0135
res@tmYLMinorOutwardLengthF = 0.0135

res@mpMinLatF = -40
res@mpMaxLatF = 40
res@mpMinLonF = 60
res@mpMaxLonF = 195
res@mpCenterLonF = 120
res@tmXBMode = "Explicit"
res@tmXBValues := (/-180,-120,-60,0,60,90,120,150,180,240,300,360,420/)
res@tmXBLabels := (/"180","120W","60W","0","60E","90E","120E","150E","180","120W","60W","0","60E"/)
res@tmYLMode = "Explicit"
res@tmYLValues := (/-40,-20,0,20,40/)
res@tmYLLabels := (/"40S","20S","0","20N","40N"/)
res@tmYLMinorValues = fspan(-90, 90, 19)


res@tmYLLabelsOn = True
res@tmXBLabelFontHeightF = 0.018
res@tmYLLabelFontHeightF = 0.018
res@cnFillPalette = "GMT_gray"
res@cnLevels     := (/0.5/)
res@cnFillColors := (/-1,8/)
plot(0) = gsn_csm_contour_map_ce(wks, key_region1, res)
plot(1) = gsn_csm_contour_map_ce(wks, key_region3, res)





;;----------------------------- text -----------------------------;;
txres               = True
txres@txFont = "helvetica"
txres@txFontHeightF = 0.019
txres@txPerimOn = True
txres@txBackgroundFillColor = "white"


txres@txPerimColor = "violetred2"
txres@txFontColor = "violetred2"
eum1 = gsn_add_text(wks,plot(0), "lag = 4 months", 175, 16, txres)
eum2 = gsn_add_text(wks,plot(0), "lag = 2-4 months", 90, 33, txres)

txres@txPerimColor = "dodgerblue"
txres@txFontColor = "dodgerblue"
eum4 = gsn_add_text(wks,plot(0), "lag = 4 months", 110, -34, txres)
eum3 = gsn_add_text(wks,plot(1), "lag = 8-14 months", 170, 30, txres)









;;-----------------------------------  Network Links (1.): SWPacific->WP  ---------------------------------;;
network4_new := networks4_SST({SWPacific_lat1:SWPacific_lat2},{SWPacific_lon1:SWPacific_lon2}, {WP_lat1:WP_lat2},{WP_lon1:WP_lon2})

dim_new = dimsizes(network4_new)
linelink1 = new(10000, "graphic")
k = 0

do ilat = 0, dim_new(0)-1
print("ilat = "+ilat)
do ilon = 0, dim_new(1)-1
do jlat = 0, dim_new(2)-1
do jlon = 0, dim_new(3)-1


if (network4_new(ilat,ilon,jlat,jlon).gt.0.0) then
lat1 := SWPacific_lat1 + ilat*2
lon1 := SWPacific_lon1 + 1.0 + ilon*2
lat2 := WP_lat1 + 1.0 + jlat*2
lon2 := WP_lon1 + 1.0 + jlon*2

npts = 100
lon := new(npts, "float")
lat := new(npts, "float")
do i=0,npts-1
  t = i / (npts - 1.0)
  lon(i) = lon1 + t * (lon2 - lon1)
  lat(i) = lat1 + t * (lat2 - lat1)
  ; lat(i) = lat(i) + 20.0 * sin(3.14159 * t)
  lon(i) = lon(i) + 20.0 * sin(3.14159 * t)
end do

lnres = True
lnres@gsLineColor="violetred2"
lnres@gsLineDashPattern=0
lnres@gsLineThicknessF=0.001
linelink1(k) = gsn_add_polyline(wks,plot(0), lon,lat, lnres)
k = k+1
end if

end do
end do
end do
end do





;;--------------------------------------  Network Links (2.): WP->IO  ------------------------------------;;
network4_new := networks4_SST({WP_lat1:WP_lat2},{WP_lon1:WP_lon2}, {IO_lat1:IO_lat2},{IO_lon1:IO_lon2})
dim_new = dimsizes(network4_new)
linelink2 = new(10000, "graphic")
k = 0

do ilat = 0, dim_new(0)-1
print("ilat = "+ilat)
do ilon = 0, dim_new(1)-1
do jlat = 0, dim_new(2)-1
do jlon = 0, dim_new(3)-1


if (network4_new(ilat,ilon,jlat,jlon).lt.0.0) then
lat1 := WP_lat1 + 1.0 + ilat*2
lon1 := WP_lon1 + 1.0 + ilon*2
lat2 := IO_lat1 + 1.0 + jlat*2
lon2 := IO_lon1 + 1.0 + jlon*2

npts = 100
lon := new(npts, "float")
lat := new(npts, "float")
do i=0,npts-1
  t = i / (npts - 1.0)
  lon(i) = lon1 + t * (lon2 - lon1)
  lat(i) = lat1 + t * (lat2 - lat1)
  lat(i) = lat(i) - 20.0 * sin(3.14159 * t)
end do

lnres@gsLineColor="dodgerblue"
lnres@gsLineDashPattern=0
lnres@gsLineThicknessF=0.001
linelink2(k) = gsn_add_polyline(wks,plot(0), lon,lat, lnres)
k = k+1
end if

end do
end do
end do
end do






;;--------------------------------------  Network Links (3.): IO->SEAsia  ------------------------------------;;
network24_new := networks24_Tmax({IO_lat1:IO_lat2},{IO_lon1:IO_lon2}, {SEAsia_lat1:SEAsia_lat2},{SEAsia_lon1:SEAsia_lon2})
network24_new = where(network24_new.gt.0.0, network24_new, 0.0)
dim_new = dimsizes(network24_new)
linelink3 = new(10000, "graphic")
k = 0

do ilat = 0, dim_new(0)-1
print("ilat = "+ilat)
do ilon = 0, dim_new(1)-1
do jlat = 0, dim_new(2)-1
do jlon = 0, dim_new(3)-1


if (.not.ismissing(network24_new(ilat,ilon,jlat,jlon)).and.network24_new(ilat,ilon,jlat,jlon).gt.0.0) then
lat1 := IO_lat1 + 1.0 + ilat*2
lon1 := IO_lon1 + 1.0 + ilon*2
lat2 := SEAsia_lat1 + jlat*2
lon2 := SEAsia_lon1 + jlon*2

npts = 100
lon := new(npts, "float")
lat := new(npts, "float")
do i=0,npts-1
  t = i / (npts - 1.0)
  lon(i) = lon1 + t * (lon2 - lon1)
  lat(i) = lat1 + t * (lat2 - lat1)
  lat(i) = lat(i) + 15.0 * sin(3.14159 * t)
end do

lnres@gsLineColor="violetred2"
lnres@gsLineDashPattern=0
lnres@gsLineThicknessF=0.001
linelink3(k) = gsn_add_polyline(wks,plot(0), lon,lat, lnres)
k = k+1
end if

end do
end do
end do
end do





;;-----------------------------------  Network Links (4.): SWPacific->SEAsia  ---------------------------------;;
network814_new := networks814_Tmax({SWPacific_lat1:SWPacific_lat2},{SWPacific_lon1:SWPacific_lon2}, {SEAsia_lat1:SEAsia_lat2},{SEAsia_lon1:SEAsia_lon2})

dim_new = dimsizes(network814_new)
linelink4 = new(10000, "graphic")
k = 0

do ilat = 0, dim_new(0)-1
print("ilat = "+ilat)
do ilon = 0, dim_new(1)-1
do jlat = 0, dim_new(2)-1
do jlon = 0, dim_new(3)-1


if (.not.ismissing(network814_new(ilat,ilon,jlat,jlon)).and.network814_new(ilat,ilon,jlat,jlon).lt.0.0) then
lat1 := SWPacific_lat1 + ilat*2
lon1 := SWPacific_lon1 + 1.0 + ilon*2
lat2 := SEAsia_lat1 + jlat*2
lon2 := SEAsia_lon1 + jlon*2

npts = 100
lon := new(npts, "float")
lat := new(npts, "float")
do i=0,npts-1
  t = i / (npts - 1.0)
  lon(i) = lon1 + t * (lon2 - lon1)
  lat(i) = lat1 + t * (lat2 - lat1)
  lat(i) = lat(i) + 30.0 * sin(3.14159 * t)
  ; lon(i) = lon(i) + 10.0 * sin(3.14159 * t)
end do

lnres = True
lnres@gsLineColor="dodgerblue"
lnres@gsLineDashPattern=0
lnres@gsLineThicknessF=0.001
linelink4(k) = gsn_add_polyline(wks,plot(1), lon,lat, lnres)
k = k+1
end if

end do
end do
end do
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

  pres@gsnPanelTop  = 0.635
  pres@gsnPanelBottom = 0.37
  pres@gsnPanelLeft  = 0.0
  pres@gsnPanelRight = 0.53
  gsn_panel(wks,plot(1),(/1,1/),pres)

  pres@gsnPanelTop  = 0.635
  pres@gsnPanelBottom = 0.37
  pres@gsnPanelLeft  = 0.46
  pres@gsnPanelRight = 0.99
  gsn_panel(wks,plot(0),(/1,1/),pres)




txres               = True
txres@txPerimOn = False
txres@txFontHeightF = 0.0125
; txres@txFont = "helvetica"
txres@txFontColor = "black"
gsn_text_ndc(wks, "Long-delayed   connection   to   Tmax",  0.275, 0.65, txres)
gsn_text_ndc(wks, "Four-step   signal   transport",  0.745, 0.65, txres)


txres@txFontHeightF = 0.0170
txres@txFont = "helvetica-bold"
gsn_text_ndc(wks, "a",  0.093, 0.67, txres)
gsn_text_ndc(wks, "b",  0.555, 0.67, txres)


frame(wks)
   delete(wks)  
   system("convert -geometry 2400x2400 -density 1800x1800 -trim " + pltName + "."+pltType + " " + pltName + ".png")     
  system("rm " + pltName + "." + pltType + " -f")



end


