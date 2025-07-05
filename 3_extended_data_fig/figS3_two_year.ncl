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





sst_fromJJA = new((/44,4,dim_00(1),dim_00(2)/), float) 
do iyear = 0,43
    sst_fromJJA(iyear,0,:,:) = dim_avg_n_Wrap(sst0(iyear*12+6:iyear*12+8,:,:), 0)
    sst_fromJJA(iyear,1,:,:) = dim_avg_n_Wrap(sst0(iyear*12+9:iyear*12+11,:,:), 0)
    sst_fromJJA(iyear,2,:,:) = dim_avg_n_Wrap(sst0(iyear*12+12:iyear*12+14,:,:), 0)
    sst_fromJJA(iyear,3,:,:) = dim_avg_n_Wrap(sst0(iyear*12+15:iyear*12+17,:,:), 0)
end do
printVarSummary(sst_fromJJA)






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



WNP_JJA = dim_avg_n_Wrap(dim_avg_n_Wrap(sst_fromJJA(0:42,0,{WNP_lat1:WNP_lat2},{WNP_lon1:WNP_lon2}), 2), 1) 
IO_DJA = dim_avg_n_Wrap(dim_avg_n_Wrap(sst_fromJJA(0:42,2,{IO_lat1:IO_lat2},{IO_lon1:IO_lon2}), 2), 1) 
WEA_JJA = dim_avg_n_Wrap(dim_avg_n_Wrap(sst_fromJJA(1:43,0,{WEA_lat1:WEA_lat2},{WEA_lon1:WEA_lon2}), 2), 1) 

multi = WNP_JJA*IO_DJA*WEA_JJA

print(multi)
print(dim_min(multi))
print(" 18 "+WNP_JJA(18)+" "+IO_DJA(18)+" "+WEA_JJA(18))

print(" 36 "+WNP_JJA(36)+" "+IO_DJA(36)+" "+WEA_JJA(36))






















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
; sres2@cnLineColors     = (/"red","red","red","red"/)
; sres2@cnLineDashPatterns = (/16,16,16,0,0,0,0/)
; sres2@cnLineThicknesses = (/4,4,4,4,4,4,4/)




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
; ores@cnFillPatterns       = (/-1,11/)     ; the patterns                                                                         
ores@cnMonoFillScale      = False            ; want different densities                                                             
; ores@cnFillScales         = (/0.6,0.6/)    ; change densities                                                                         
ores@cnMonoFillColor      =True                                                                                                     
ores@cnFillDotSizeF       = 0.003    
ores@cnFillDrawOrder ="postDraw"
ores@cnFillColor = "gray30"

ores@cnLevels        = (/0.05/)   ;;90%
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
pltName = "3_extended_data_fig/FigureS3_two_year"
wks = gsn_open_wks(pltType, pltName)



res@tmXBLabelsOn = True
res@cnMissingValFillColor = -1
; res@cnFillMode = "CellFill"
res@gsnLeftString = ""


res@tmYLMajorLengthF = 0.016
res@tmXBMajorLengthF = 0.016
res@tmYLMinorLengthF = 0.0115
res@tmXBMinorLengthF = 0.0115
res@tmXBMajorOutwardLengthF = 0.016
res@tmYLMajorOutwardLengthF = 0.016
res@tmXBMinorOutwardLengthF = 0.0115
res@tmYLMinorOutwardLengthF = 0.0115

res@mpMinLatF = -60
res@mpMaxLatF = 60
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
res@cnFillPalette := "MPL_RdBu" 
res@cnLevels     := (/-0.5,-0.3,-0.1,0.1,0.3,0.5/)
res@cnFillColors := (/94,88,80,-1,45,37,31/)

res@tmXBLabelsOn = False
plot(0) = gsn_csm_contour_map_ce(wks, sst_fromJJA(18,0,:,:), res)
plot(1) = gsn_csm_contour_map_ce(wks, sst_fromJJA(18,2,:,:), res)
res@tmXBLabelsOn = True
plot(2) = gsn_csm_contour_map_ce(wks, sst_fromJJA(19,0,:,:), res)


res@tmYLLabelsOn = False
res@tmXBLabelsOn = False
plot(3) = gsn_csm_contour_map_ce(wks, sst_fromJJA(36,0,:,:), res)
plot(4) = gsn_csm_contour_map_ce(wks, sst_fromJJA(36,2,:,:), res)
res@tmXBLabelsOn = True
plot(5) = gsn_csm_contour_map_ce(wks, sst_fromJJA(37,0,:,:), res)



lnres   =    True
lnres@gsLineThicknessF=2.0
lnres@gsLineDashPattern=0
lnres@gsLineColor="darkgreen"


y1=WNP_lat1
y2=WNP_lat2
x1=WNP_lon1
x2=WNP_lon2
x=(/x1,x2,x2,x1,x1/)
y=(/y2,y2,y1,y1,y2/)
dum21 = gsn_add_polyline(wks,plot(0),x,y,lnres)
dum22 = gsn_add_polyline(wks,plot(3),x,y,lnres)

y1=IO_lat1
y2=IO_lat2
x1=IO_lon1
x2=IO_lon2
x=(/x1,x2,x2,x1,x1/)
y=(/y2,y2,y1,y1,y2/)
dum23 = gsn_add_polyline(wks,plot(1),x,y,lnres)
dum24 = gsn_add_polyline(wks,plot(4),x,y,lnres)

y1=WEA_lat1
y2=WEA_lat2
x1=WEA_lon1
x2=WEA_lon2
x=(/x1,x2,x2,x1,x1/)
y=(/y2,y2,y1,y1,y2/)
dum25 = gsn_add_polyline(wks,plot(2),x,y,lnres)
dum26 = gsn_add_polyline(wks,plot(5),x,y,lnres)







  pres = True
  pres@gsnFrame = False
  pres@gsnPanelLabelBar = True
  pres@lbOrientation        = "Horizontal"
  pres@pmLabelBarWidthF     = 0.4
  pres@pmLabelBarHeightF    = 0.05
  pres@pmLabelBarOrthogonalPosF = -0.01
  pres@pmLabelBarParallelPosF = -0.25
  pres@lbLabelFontHeightF  = 0.010


  pres@gsnPanelLabelBar = False
  pres@pmLabelBarOrthogonalPosF = 0.0

  pres@gsnPanelTop  = 0.84
  pres@gsnPanelBottom = 0.64
  pres@gsnPanelLeft  = 0.0
  pres@gsnPanelRight = 0.505
  gsn_panel(wks,plot(0),(/1,1/),pres)

  pres@gsnPanelTop  = 0.6
  pres@gsnPanelBottom = 0.4
  gsn_panel(wks,plot(1),(/1,1/),pres)

  pres@gsnPanelTop  = 0.37
  pres@gsnPanelBottom = 0.14
  pres@gsnPanelLeft  = 0.0
  pres@gsnPanelRight = 0.515
  gsn_panel(wks,plot(2),(/1,1/),pres)



  pres@gsnPanelTop  = 0.84 - 0.003
  pres@gsnPanelBottom = 0.64 - 0.003
  pres@gsnPanelLeft  = 0.520
  pres@gsnPanelRight = 1.0
  gsn_panel(wks,plot(3),(/1,1/),pres)

  pres@gsnPanelTop  = 0.6 - 0.003
  pres@gsnPanelBottom = 0.4 - 0.003
  gsn_panel(wks,plot(4),(/1,1/),pres)

  pres@gsnPanelTop  = 0.37
  pres@gsnPanelBottom = 0.14-0.05
  pres@gsnPanelLeft  = 0.513
  pres@gsnPanelRight = 1.0
  pres@gsnPanelLabelBar = True
  gsn_panel(wks,plot(5),(/1,1/),pres)  



txres               = True
txres@txPerimOn = False
txres@txFontHeightF = 0.014
; txres@txFont = "helvetica"
txres@txFontColor = "black"
gsn_text_ndc(wks, "JJA  1997",  0.260, 0.855, txres)
gsn_text_ndc(wks, "DJF  1997/98",  0.260, 0.615, txres)
gsn_text_ndc(wks, "JJA  1998",  0.260, 0.375, txres)

gsn_text_ndc(wks, "JJA  2015",  0.75, 0.855, txres)
gsn_text_ndc(wks, "DJF  2015/16",  0.75, 0.615, txres)
gsn_text_ndc(wks, "JJA  2016",  0.75, 0.375, txres)


txres@txFontHeightF = 0.0180
txres@txFont = "helvetica-bold"
gsn_text_ndc(wks, "a",  0.05, 0.86, txres)
gsn_text_ndc(wks, "b",  0.535, 0.86, txres)
gsn_text_ndc(wks, "c",  0.05, 0.62, txres)
gsn_text_ndc(wks, "d",  0.535, 0.62, txres)
gsn_text_ndc(wks, "e",  0.05, 0.38, txres)
gsn_text_ndc(wks, "f",  0.535, 0.38, txres)

frame(wks)
   delete(wks) 
   system("convert -geometry 1200x1200 -density 800x800 -trim " + pltName + "."+pltType + " " + pltName + ".png")       
  system("rm " + pltName + "." + pltType + " -f")



end


