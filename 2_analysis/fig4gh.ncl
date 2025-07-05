load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
begin



dim_0 = (/60,180, 80,180/)
latt=fspan(-59.0,59.0,60)   ;; 2*2 resolution
lonn=fspan(-179.0,179.0,180)
latt2=fspan(-79.0,79.0,80)   ;; 2*2 resolution
lonn2=fspan(1.0,359.0,180)
latt@units="degrees_north"
latt2@units="degrees_north"
lonn@units="degrees_east"
lonn2@units="degrees_east"









f1 = addfile("2_analysis/fig4_degree/Network_Tmax_lag13_36.nc","r")
network_Tmax_weight = (f1->network_weight)

degree1_SST = dim_sum_n_Wrap(dim_sum_n_Wrap(network_Tmax_weight(:,:,:,:), 3), 2)
copy_VarCoords(network_Tmax_weight(:,:,0,0), degree1_SST)
degree1_Tmax = dim_sum_n_Wrap(dim_sum_n_Wrap(network_Tmax_weight(:,:,:,:), 1), 0)
copy_VarCoords(network_Tmax_weight(0,0,:,:), degree1_Tmax)



f2 = addfile("2_analysis/fig4_degree/Network_Rain_lag13_36.nc","r")
network_Rain_weight = (f2->network_weight)

degree2_SST = dim_sum_n_Wrap(dim_sum_n_Wrap(network_Rain_weight(:,:,:,:), 3), 2)
copy_VarCoords(network_Rain_weight(:,:,0,0), degree2_SST)
degree2_Rain = dim_sum_n_Wrap(dim_sum_n_Wrap(network_Rain_weight(:,:,:,:), 1), 0)
copy_VarCoords(network_Rain_weight(0,0,:,:), degree2_Rain)






degree1_SST = smth9_Wrap(degree1_SST, 0.5, 0.25, True)
degree1_Tmax = smth9_Wrap(degree1_Tmax, 0.5, 0.25, True)
degree2_SST = smth9_Wrap(degree2_SST, 0.5, 0.25, True)
degree2_Rain = smth9_Wrap(degree2_Rain, 0.5, 0.25, True)


f00 = addfile("0_data/topo_2x2.nc","r")
topo = f00->topo({-80:80},:)
degree1_Tmax = where(topo.ge.-50.0, degree1_Tmax, 1.0e20)
degree1_Tmax@_FillValue = 1.0e20
degree2_Rain = where(topo.ge.-50.0, degree2_Rain, 1.0e20)
degree2_Rain@_FillValue = 1.0e20
copy_VarCoords(network_Tmax_weight(0,0,:,:), degree1_Tmax)
copy_VarCoords(network_Rain_weight(0,0,:,:), degree2_Rain)








;;---------------------------------------------------  plots  ---------------------------------------------------
res = True
res@mpPerimOn = False
res@tmXTBorderOn = False
res@tmXBBorderOn = False
res@tmYLBorderOn = False
res@tmYRBorderOn = False
res@tmBorderThicknessF = 1.1
res@gsnDraw      =  False
res@gsnFrame     =  False
res@gsnAddCyclic =  True
res@gsnRightString       = ""
res@gsnLeftString        = ""
res@gsnLeftStringFontHeightF   = 0.04
res@gsnRightStringFontHeightF  = 0.04
res@gsnCenterStringFontHeightF  = 0.04
; res@tmXTLabelsOn  = False
; res@tmYRLabelsOn  = False
vcres = res
cnres = res
cnres@gsnAddCyclic =  False
res@tmXTOn        = False
res@tmYROn        = False
res@tmXBOn        = False
res@tmYLOn        = False
; res@tmXBLabelFontHeightF = 0.038
; res@tmYLLabelFontHeightF = 0.038
res@tmXBTickSpacingF = 45.0
res@tmYLTickSpacingF = 20.0







res@mpFillOn                    = False
res@mpOutlineOn                 = True
res@mpGeophysicalLineThicknessF = 0.7
res@mpGeophysicalLineColor      = "black";"gray20"
;res@mpGridAndLimbOn = True
;res@mpGridLatSpacingF = 90
;res@mpGridLonSpacingF = 360
;res@gsnMaximize = True
res@cnFillOn             = True
res@cnLinesOn            = False
;res@cnLineThicknessF     = 6.0
;res@cnLineColor          = "red"
res@cnLineLabelsOn       = False
res@lbLabelBarOn         = False
res@lbOrientation        = "Vertical"
res@pmLabelBarWidthF     = 0.045
res@pmLabelBarHeightF    = 0.155
res@pmLabelBarOrthogonalPosF = 0.015
res@lbLabelFontHeightF  = 0.015
;res@cnMissingValFillColor = "white"
;res@gsnYRefLine = 0.0
;res@gsnYRefLineColor = "black"
;res@gsnYRefLineThicknessF = 5.0
res@cnMissingValFillColor = -1
res@cnLevelSelectionMode = "ExplicitLevels"              
; res@cnLevels             = (/140.0,155.0,170.0,185.0,200.0,215.0,230.0,245.0/)
; res@cnFillColors         = (/16,15,13,12,8,6,5,3,-1/)
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
sres@cnLineColor = "blue"
sres@cnLevelSelectionMode = "ExplicitLevels"
sres@cnLevels         = (/10,20,30,40/)
; sres@cnFillColors     = (/"red","gray20"/)
sres@cnMissingValFillColor= -1
sres@cnLineThicknessF = 1.0
sres@cnLineDashPattern = 0


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
lires@vpWidthF  = 0.99
lires@vpHeightF = 0.75
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
lires@tmYLMajorLengthF = 0.06
lires@tmYRMajorLengthF = 0.06
lires@tmXBMajorLengthF = 0.06
lires@tmYLMinorLengthF = 0.04
lires@tmYRMinorLengthF = 0.04
lires@tmXBMinorLengthF = 0.04
lires@tmXBLabelDeltaF = -0.25
lires@tmYLLabelDeltaF = -0.25
lires@tmXBMajorOutwardLengthF = 0.06
lires@tmYLMajorOutwardLengthF = 0.06
lires@tmYRMajorOutwardLengthF = 0.06
lires@tmXBMinorOutwardLengthF = 0.04
lires@tmYLMinorOutwardLengthF = 0.04
lires@tmYRMinorOutwardLengthF = 0.04






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

ores@cnLevels        = (/0.5/)   ;;90%
; ores@cnLevels        = (/-0.312,0.312/)   ;;95%
ores@cnMonoFillPattern    = False            ; want multiple patterns                                                               
ores@cnFillPatterns       = (/-1,12/)     ; the patterns                                                                         
ores@cnMonoFillScale      = False            ; want different densities                                                             
ores@cnFillScales         = (/0.8,0.8/)    ; change densities   
ores@cnFillColor = "black"




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
pltName = "2_analysis/Figure4gh"
wks = gsn_open_wks(pltType, pltName)



res@tmXBLabelsOn = True
res@cnMissingValFillColor = -1
; res@cnFillMode = "CellFill"
res@gsnLeftString = ""


res@mpProjection = "Robinson"
res@mpLimitMode = "LatLon"
; res@mpLimitMode = "LatLon" ; "Angles" ;
; res@mpGridAndLimbOn   = True              ; turn on lat/lon lines
res@mpPerimOn = False

res@mpMinLatF = -60
res@mpMaxLatF = 60
res@mpMinLonF = 0
res@mpMaxLonF = 360
res@mpCenterLonF = 15
res@tmXBLabelFontHeightF = 0.011
res@tmYLLabelFontHeightF = 0.011


res@cnFillPalette = "WhiteBlue"
res@cnLevels     := (/10,40,120/)
res@cnFillColors := (/-1,55,115,170/)
res@tmXBLabelsOn = False
res@tmYLLabelsOn  = False
plot(0) = gsn_csm_contour_map(wks, degree1_SST, res)

res2 = res
res2@mpMinLatF = -60
res2@mpMaxLatF = 60
res2@cnFillPalette = "WhiteYellowOrangeRed"
res2@cnLevels     := (/5,40,250/)
res2@cnFillColors := (/-1,55,110,190/)
res2@cnInfoLabelOn   = False
res2@cnMissingValFillColor = -1
res2@lbLabelBarOn         = True
res2@lbOrientation        = "Vertical"
res2@pmLabelBarWidthF     = 0.035
res2@pmLabelBarHeightF    = 0.28
res2@pmLabelBarOrthogonalPosF = 0.05
res2@pmLabelBarParallelPosF = 0.23
res2@lbLabelFontHeightF  = 0.0104
contour1(0) = gsn_csm_contour(wks, degree1_Tmax, res2)
overlay(plot(0), contour1(0))



res@cnFillPalette = "WhiteBlue"
res@cnLevels     := (/5,15,60/)
res@cnLevels     := (/10,40,120/)
res@cnFillColors := (/-1,55,115,170/)
res@tmXBLabelsOn = True
plot(1) = gsn_csm_contour_map(wks, degree2_SST, res)


res2@cnFillPalette = "MPL_BrBG"
res2@cnLevels     := (/5,10,50/)
res2@cnLevels     := (/5,40,250/)
res2@cnFillColors := (/-1,72,86,110/)
res2@pmLabelBarOrthogonalPosF = 0.05
contour1(1) = gsn_csm_contour(wks, degree2_Rain, res2)
overlay(plot(1), contour1(1))




angle = fspan(0,90,100)
r = 10.0
lon = -180 + r * cos(angle*3.14159/180)
lat = 90 - r * sin(angle*3.14159/180)
polyres = True
polyres@gsFillColor = "white"
po1=gsn_add_polygon(wks, plot(0), lon, lat, polyres)




; po4=gsn_add_polygon(wks, plot(1), (/-180,0,180/), (/-60,-60,-60/), polyres)
; po3=gsn_add_polygon(wks, plot(1), (/-180,0,180/), (/60,60,60/), polyres)

lnres   =    True
lnres@gsLineThicknessF=3.0
lnres@gsLineDashPattern=0
pl1 = gsn_add_polyline(wks,plot(0),(/-180,-180+15,0,180/), (/-60,-60,-60,-60/),lnres)
pl2 = gsn_add_polyline(wks,plot(1),(/-180,-180+15,0,180/), (/-60,-60,-60,-60/),lnres)
pl3 = gsn_add_polyline(wks,plot(0),(/-180,-180+15,0,180/), (/60,60,60,60/),lnres)
pl4 = gsn_add_polyline(wks,plot(1),(/-180,-180+15,0,180/), (/60,60,60,60/),lnres)
lnres@gsLineThicknessF=1.5
pl5 = gsn_add_polyline(wks,plot(0),(/-180+15,-180+15/), (/-60,60/),lnres)
pl6 = gsn_add_polyline(wks,plot(1),(/-180+15,-180+15/), (/-60,60/),lnres)
pl7 = gsn_add_polyline(wks,plot(0),(/180+15,180+15/), (/-60,60/),lnres)
pl8 = gsn_add_polyline(wks,plot(1),(/180+15,180+15/), (/-60,60/),lnres)


;;----------------------------------------------------------;;
lnres   =    True
lnres@gsLineThicknessF=2.0
lnres@gsLineDashPattern=0




  pres = True
  pres@gsnFrame = False
  pres@gsnPanelLabelBar = True
  pres@lbOrientation        = "Vertical"
  pres@pmLabelBarWidthF     = 0.04
  pres@pmLabelBarHeightF    = 0.12
  pres@pmLabelBarOrthogonalPosF = -0.045
  pres@pmLabelBarParallelPosF = 0.065
  pres@lbLabelFontHeightF  = 0.0125


  pres@gsnPanelTop  = 0.90
  pres@gsnPanelBottom = 0.50
  pres@gsnPanelLeft  = 0.01
  pres@gsnPanelRight = 0.995
  pres@pmLabelBarOrthogonalPosF = -0.052
  gsn_panel(wks,plot(0),(/1,1/),pres)
  pres@gsnPanelTop  = 0.53
  pres@gsnPanelBottom = 0.13
  pres@gsnPanelLeft  = 0.01
  pres@gsnPanelRight = 0.995
  pres@pmLabelBarOrthogonalPosF = -0.052
  gsn_panel(wks,plot(1),(/1,1/),pres)





txres               = True
txres@txPerimOn = False
txres@txFontHeightF = 0.0125
; txres@txFont = "helvetica"
txres@txFontColor = "black"

txres@txFontHeightF = 0.0195
txres@txFont = "helvetica-bold"
gsn_text_ndc(wks, "g",  0.075, 0.885, txres)
gsn_text_ndc(wks, "h",  0.075, 0.515, txres)

txres@txFontHeightF = 0.015
txres@txFont = "helvetica"
gsn_text_ndc(wks, "Long-delayed   signals   in   SST-Tmax   networks",  0.38, 0.865, txres)
gsn_text_ndc(wks, "Long-delayed   signals   in   SST-Precip   networks",  0.38, 0.495, txres)


frame(wks)
   delete(wks)  
   system("convert -geometry 2400x2400 -density 1800x1800 -trim " + pltName + "."+pltType + " " + pltName + ".png")     
  system("rm " + pltName + "." + pltType + " -f")



end





