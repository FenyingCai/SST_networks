

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.colors as mcolors
import xarray as xr
import numpy as np



labels1 = ["  tropics", "       NH<br> extratropics", "       SH<br> extratropics", " tropics  ", "NH       <br> extratropics ", "SH       <br> extratropics "]
source1 = [0, 0, 0, 1, 1, 1, 2, 2, 2]
target1 = [3, 4, 5, 3, 4, 5, 3, 4, 5]


# source_circle = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
# target_circle = [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]
link_color1 = ["rgba(128,128,128,0.3)","rgba(128,128,128,0.3)","rgba(128,128,128,0.3)",
              "rgba(43,42,118,0.3)","rgba(128,128,128,0.3)","rgba(128,128,128,0.3)",
              "rgba(115,170,67,0.3)","rgba(128,128,128,0.3)","rgba(128,128,128,0.3)"]
link_color2 = ["rgba(128,128,128,0.3)","rgba(128,128,128,0.3)","rgba(128,128,128,0.3)",
              "rgba(128,128,128,0.3)","rgba(43,42,118,0.3)","rgba(128,128,128,0.3)",
              "rgba(128,128,128,0.3)","rgba(115,170,67,0.3)","rgba(128,128,128,0.3)"]



ds1 = xr.open_dataset("2_analysis/fig5_degree/Flow_hotspots3x3_Tmax_tropics.nc")
flow0 = ds1.flow.data
flow_Tmax = np.zeros(9)
flow_Tmax[0:3] = flow0[0,:]
flow_Tmax[3:6] = flow0[1,:]
flow_Tmax[6:9] = flow0[2,:]


ds2 = xr.open_dataset("2_analysis/fig5_degree/Flow_hotspots3x3_Rain_tropics.nc")
flow0 = ds2.flow.data
flow_Rain = np.zeros(9)
flow_Rain[0:3] = flow0[0,:]
flow_Rain[3:6] = flow0[1,:]
flow_Rain[6:9] = flow0[2,:]
print(flow0)




##--------------------------------------------- Plot  --------------------------------------------##
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'sankey'},{'type':'sankey'}]])

Sankey1 = go.Sankey(
node=dict(pad=15, thickness=20,
          color=["rgba(162,29,47,0.7)","rgba(43,42,118,0.75)","rgba(115,170,67,0.75)","rgba(162,29,47,0.7)","rgba(43,42,118,0.75)","rgba(115,170,67,0.75)"],
          line=dict(color="black", width=0.5),label=labels1),
link=dict(source=source1, target=target1, value=flow_Tmax, color=link_color1))

Sankey2 = go.Sankey(
node=dict(pad=15, thickness=20,
          color=["rgba(162,29,47,0.7)","rgba(43,42,118,0.75)","rgba(115,170,67,0.75)","rgba(162,29,47,0.7)","rgba(43,42,118,0.75)","rgba(115,170,67,0.75)"],
          line=dict(color="black", width=0.5),label=labels1),
link=dict(source=source1, target=target1, value=flow_Rain, color=link_color2))



fig.add_trace(Sankey1, row=1, col=1)
fig.add_trace(Sankey2, row=1, col=2)





fig.update_layout(
    font=dict(color='black', size=13),
    width = 500,
    height = 320,
    
    title=dict(
    text='',#'Flow    among    three    oceans',
    x=0.5,
    y=0.995,
    xanchor='center',
    font=dict(color='black', size=20)),
    margin=dict(l=10,r=10,t=60,b=28)
)


fig.add_annotation(text='i', x=0.007, y=1.25, xref='paper',yref='paper',showarrow=False, font=dict(color='black', size=20.5, family='Arial', weight='bold'))
fig.add_annotation(text='j', x=0.57, y=1.25, xref='paper',yref='paper',showarrow=False, font=dict(color='black', size=20.5, family='Arial', weight='bold'))

fig.add_annotation(text='SST -> Tmax', x=0.12, y=1.14, xref='paper',yref='paper',showarrow=False, font=dict(color='black', size=15, family='Arial'))
fig.add_annotation(text='SST -> Precip', x=0.875, y=1.14, xref='paper',yref='paper',showarrow=False, font=dict(color='black', size=15, family='Arial'))


fig.write_image('2_analysis/Figure4ij_flow_tropics.png', scale=4)


