# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 19:19:05 2020

@author: adogan
"""

import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
#Load p - values 
f =pd.read_csv(r'C:\Users\adogan\OneDrive - Oklahoma A and M System\Classified\Codes2\MI\MI_p_list.txt')
f.columns = ['p_val']

#sns.distplot(f['p_val'], color="maroon", label = 'H0: m_0 = m_r \nH1 = m_0 > m_r ') #.format(ori_risk,d_bar) )
#plt.plot(z_0, 6,'bo')
counts, bins = np.histogram(f.p_val, bins = np.arange(0, 0.50, 0.05))
#bins = 0.5 * (bins[:-1] + bins[1:])
#x = [1,2,3,4,5,6,7,8,9,10]
x = list(bins)
y = list(counts)
y_per = list()
num_p = f.shape[0]
for i in range(len(y)):
    if y[i] != 0:
        per_val = str(round(y[i]/num_p,3))+'%'
        y_per.append(per_val)
    else:
        per_val = ''
        y_per.append(per_val)
    
y_axis = np.arange(0, 1, 0.05)
# Use textposition='auto' for direct text
fig = go.Figure(data=[go.Bar(
            x=x, y=y_per,
            text=y_per,
            textposition='outside',
            textfont_size = 20,
            marker_color=px.colors.qualitative.Dark24[3],
            textfont=dict(
                family="sans serif",
                size=20,
                color="black"
            )            
        )])
    
fig.update_layout(
    title="p value distribution of patients",
    xaxis_title="p-values",
    yaxis_title="rate of the patients",
    font=dict(
        family="sans serif",
        size=18,
        color="black"
    )
)
fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.9)
fig.update_traces(text=y_per,textposition='outside', textfont_size=14)
#fig.add_trace(text=y_per,textposition='outside',textfont=dict(
#        family="sans serif",
#        size=18,
#        color="LightSeaGreen"
#    ))
fig.show()