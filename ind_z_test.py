# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 22:42:19 2020

@author: adogan
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 23:41:26 2020

@author: adoga
"""


# t-test for dependent samples
from math import sqrt
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from scipy.stats import t
import numpy as np
import scipy as sp
import scipy.stats

#original risks
ori_risk_list = list()
f = open(r'C:\Users\adogan\OneDrive - Oklahoma A and M System\Classified\Codes2\MI\MI_New_R0.txt', "r")
for x in f:
    for value in x[1:-1].split(','):
        value = float(value)
        #print(value)
        ori_risk_list.append(value)
        
#Recommended risks
rec_risk_list = list()
f = open(r'C:\Users\adogan\OneDrive - Oklahoma A and M System\Classified\Codes2\MI\MI_New_R1.txt', "r")
for x in f:
    for value in x[1:-1].split(','):
        value = float(value)
        #print(value)
        rec_risk_list.append(value)
        
#Calculation of the differences
ori_rec = list()
for i in range(len(rec_risk_list)):
    d = ori_risk_list[i] - rec_risk_list[i]
    ori_rec.append(d)
    
#upper side t-test
#Recommended risks
ori_rec = list()
import pandas as pd
f =pd.read_csv(r'C:\Users\adogan\OneDrive - Oklahoma A and M System\Classified\Codes2\FATCHD\p_list.txt')
f.columns = [0]
ori_risk = 0
#rec_risk = ori_data.iloc[1,2]
f = f.iloc[1:,:]
rec_risk = f[0].mean()
ori_rec = f[0].to_list()
end_point = max(ori_rec)
  
    
import statistics 
import math
#Calculation of mean
d_bar = statistics.mean(ori_rec)
# Standard deviation of list 
# Using pstdev() 
st_d = statistics.pstdev(ori_rec) 
#Sample size
n = len(ori_rec)

#Calculation of t_0
z = d_bar/ (st_d/math.sqrt(n))
#Get p_val
p = scipy.stats.norm.sf(abs(z)) #one-side

a = 0.05
import scipy.stats as st
#critical value
z_0 = st.norm.ppf(1 - a)


import seaborn as sns
import scipy.stats as stats
import numpy as np
import random
import warnings
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
values = ori_rec.copy()
two_std_from_mean = np.mean(values) + np.std(values)*1.645
kde = stats.gaussian_kde(values)
pos = np.linspace(np.min(values), np.max(values), 10000)
plt.plot(pos, kde(pos), color='teal')
shade = np.linspace(two_std_from_mean, 0.65, 300)
plt.fill_between(shade, kde(shade), alpha=0.45, color='teal')
plt.title("Sampling Distribution for One-Tail Hypothesis Test", y=1.015, fontsize=20)
plt.xlabel("sample mean value", labelpad=14)
plt.ylabel("frequency of occurence", labelpad=14);



#another p calculation way
#import scipy.stats as st
#pval = (1 - st.norm.cdf(z))
#
if p<a and z_0 < z:
    print("reject null hypothesis a = {}, p = {:.4f}, z = {:.4f}, z_0 = {:.4f}".format(a,p,z,z_0))
else:
    print("accept null hypothesisa = {}, p = {:.4f}, z = {:.4f}, z_0 = {:.4f}".format(a,p,z,z_0))   

#distrinution Graph
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import numpy as np
import warnings
import matplotlib.pyplot as plt
sns.set(rc={'figure.figsize':(12, 7.5)})
sns.set_context('talk')
warnings.filterwarnings('ignore')
np.random.seed(42)
population_size = len(ori_rec)
df_heights = pd.DataFrame(ori_rec) #data={'us_height_inches': np.random.normal(loc=66, scale=2.9, size=population_size)})


import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

ax = sns.distplot(df_heights[0], color="maroon", label = 'H0: m_0 = m_r \nH1 = m_0 > m_r ') #.format(ori_risk,d_bar) )
#plt.plot(z_0, 6,'bo')
plt.plot([rec_risk, rec_risk], [0, 11.5],color=(0.1,0.0,0.0),marker = "o", linestyle = "dotted",linewidth= 5, label ="m_r = {:.2f}".format(d_bar) )
plt.plot([ori_risk, ori_risk], [0, 11.5], color = (0.9,0.0,0.0), marker = "v", linewidth= 5,label ="m_0 = {:.2f}".format(ori_risk))
plt.xlabel("original risk - recommended risk", labelpad=14)
plt.ylabel("probability of occurence", labelpad=14)
plt.title("Distribution of Risk Difference of Original and Recommended Lifestyles", y=1.015, fontsize=20);
plt.legend(loc='upper right')
#plt.plot(label = 'p = {:.5f}'.format(p))
#red_patch = mpatches.Patch(color='red', label='The red data')
# #,label = 'p = {:.5f}'.format(p), a ='{:.5f}'.format(a) )
#plt.legend(handles=[red_patch])
#plt.annotate('p = {:.5f}'.format(p))
values = ori_rec.copy()
two_std_from_mean = np.mean(values) + np.std(values)*1.645
kde = stats.gaussian_kde(values)
pos = np.linspace(np.min(values), np.max(values), 10000)
#plt.plot(pos, kde(pos), color=(0.9,0.0,0.0))
shade = np.linspace(two_std_from_mean, round(end_point,1), 300)
plt.fill_between(shade, kde(shade), alpha=0.45, color='teal')
plt.title("Sampling Distribution for One-Tail Hypothesis Test", y=1.015, fontsize=20)
plt.xlabel("sample mean value", labelpad=14)
plt.ylabel("frequency of occurence", labelpad=14);
sns.distplot(df_heights[0], color="maroon", label = 'H0: m_0 = m_r \nH1 = m_0 > m_r ') #.format(ori_risk,d_bar) )
#plt.plot(z_0, 6,'bo')
plt.plot([0.47, 0.47], [0, 10.5],color=(0.1,0.0,0.0),marker = "o", linestyle = "dotted",linewidth= 5, label ="m_r = {:.2f}".format(d_bar) )
plt.plot([0.55, 0.55], [0, 10.5], color = (0.9,0.0,0.0), marker = "v", linewidth= 5,label ="m_0 = {:.2f}".format(ori_risk))
plt.xlabel("original risk - recommended risk", labelpad=14)
plt.ylabel("probability of occurence", labelpad=14)
plt.title("Distribution of Risk Difference of Original and Recommended Lifestyles", y=1.015, fontsize=20);
plt.legend(loc='upper right')    



nmax = np.max(df_heights[0])
arg_max = None
for j, _n in enumerate(df_heights[0]):
    if _n == nmax:
        arg_max = j
        break
print (b[arg_max])
l = ax.get_xaxis(self)





#two-tailed z test   
import pandas as pd
from scipy import stats
from statsmodels.stats import weightstats as stests
ztest ,pval = stests.ztest(ori_rec, x2=None, value=0)
print(float(pval))
if pval<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")   







import matplotlib.pyplot as plt
import numpy as np

x = list(ori_rec.copy())
#x = np.random.normal(size=1000)
plt.hist(x, density=True, bins=10)  # `density=False` would make counts
plt.ylabel('Probability')
plt.xlabel('Data')

fig=plt.figure() #Plots in matplotlib reside within a figure object, use plt.figure to create new figure
#Create one or more subplots using add_subplot, because you can't create blank figure
ax = fig.add_subplot(1,1,1)
#Variable
ax.hist(x,bins = 20, color = 'green' ) # Here you can play with number of bins
#Labels and Tit
plt.title('Ori - Rec distribution', fontsize = 20)
plt.xlabel('Risk rates', fontsize = 15)
plt.ylabel('#Patients', fontsize = 15)
plt.show()
    