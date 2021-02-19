# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 14:10:51 2020

@author: adogan
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 16:07:16 2020

@author: adogan
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 22:27:09 2020

@author: adogan
"""

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy.stats as st
import statistics 
import scipy
import math
from scipy import stats
import statsmodels
from statsmodels.stats.weightstats import ztest
from statsmodels.tools.decorators import cache_readonly

#loading data
df_ori =pd.read_csv(r'C:\Users\adogan\OneDrive - Oklahoma A and M System\Classified\Codes2\TotalOutcome\TO_5DC5IC.csv')
cols = [        'BMI01', 'ETHANL03', 'SMKSTA', 'MOVE', 'CARB', 'CHOL', 
        'DFIB','PROT', 'SFAT', 'TFAT', 'TOTCAL03',	"HDLSIU02","LDLSIU02",	"TCHSIU01",	"TRGSIU01",	"SBPA21",	"SBPA22",	"ANTA07A",	"ANTA07B",	
           "ECGMA31",	"HMTA03",	"APASIU01",	"APBSIU01",	"LIPA08",	"CHMA09",	'HYPTMD01','ANTICOAGCODE01','ASPIRINCODE01','STATINCODE01',
           "HYPTMDCODE01",	"HYPERT04",	"CIGTYR01",
           "V1AGE01",	"RACEGRP",	"DIABTS02",	"ELEVEL01",	"GENDER",	"CIGT01",	"ANTA01",	"CHOLMDCODE01",	"CLASS"]
df_ori.columns = cols
y1 = df_ori['CLASS']
X1 = df_ori.drop(columns=['CLASS'], axis = 1)#'ID_C'

random_state = 9
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.5, random_state=9)
X_forTrain = X_train.copy()
y_forTrain  = y_train.copy()
X_forTrain['CLASS'] = y_train
df1 = X_forTrain.copy().drop(columns=['HDLSIU02', 'LDLSIU02', 'TCHSIU01','TRGSIU01', 'SBPA21', 'SBPA22', 'ANTA07A', 'ANTA07B', 'ECGMA31',
       'HMTA03','APASIU01', 'APBSIU01', 'LIPA08', 'CHMA09'], axis = 1)


#the variables that has to be learned how to impute by GAIN
#changes = ['BMI01', 'ETHANL03', 'SMKSTA', 'MOVE', 'CARB', 'CHOL', 'DFIB',
#       'PROT', 'SFAT', 'TFAT', 'TOTCAL03',
#       'HDLSIU02', 'LDLSIU02', 'TCHSIU01','TRGSIU01', 'SBPA21', 'SBPA22', 'ANTA07A', 'ANTA07B', 'ECGMA31',
#       'HMTA03','APASIU01', 'APBSIU01', 'LIPA08', 'CHMA09']
changes = [ 'BMI01', 'ETHANL03', 'SMKSTA', 'MOVE', 'CARB', 'CHOL', 
        'DFIB','PROT', 'SFAT', 'TFAT', 'TOTCAL03',]
#
##defining those variables as missing values 
for col in changes:
    df1.loc[df1.sample(frac=0.20).index, col] = pd.np.nan

#convert this dataset to numpy
data_x = df1.to_numpy()

# Parameters
no, dim = data_x.shape
batch_size = 32
hint_rate = 0.5
alpha = 1
iterations = 50000

# Introduce missing data
data_m = np.ones([no, dim])
for i in range(no):
    for j in range(dim):
        if np.isnan(data_x[i,j]):
            data_m[i,j] = 0
miss_data_x = data_x.copy()
miss_data_x[data_m == 0] = np.nan

def normalization(data):
  # Parameters
  _, dim = data.shape
  norm_data = data.copy()
  
  # MixMax normalization
  min_val = np.zeros(dim)
  max_val = np.zeros(dim)
  
  # For each dimension
  for i in range(dim):
    min_val[i] = np.nanmin(norm_data[:,i])
    norm_data[:,i] = norm_data[:,i] - np.nanmin(norm_data[:,i])
    max_val[i] = np.nanmax(norm_data[:,i])
    norm_data[:,i] = norm_data[:,i] / (np.nanmax(norm_data[:,i]) + 1e-6)   
    
  # Return norm_parameters for renormalization
  norm_parameters = {'min_val': min_val,
                     'max_val': max_val}
      
  return norm_data, norm_parameters

def renormalization(norm_data, norm_parameters):
  min_val = norm_parameters['min_val']
  max_val = norm_parameters['max_val']

  _, dim = norm_data.shape
  renorm_data = norm_data.copy()
    
  for i in range(dim):
    renorm_data[:,i] = renorm_data[:,i] * (max_val[i] + 1e-6)   
    renorm_data[:,i] = renorm_data[:,i] + min_val[i]
    
  return renorm_data

def binary_sampler(p, rows, cols):
  unif_random_matrix = np.random.uniform(0., 1., size = [rows, cols])
  binary_random_matrix = 1*(unif_random_matrix < p)
  return binary_random_matrix

def uniform_sampler(low, high, rows, cols):
  return np.random.uniform(low, high, size = [rows, cols])   

def rounding(imputed_data, data_x):
  _, dim = data_x.shape
  rounded_data = imputed_data.copy()
  
  for i in range(dim):
    temp = data_x[~np.isnan(data_x[:, i]), i]
    # Only for the categorical variable
    if len(np.unique(temp)) < 20:
      rounded_data[:, i] = np.round(rounded_data[:, i])
      
  return rounded_data

def xavier_init(size):
  in_dim = size[0]
  xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
  return tf.random_normal(shape = size, stddev = xavier_stddev)

def sample_batch_index(total, batch_size):
  total_idx = np.random.permutation(total)
  batch_idx = total_idx[:batch_size]
  return batch_idx

# Hidden state dimensions
h_dim = int(dim)
  
# Normalization
#data_x = np.array(data_x)
data_x = data_x.astype(np.float)
norm_data, norm_parameters = normalization(data_x)
norm_data_x = np.nan_to_num(norm_data, 0)
  
## GAIN architecture   
# Input placeholders
# Data vector
X = tf.placeholder(tf.float32, shape = [None, dim])
# Mask vector 
M = tf.placeholder(tf.float32, shape = [None, dim])
# Hint vector
H = tf.placeholder(tf.float32, shape = [None, dim])
  
# Discriminator variables
D_W1 = tf.Variable(xavier_init([dim*2, h_dim])) # Data + Hint as inputs
D_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
  
D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
D_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
  
D_W3 = tf.Variable(xavier_init([h_dim, dim]))
D_b3 = tf.Variable(tf.zeros(shape = [dim]))  # Multi-variate outputs
  
theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
  
#Generator variables
# Data + Mask as inputs (Random noise is in missing components)
G_W1 = tf.Variable(xavier_init([dim*2, h_dim]))  
G_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
  
G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
G_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
  
G_W3 = tf.Variable(xavier_init([h_dim, dim]))
G_b3 = tf.Variable(tf.zeros(shape = [dim]))
  
theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]
  
## GAIN functions
# Generator
def generator(x,m):
    # Concatenate Mask and Data
    inputs = tf.concat(values = [x, m], axis = 1) 
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)   
    # MinMax normalized output
    G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3) 
    return G_prob
  
# Discriminator
def discriminator(x, h):
    # Concatenate Data and Hint
    inputs = tf.concat(values = [x, h], axis = 1) 
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)  
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob
  
## GAIN structure
# Generator
G_sample = generator(X, M)
 
# Combine with observed data
Hat_X = X * M + G_sample * (1-M)
  
# Discriminator
D_prob = discriminator(Hat_X, H)
  
## GAIN loss
D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) + (1-M) * tf.log(1. - D_prob + 1e-8)) 
  
G_loss_temp = -tf.reduce_mean((1-M) * tf.log(D_prob + 1e-8))
  
MSE_loss = tf.reduce_mean((M * X - M * G_sample)**2) / tf.reduce_mean(M)
  
D_loss = D_loss_temp
G_loss = G_loss_temp + alpha * MSE_loss 
  
## GAIN solver
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
  
## Iterations
sess = tf.Session()
sess.run(tf.global_variables_initializer())
   
D = np.array(range(0,iterations), dtype = "float64")
G = np.array(range(0,iterations), dtype = "float64")

# Start Iterations
for it in tqdm(range(iterations)):    
      
    # Sample batch
    batch_idx = sample_batch_index(no, batch_size)
    X_mb = norm_data_x[batch_idx, :]  
    M_mb = data_m[batch_idx, :]  
    # Sample random vectors  
    Z_mb = uniform_sampler(0, 1, batch_size, dim) 
    # Sample hint vectors
    H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
    H_mb = M_mb * H_mb_temp
      
    # Combine random vectors with observed vectors
    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
      
    _, D_loss_curr = sess.run([D_solver, D_loss_temp], 
                              feed_dict = {M: M_mb, X: X_mb, H: H_mb})
    _, G_loss_curr, MSE_loss_curr = sess.run([G_solver, G_loss_temp, MSE_loss],
             feed_dict = {X: X_mb, M: M_mb, H: H_mb})
    
    D[it] = D_loss_curr
    G[it] = G_loss_curr

x = range(iterations)
plt.title("Loss plot")
plt.plot(x,D,color = 'blue',linewidth=1, label = 'D loss')
plt.plot(x,G,color = 'red',linewidth=1, label = 'G loss')
plt.legend()
plt.xlabel("Iters")
plt.ylabel("loss")
plt.show()


#Train the model for indirectly changable features


df2 = X_forTrain.copy()
#X.to_csv('ClassificationData.csv')
#data = pd.read_csv(r'ClassificationData.csv')
#y = data['CLASS']
#X = data.drop(columns=[ 'CLASS'], axis = 1)#'ID_C'

#the variables that has to be learned how to impute by GAIN
changes2 = [ 'HDLSIU02', 'LDLSIU02', 'TCHSIU01','TRGSIU01', 'SBPA21', 'SBPA22', 'ANTA07A', 'ANTA07B', 'ECGMA31',
       'HMTA03','APASIU01', 'APBSIU01', 'LIPA08', 'CHMA09']

#defining those variables as missing values 
for col in changes2:
    df2.loc[df2.sample(frac=0.25).index, col] = pd.np.nan

#convert this dataset to numpy
data_x2 = df2.to_numpy()

# Parameters
no2, dim2 = data_x2.shape

batch_size2 = 32
hint_rate2 = 0.5 
alpha2 = 1
iterations2 = 50000

# Introduce missing data
data_m2 = np.ones([no2, dim2])
for i in range(no2):
    for j in range(dim2):
        if np.isnan(data_x2[i,j]):
            data_m2[i,j] = 0
miss_data_x2 = data_x2.copy()
miss_data_x2[data_m2 == 0] = np.nan


def normalization2(data):
  # Parameters
  _, dim = data.shape
  norm_data2 = data.copy()
  
  # MixMax normalization
  min_val2 = np.zeros(dim)
  max_val2 = np.zeros(dim)
  
  # For each dimension
  for i in range(dim):
    min_val2[i] = np.nanmin(norm_data2[:,i])
    norm_data2[:,i] = norm_data2[:,i] - np.nanmin(norm_data2[:,i])
    max_val2[i] = np.nanmax(norm_data2[:,i])
    norm_data2[:,i] = norm_data2[:,i] / (np.nanmax(norm_data2[:,i]) + 1e-6)   
    
  # Return norm_parameters for renormalization
  norm_parameters2 = {'min_val2': min_val2,
                     'max_val2': max_val2}
      
  return norm_data2, norm_parameters2

def renormalization2(norm_data2, norm_parameters2):
  min_val2 = norm_parameters2['min_val2']
  max_val2 = norm_parameters2['max_val2']

  _, dim = norm_data2.shape
  renorm_data2 = norm_data2.copy()
    
  for i in range(dim):
    renorm_data2[:,i] = renorm_data2[:,i] * (max_val2[i] + 1e-6)   
    renorm_data2[:,i] = renorm_data2[:,i] + min_val2[i]
    
  return renorm_data2

def binary_sampler2(p, rows, cols):
  unif_random_matrix2 = np.random.uniform(0., 1., size = [rows, cols])
  binary_random_matrix2 = 1*(unif_random_matrix2 < p)
  return binary_random_matrix2

def uniform_sampler2(low, high, rows, cols):
  return np.random.uniform(low, high, size = [rows, cols])   

def rounding2(imputed_data, data_x):
  _, dim = data_x2.shape
  rounded_data2 = imputed_data.copy()
  
  for i in range(dim):
    temp2 = data_x2[~np.isnan(data_x2[:, i]), i]
    # Only for the categorical variable
    if len(np.unique(temp2)) < 20:
      rounded_data2[:, i] = np.round(rounded_data2[:, i])
      
  return rounded_data2

def xavier_init2(size):
  in_dim = size[0]
  xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
  return tf.random_normal(shape = size, stddev = xavier_stddev)

def sample_batch_index2(total, batch_size):
  total_idx = np.random.permutation(total)
  batch_idx2 = total_idx[:batch_size]
  return batch_idx2


# Hidden state dimensions
h_dim2 = int(dim2)
  
# Normalization
#data_x = np.array(data_x)
data_x2 = data_x2.astype(np.float)
norm_data2, norm_parameters2 = normalization2(data_x2)
norm_data_x2 = np.nan_to_num(norm_data2, 0)
  
## GAIN architecture   
# Input placeholders
# Data vector
X2 = tf.placeholder(tf.float32, shape = [None, dim2])
# Mask vector 
M2 = tf.placeholder(tf.float32, shape = [None, dim2])
# Hint vector
H2 = tf.placeholder(tf.float32, shape = [None, dim2])
  
# Discriminator variables
D_W12 = tf.Variable(xavier_init([dim2*2, h_dim2])) # Data + Hint as inputs
D_b12 = tf.Variable(tf.zeros(shape = [h_dim2]))
  
D_W22 = tf.Variable(xavier_init([h_dim2, h_dim2]))
D_b22 = tf.Variable(tf.zeros(shape = [h_dim2]))
  
D_W32 = tf.Variable(xavier_init([h_dim2, dim2]))
D_b32 = tf.Variable(tf.zeros(shape = [dim2]))  # Multi-variate outputs
  
theta_D2 = [D_W12, D_W22, D_W32, D_b12, D_b22, D_b32]
  
#Generator variables
# Data + Mask as inputs (Random noise is in missing components)
G_W12 = tf.Variable(xavier_init([dim2*2, h_dim2]))  
G_b12 = tf.Variable(tf.zeros(shape = [h_dim2]))
  
G_W22 = tf.Variable(xavier_init([h_dim2, h_dim2]))
G_b22 = tf.Variable(tf.zeros(shape = [h_dim2]))
  
G_W32 = tf.Variable(xavier_init([h_dim2, dim2]))
G_b32 = tf.Variable(tf.zeros(shape = [dim2]))
  
theta_G2 = [G_W12, G_W22, G_W32, G_b12, G_b22, G_b32]
  
## GAIN functions
# Generator
def generator(x,m):
    # Concatenate Mask and Data
    inputs = tf.concat(values = [x, m], axis = 1) 
    G_h12 = tf.nn.relu(tf.matmul(inputs, G_W12) + G_b12)
    G_h22 = tf.nn.relu(tf.matmul(G_h12, G_W22) + G_b22)   
    # MinMax normalized output
    G_prob2 = tf.nn.sigmoid(tf.matmul(G_h22, G_W32) + G_b32) 
    return G_prob2
  
# Discriminator
def discriminator(x, h):
    # Concatenate Data and Hint
    inputs = tf.concat(values = [x, h], axis = 1) 
    D_h12 = tf.nn.relu(tf.matmul(inputs, D_W12) + D_b12)  
    D_h22 = tf.nn.relu(tf.matmul(D_h12, D_W22) + D_b22)
    D_logit2 = tf.matmul(D_h22, D_W32) + D_b32
    D_prob2 = tf.nn.sigmoid(D_logit2)
    return D_prob2
  
## GAIN structure
# Generator
G_sample2 = generator(X2, M2)
 
# Combine with observed data
Hat_X2 = X2 * M2 + G_sample2 * (1-M2)
  
# Discriminator
D_prob2 = discriminator(Hat_X2, H2)
  
## GAIN loss
D_loss_temp2 = -tf.reduce_mean(M2 * tf.log(D_prob2 + 1e-8) + (1-M2) * tf.log(1. - D_prob2 + 1e-8)) 
  
G_loss_temp2 = -tf.reduce_mean((1-M2) * tf.log(D_prob2 + 1e-8))
  
MSE_loss2 = tf.reduce_mean((M2 * X2 - M2 * G_sample2)**2) / tf.reduce_mean(M2)
  
D_loss2 = D_loss_temp2
G_loss2 = G_loss_temp2 + alpha2 * MSE_loss2 
  
## GAIN solver
D_solver2 = tf.train.AdamOptimizer().minimize(D_loss2, var_list=theta_D2)
G_solver2 = tf.train.AdamOptimizer().minimize(G_loss2, var_list=theta_G2)
  
## Iterations
sess2 = tf.Session()
sess2.run(tf.global_variables_initializer())
   
D2 = np.array(range(0,iterations2), dtype = "float64")
G2 = np.array(range(0,iterations2), dtype = "float64")

# Start Iterations
for it in tqdm(range(iterations2)):    
      
    # Sample batch
    batch_idx2 = sample_batch_index(no2, batch_size2)
    X_mb2 = norm_data_x2[batch_idx2, :]  
    M_mb2 = data_m2[batch_idx2, :]  
    # Sample random vectors  
    Z_mb2 = uniform_sampler(0, 1, batch_size2, dim2) 
    # Sample hint vectors
    H_mb_temp2 = binary_sampler(hint_rate2, batch_size2, dim2)
    H_mb2 = M_mb2 * H_mb_temp2
      
    # Combine random vectors with observed vectors
    X_mb2 = M_mb2 * X_mb2 + (1-M_mb2) * Z_mb2 
      
    _, D_loss_curr2 = sess2.run([D_solver2, D_loss_temp2], 
                              feed_dict = {M2: M_mb2, X2: X_mb2, H2: H_mb2})
    _, G_loss_curr2, MSE_loss_curr2 = sess2.run([G_solver2, G_loss_temp2, MSE_loss2],
             feed_dict = {X2: X_mb2, M2: M_mb2, H2: H_mb2})
    
    D2[it] = D_loss_curr2
    G2[it] = G_loss_curr2

x = range(iterations2)
plt.title("Loss plot")
plt.plot(x,D2,color = 'blue',linewidth=1, label = 'D loss')
plt.plot(x,G2,color = 'red',linewidth=1, label = 'G loss')
plt.legend()
plt.xlabel("Iters")
plt.ylabel("loss")
plt.show()

#Lifestyle imputations
           
## Return imputed data    
#load testing data
#test2 =pd.read_csv(r'C:UsersadoganOneDrive - Oklahoma A and M SystemClassifiedCodes2FATCHDFATCHD_5DC10DC_UH.csv')
X_forTest = X_test.copy()
y_forTest = y_test.copy()
X_forTest['CLASS'] = y_forTest
test2 = X_forTest.copy()

#drop the class which is output
def drop_class(data_test):
    data_test = data_test.drop(columns=['CLASS'], axis = 1)
    return data_test
random_state = np.random.RandomState(0)
n_samples, n_features = X_train.shape


from sklearn.ensemble import RandomForestClassifier
import statistics
from sklearn.utils import resample
'''
Expectation
'''
#Training classifier for the risk calculation
#rf = RandomForestClassifier(max_features=5, n_estimators=500,random_state=0)
rf = RandomForestClassifier(n_estimators=10000, criterion='entropy', max_depth=None, min_samples_split=200, min_samples_leaf=20, 
                            min_weight_fraction_leaf=0.0, max_features=5, max_leaf_nodes=None, 
                            min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, 
                            n_jobs=None, random_state=42, verbose=0, warm_start=True, class_weight=None, 
                            ccp_alpha=0.0, max_samples=945)
#rf.fit(X_train, y_train)
# Separate majority and minority classes
df_majority = X_forTrain[X_forTrain.CLASS==0]
df_minority = X_forTrain[X_forTrain.CLASS==1]
 
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=df_majority.shape[0],    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
df_upsampled = df_upsampled #.drop(columns=['HDLSIU02', 'LDLSIU02', 'TCHSIU01','TRGSIU01', 'SBPA21', 'SBPA22', 'ANTA07A', 'ANTA07B', 'ECGMA31',
       #'HMTA03','APASIU01', 'APBSIU01', 'LIPA08', 'CHMA09'], axis = 1)
# 1    576
# 0    576
# Name: balance, dtype: int64

# Separate input features (X) and target variable (y)
y_train = df_upsampled.CLASS
X_train= df_upsampled.drop('CLASS', axis=1)
rf.fit(X_train, y_train)
#risk calculation function
def calc_Expectation(X_test): 
    rf_probs_2 = rf.predict_proba(X_test)
    rf_probs = rf_probs_2[:, 1].mean()
    print(rf_probs)
    return rf_probs
import math

#Defining the utility function
new_lambda=0 #lambda value to calculation of the life style changes 
def utility(r1):
    lambda_0 = 0.0000001 #lambda value for the personal references 
    #total lambda 
    lambda_main = lambda_0+ new_lambda
    uti = ((1 - math.exp(-lambda_main *(1- r1) ))/lambda_main)
    return uti
    
#Cost of the 1 change in lifestyles:
#'BMI01', 'ETHANL03', 'SMKSTA', 'MOVE', 'CARB', 'CHOL', 'DFIB','PROT', 'SFAT', 'TFAT', 'TOTCAL03' (respectively)
coefficients = pd.DataFrame([0.01,	0.0005,	0.001,	0.001,	0.0005,	0.001,	0.001,	0.0005,	0.0005,	0.0005, 0.0005])


#test2=pd.read_csv(r'C:\Users\adogan\OneDrive - Oklahoma A and M System\Classified\Codes2\FATCHD\test.csv')
#The variables that needs to be imputed by the model to get lifestyle recommendations
#Directly changable variables with the  Indirectly changable variables
ind_dir_feat = ['BMI01', 'ETHANL03', 'SMKSTA', 'MOVE', 'CARB', 'CHOL', 'DFIB',
       'PROT', 'SFAT', 'TFAT', 'TOTCAL03','CLASS']
#loading data and defining variables before the loop
a = 0
z = 0
New_R1  = list()
New_R0  = list()
New_R0L = list()
New_R1L = list()
min_risk = list()
max_risk = list()

recommendedPatients = list() 
notRecommended = list()
whole_rec = pd.DataFrame()
whole_ic = pd.DataFrame()
summary_all = pd.DataFrame()
p_selected = list()
#test2_whole = test2 =pd.read_csv(r'C:\Users\adogan\OneDrive - Oklahoma A and M System\Classified\Codes2\FATCHD\NonRecommended35.csv')

#1st loop to create lifestyles that are suitable to constraints 
sim = pd.DataFrame()
#test2 = test2_whole.iloc[:,6:]
for z in range(test2.shape[0]):#test2.shape[0]
    recommendations = pd.DataFrame()
    discrete = pd.DataFrame()
    pat = pd.DataFrame()
    test = pd.DataFrame()
    Z_mb = pd.DataFrame()
    M_mb = pd.DataFrame() #0, 1 matrix generate 
    X_mb = pd.DataFrame() 
       
    imputed_data = pd.DataFrame()
    df = pd.DataFrame()
    X_new = pd.DataFrame()
    X_new1 = pd.DataFrame()
    ori = pd.DataFrame()
    ori = pd.DataFrame(test2.iloc[z:z+1])
    ori_direct = ori.copy()
    if ori.iloc[:,-1].item() == 1:
        pat = pd.concat([ori]*500, ignore_index=True)
        pat_new = pat.drop(columns=['CLASS'], axis = 1)
        rf_probs = rf.predict_proba(pat_new)[:, 1]
        
        tot = 0
        new_lambda = 0
        for i in range(len(rf_probs)):
            tot = tot + utility(rf_probs[i])
        #print("Mean value of the Utility: " , tot/len(rf_probs))
            
        # Display expectation of given array 
        expect_ori = calc_Expectation(pat_new)
        var = statistics.variance(rf_probs, xbar=None)
        uti = tot/len(rf_probs)
    
        cols = ['utility','expectation', 'variance','minimum','maximum',
                'BMI01', 'ETHANL03', 'SMKSTA', 'MOVE', 'CARB', 'CHOL', 
                'DFIB','PROT', 'SFAT', 'TFAT', 'TOTCAL03','HDLSIU02', 'LDLSIU02', 'TCHSIU01','TRGSIU01', 'SBPA21', 'SBPA22', 'ANTA07A', 'ANTA07B', 'ECGMA31',
                'HMTA03','APASIU01', 'APBSIU01', 'LIPA08', 'CHMA09',	'HYPTMD01','ANTICOAGCODE01','ASPIRINCODE01','STATINCODE01',
                "HYPTMDCODE01",	"HYPERT04",	"CIGTYR01",
                "V1AGE01",	"RACEGRP",	"DIABTS02",	"ELEVEL01",	"GENDER",	"CIGT01",	"ANTA01",	"CHOLMDCODE01",	"CLASS"]
        new_row = []
        new_row = [uti, expect_ori, var, min(rf_probs),max(rf_probs),ori.iloc[0,0],ori.iloc[0,1],
              ori.iloc[0,2],ori.iloc[0,3],ori.iloc[0,4],ori.iloc[0,5],ori.iloc[0,6],ori.iloc[0,7],ori.iloc[0,8],
              ori.iloc[0,9],ori.iloc[0,10],ori.iloc[0,11],ori.iloc[0,12],ori.iloc[0,13],ori.iloc[0,14],ori.iloc[0,15],
              ori.iloc[0,16],ori.iloc[0,17],ori.iloc[0,18],ori.iloc[0,19],ori.iloc[0,20],ori.iloc[0,21],ori.iloc[0,22],
              ori.iloc[0,23],ori.iloc[0,24],ori.iloc[0,25],ori.iloc[0,26],ori.iloc[0,27],ori.iloc[0,28],ori.iloc[0,29],
              ori.iloc[0,30],ori.iloc[0,31],ori.iloc[0,32],ori.iloc[0,33],ori.iloc[0,34], ori.iloc[0,35],ori.iloc[0,36],
              ori.iloc[0,37], ori.iloc[0,38], ori.iloc[0,39], ori.iloc[0,40]]
        values = pd.DataFrame(new_row).T# .append(new_row,ignore_index=True)
        values.columns = cols
        new1 = pd.DataFrame()
        new_large =  pd.DataFrame()
        ori_direct = ori_direct.drop(columns=['HDLSIU02', 'LDLSIU02', 'TCHSIU01','TRGSIU01', 'SBPA21', 'SBPA22', 'ANTA07A', 'ANTA07B', 'ECGMA31',
           'HMTA03','APASIU01', 'APBSIU01', 'LIPA08', 'CHMA09'], axis = 1)
        #test = test.append(pat)
        test = pat.copy()
        for col in ind_dir_feat:
            ori_direct.loc[ori_direct.sample(frac=1).index, col] = pd.np.nan
        test = pd.concat([ori_direct]*500, ignore_index=True)
        test = test.to_numpy()
        #test = test1.to_numpy()
        no, dim = test.shape
        data_m = np.ones([no, dim])
        for i in range(no):
            for j in range(dim):
                if np.isnan(test[i,j]):
                    data_m[i,j] = 0
        # Parameters
        _, dim = test.shape
        norm_data1 = test.copy()
        #  # Return norm_parameters for renormalization
        min_val=norm_parameters['min_val']
        max_val=norm_parameters['max_val']
        
        #  # MixMax normalization
        #min_val = np.zeros(dim)
        #max_val = np.zeros(dim)
          
          # For each dimension
        for i in range(dim):
        
            norm_data1[:,i] = norm_data1[:,i] - min_val[i]
        
            norm_data1[:,i] = norm_data1[:,i] / (max_val[i] + 1e-6)   
        
        #  # Return norm_parameters for renormalization
        
          
        
        #        else:
        #            test[i,j]= norm_data[33,j]
        #norm_data, norm_parameters = normalization()
        norm_data_test = np.nan_to_num(norm_data1, 0)
        #Z_mb = uniform_sampler(0, 0.1, no, dim) 
        #fix the parts that has no values 
        Z_mb = uniform_sampler(0,1,no,dim) # ,random_state = 0)
        M_mb = data_m #0, 1 matrix generate 
        X_mb = norm_data_test          
        X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
        #
        
        imputed_data = sess.run([G_sample], feed_dict = {X: X_mb, M: M_mb})[0]
          
        imputed_data = data_m * norm_data_test + (1-data_m) * imputed_data
          
        # Renormalization
        imputed_data = renormalization(imputed_data, norm_parameters)  
          
          
        # Rounding for final imputed data
        imputed_data = rounding(imputed_data, data_x)  
        
        #"ID_C",
        #columns = [ 	"BMI01",	"ETHANL03",	"CIGT01",	"MOVE",	"P_CARB",	"CHOL",	"DFIB",	"P_PROT",	"P_SFAT",	"P_TFAT",	"CHOLMDCODE01",	"HDLSIU02",	"LDLSIU02",	"TCHSIU01",	"TRGSIU01",	"SBPA21",	"SBPA22",	"ANTA01",	"ANTA07A",	"ANTA07B",	"ECGMA31",	"HMTA03",	"APASIU01",	"APBSIU01",	"LIPA08",	"CHMA09",	"HYPTMDCODE01",	"HYPERT04",	"CIGTYR01",	"V1AGE01",	"RACEGRP",	"DIABTS02",	"ELEVEL01",	"GENDER",	"CLASS"
        #]
        #df = pd.DataFrame(data=imputed_data,columns = columns)
        df = pd.DataFrame(data=imputed_data)
        #df[1] = df[1].replace({1.0:0.0})
        #df.replace(df[2],0)
        columns = [	"BMI01",	"ETHANL03",	"SMKSTA",	"MOVE",	"CARB",	"CHOL",	"DFIB",	"PROT",	"SFAT",	
                   "TFAT", "TOTCAL03",'HYPTMD01','ANTICOAGCODE01','ASPIRINCODE01','STATINCODE01',	"HYPTMDCODE01",	"HYPERT04",	"CIGTYR01",
                   "V1AGE01",	"RACEGRP",	"DIABTS02",	"ELEVEL01",	"GENDER",	"CIGT01",	"ANTA01",	"CHOLMDCODE01",	"CLASS"] #,	"Smoke#", "SmokeSta",
        
        df.columns = columns 
        ##df = df.drop(columns = ["ID_C"], axis = 1)
        df.to_csv('Recs.csv')
    
        #df = df.drop(columns = ['ID_C'], axis = 1)
        X_new = pd.DataFrame()  
        X_new = X_new.append(df.head(0), ignore_index = False)
        #newdata = pd.DataFrame()
        #df.iloc[i, 0] <= test2['BMI01'].mean() and and df.iloc[i,6] >= test2['DFIB'].mean()
        a = 0
        for i in range(_):
            if ori['BMI01'].mean() <= 2:
                if ori['ETHANL03'].mean() <= 1:
                    if ori['SMKSTA'].mean() == 1:
                        if ori['CHOL'].mean() <= 4:
                            if ori['DFIB'].mean() >= 5 :
                                if ori['SFAT'].mean() <= 3:
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= 5  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint1')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= 5  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint2')
                                else: #for SFAT
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= 5  and df.iloc[i,8] <= ori['SFAT'].mean() and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint3')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= 5  and df.iloc[i,8] <= ori['SFAT'].mean() and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint4')
                            else: #for DFIB
                                if ori['SFAT'].mean() <= 3:
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint5')                                           
                                    else: # for TFAT
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint6')
                                else: #for SFAT
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >=ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint7')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint8')                           
                        else: #for CHOL
                            if ori['DFIB'].mean() >= 5 :
                                if ori['SFAT'].mean() <= 3:
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= 5  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint9')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= 5  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint10')
                                else: #for SFAT
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= 5  and df.iloc[i,8] <= ori['SFAT'].mean() and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint11')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= 5  and df.iloc[i,8] <= ori['SFAT'].mean() and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint12')
                            else: #for DFIB
                                if ori['SFAT'].mean() <= 3:
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint13')                                           
                                    else: # for TFAT
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint14')
                                else: #for SFAT
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >=ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint15')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint16')                         
                    else: # for SMKSTA   #if ori['SMKSTA'].mean() == 1:
                        if ori['CHOL'].mean() <= 4:
                            if ori['DFIB'].mean() >= 5 :
                                if ori['SFAT'].mean() <= 3:
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= 5  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint17')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= 5  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint18')
                                else: #for SFAT
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= 5  and df.iloc[i,8] <= ori['SFAT'].mean() and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint19')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= 5  and df.iloc[i,8] <= ori['SFAT'].mean() and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint20')
                            else: #for DFIB
                                if ori['SFAT'].mean() <= 3:
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint21')                                           
                                    else: # for TFAT
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint22')
                                else: #for SFAT
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >=ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint23')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint24')                           
                        else: #for CHOL
                            if ori['DFIB'].mean() >= 5 :
                                if ori['SFAT'].mean() <= 3:
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= 5  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint25')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= 5  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint26')
                                else: #for SFAT
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= 5  and df.iloc[i,8] <= ori['SFAT'].mean() and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint27')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= 5  and df.iloc[i,8] <= ori['SFAT'].mean() and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint28')
                            else: #for DFIB
                                if ori['SFAT'].mean() <= 3:
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint29')                                           
                                    else: # for TFAT
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint30')
                                else: #for SFAT
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >=ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint31')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint32')                                              
                else: #for ETHANOL03   if ori['ETHANL03'].mean() <= 1:
                    if ori['SMKSTA'].mean() == 1:
                        if ori['CHOL'].mean() <= 4:
                            if ori['DFIB'].mean() >= 5 :
                                if ori['SFAT'].mean() <= 3:
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= 5  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint33')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= 5  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint34')
                                else: #for SFAT
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= 5  and df.iloc[i,8] <= ori['SFAT'].mean() and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint35')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= 5  and df.iloc[i,8] <= ori['SFAT'].mean() and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint36')
                            else: #for DFIB
                                if ori['SFAT'].mean() <= 3:
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint37')                                           
                                    else: # for TFAT
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint38')
                                else: #for SFAT
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >=ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint39')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint40')                           
                        else: #for CHOL
                            if ori['DFIB'].mean() >= 5 :
                                if ori['SFAT'].mean() <= 3:
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= 5  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint41')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= 5  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint42')
                                else: #for SFAT
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= 5  and df.iloc[i,8] <= ori['SFAT'].mean() and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint43')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= 5  and df.iloc[i,8] <= ori['SFAT'].mean() and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint44')
                            else: #for DFIB
                                if ori['SFAT'].mean() <= 3:
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint45')                                           
                                    else: # for TFAT
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint46')
                                else: #for SFAT
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >=ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint47')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint48')                         
                    else: # for SMKSTA   #if ori['SMKSTA'].mean() == 1:
                        if ori['CHOL'].mean() <= 4:
                            if ori['DFIB'].mean() >= 5 :
                                if ori['SFAT'].mean() <= 3:
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= 5  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint49')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= 5  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint50')
                                else: #for SFAT
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= 5  and df.iloc[i,8] <= ori['SFAT'].mean() and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint51')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= 5  and df.iloc[i,8] <= ori['SFAT'].mean() and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint52')
                            else: #for DFIB
                                if ori['SFAT'].mean() <= 3:
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint53')                                           
                                    else: # for TFAT
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint54')
                                else: #for SFAT
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >=ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint55')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint56')                           
                        else: #for CHOL
                            if ori['DFIB'].mean() >= 5 :
                                if ori['SFAT'].mean() <= 3:
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= 5  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint57')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= 5  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint58')
                                else: #for SFAT
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= 5  and df.iloc[i,8] <= ori['SFAT'].mean() and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint59')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= 5  and df.iloc[i,8] <= ori['SFAT'].mean() and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint60')
                            else: #for DFIB
                                if ori['SFAT'].mean() <= 3:
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint61')                                           
                                    else: # for TFAT
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint62')
                                else: #for SFAT
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >=ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint63')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= 2 and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint64')                                  
            else: #for BMI01 if ori['BMI01'].mean() <= 2:
                if ori['ETHANL03'].mean() <= 1:
                    if ori['SMKSTA'].mean() == 1:
                        if ori['CHOL'].mean() <= 4:
                            if ori['DFIB'].mean() >= 5 :
                                if ori['SFAT'].mean() <= 3:
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= 5  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint1b')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= 5  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint2b')
                                else: #for SFAT
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= 5  and df.iloc[i,8] <= ori['SFAT'].mean() and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint3b')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= 5  and df.iloc[i,8] <= ori['SFAT'].mean() and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint4b')
                            else: #for DFIB
                                if ori['SFAT'].mean() <= 3:
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint5b')                                           
                                    else: # for TFAT
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint6b')
                                else: #for SFAT
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >=ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint7b')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint8b')                           
                        else: #for CHOL
                            if ori['DFIB'].mean() >= 5 :
                                if ori['SFAT'].mean() <= 3:
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= 5  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint9b')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= 5  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint10b')
                                else: #for SFAT
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= 5  and df.iloc[i,8] <= ori['SFAT'].mean() and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint11b')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= 5  and df.iloc[i,8] <= ori['SFAT'].mean() and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint12b')
                            else: #for DFIB
                                if ori['SFAT'].mean() <= 3:
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint13b')                                           
                                    else: # for TFAT
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint14b')
                                else: #for SFAT
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >=ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint15b')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint16b')                         
                    else: # for SMKSTA   #if ori['SMKSTA'].mean() == 1:
                        if ori['CHOL'].mean() <= 4:
                            if ori['DFIB'].mean() >= 5 :
                                if ori['SFAT'].mean() <= 3:
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= 5  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint17b')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= 5  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint18b')
                                else: #for SFAT
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= 5  and df.iloc[i,8] <= ori['SFAT'].mean() and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint19b')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= 5  and df.iloc[i,8] <= ori['SFAT'].mean() and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint20b')
                            else: #for DFIB
                                if ori['SFAT'].mean() <= 3:
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint21b')                                           
                                    else: # for TFAT
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint22b')
                                else: #for SFAT
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >=ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint23b')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint24b')                           
                        else: #for CHOL
                            if ori['DFIB'].mean() >= 5 :
                                if ori['SFAT'].mean() <= 3:
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= 5  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint25b')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= 5  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint26b')
                                else: #for SFAT
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= 5  and df.iloc[i,8] <= ori['SFAT'].mean() and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint27b')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= 5  and df.iloc[i,8] <= ori['SFAT'].mean() and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint28b')
                            else: #for DFIB
                                if ori['SFAT'].mean() <= 3:
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint29b')                                           
                                    else: # for TFAT
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint30b')
                                else: #for SFAT
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >=ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint31b')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= 1 and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint32b')                                              
                else: #for ETHANOL03   if ori['ETHANL03'].mean() <= 1:
                    if ori['SMKSTA'].mean() == 1:
                        if ori['CHOL'].mean() <= 4:
                            if ori['DFIB'].mean() >= 5 :
                                if ori['SFAT'].mean() <= 3:
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= 5  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint33b')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= 5  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint34b')
                                else: #for SFAT
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= 5  and df.iloc[i,8] <= ori['SFAT'].mean() and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint35b')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= 5  and df.iloc[i,8] <= ori['SFAT'].mean() and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint36b')
                            else: #for DFIB
                                if ori['SFAT'].mean() <= 3:
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint37b')                                           
                                    else: # for TFAT
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint38b')
                                else: #for SFAT
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >=ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint39b')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint40b')                           
                        else: #for CHOL
                            if ori['DFIB'].mean() >= 5 :
                                if ori['SFAT'].mean() <= 3:
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= 5  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint41b')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= 5  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint42b')
                                else: #for SFAT
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= 5  and df.iloc[i,8] <= ori['SFAT'].mean() and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint43b')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= 5  and df.iloc[i,8] <= ori['SFAT'].mean() and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint44b')
                            else: #for DFIB
                                if ori['SFAT'].mean() <= 3:
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint45b')                                           
                                    else: # for TFAT
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint46b')
                                else: #for SFAT
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >=ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint47b')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] <= 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint48b')                         
                    else: # for SMKSTA   #if ori['SMKSTA'].mean() == 1:
                        if ori['CHOL'].mean() <= 4:
                            if ori['DFIB'].mean() >= 5 :
                                if ori['SFAT'].mean() <= 3:
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= 5  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint49b')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= 5  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint50b')
                                else: #for SFAT
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= 5  and df.iloc[i,8] <= ori['SFAT'].mean() and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint51b')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= 5  and df.iloc[i,8] <= ori['SFAT'].mean() and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint52b')
                            else: #for DFIB
                                if ori['SFAT'].mean() <= 3:
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint53b')                                           
                                    else: # for TFAT
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint54b')
                                else: #for SFAT
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >=ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint55b')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= 4 and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint56b')                           
                        else: #for CHOL
                            if ori['DFIB'].mean() >= 5 :
                                if ori['SFAT'].mean() <= 3:
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= 5  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint57b')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= 5  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint58b')
                                else: #for SFAT
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= 5  and df.iloc[i,8] <= ori['SFAT'].mean() and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint59b')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= 5  and df.iloc[i,8] <= ori['SFAT'].mean() and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint60b')
                            else: #for DFIB
                                if ori['SFAT'].mean() <= 3:
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint61b')                                           
                                    else: # for TFAT
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint62b')
                                else: #for SFAT
                                    if ori['TFAT'].mean() <= 4:
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >=ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= 4:
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint63b')                                           
                                    else: #for TFAT
                                        if df.iloc[i, 0] <= ori['BMI01'].mean() and df.iloc[i, 1]  <= ori['ETHANL03'].mean() and df.iloc[i, 2] < 1 and df.iloc[i,3] >= ori['MOVE'].mean() and df.iloc[i,5] <= ori['CHOL'].mean() and df.iloc[i,6] >= ori['DFIB'].mean()  and df.iloc[i,8] <= 3 and df.iloc[i,9] <= ori['TFAT'].mean():
                                            lifestyle = pd.DataFrame(df.iloc[i:i+1])
                                            #print(lifestyle)
                                            X_new = X_new.append(lifestyle)
                                            #print('Constraint64b')   
#        print(X_new)
        if X_new.shape[0] == 1:
            X_new1 = X_new
        else:
            for j in range(X_new.shape[0]-1):
                for t in range(X_new.shape[0]-(j+1)):  
                    line = pd.DataFrame(np.asarray(X_new.iloc[j,0:11])) 
                    line2 = pd.DataFrame(np.asarray(X_new.iloc[t+1,0:11])) 
                    if line.equals(line2) == True:        
                        X_new1 =X_new.drop(X_new.index[t+1])
                        j = j+1
                        t =t+1
                    elif t == 0:
                        X_new1 = X_new                    
                    
                
    
#        print(X_new1)
        #print(df.iloc[i])
        X_new1.to_csv('Cond_Recs'+str(z)+'.csv')
#        print(X_new1.shape)
    
        ver, hor = X_new1.shape
        if ver == 0: 
            notRecommended.append(z)
#            print('a')
            continue
        #if ver != 0:    
        recommendedPatients.append(z)
#        print('b')    
    #convert the data size to 41
        #test2 = test2.iloc[:, 1:]
        def listmaker(n,i):
            listofzeros = [i] * n
            return listofzeros
        ic_feat = ['HDLSIU02', 'LDLSIU02', 'TCHSIU01',
               'TRGSIU01', 'SBPA21', 'SBPA22', 'ANTA07A', 'ANTA07B', 'ECGMA31', 'HMTA03',
               'APASIU01', 'APBSIU01', 'LIPA08', 'CHMA09']
        ic_feat.reverse()
        val_nan = pd.DataFrame(listmaker(int(X_new1.shape[0]),0))
        for val in val_nan:
            val_nan.loc[val_nan.sample(frac=1).index, val] = pd.np.nan
        for col in ic_feat:
            X_new1.insert(11, col,val_nan)
        p_list = list()
        t_list = list()    
    #IC Imputations for Recommended lifestyles 
        for k in range(X_new1.shape[0]):#X_new1.shape[0]

            recommended = pd.DataFrame()
            recommend = pd.DataFrame()
            recommend = pd.DataFrame(X_new1.iloc[k:k+1])
            recommend_ind =recommend.copy() 
            ind_feat = [ 'HDLSIU02', 'LDLSIU02', 'TCHSIU01',
                   'TRGSIU01', 'SBPA21', 'SBPA22', 'ANTA07A', 'ANTA07B', 'ECGMA31', 'HMTA03',
                   'APASIU01', 'APBSIU01', 'LIPA08', 'CHMA09','CLASS']
            recommended = pd.concat([recommend_ind]*500, ignore_index=True)
            test = recommended.to_numpy()
            #test = test1.to_numpy()
            no2, dim2 = test.shape
            data_m2 = np.ones([no2, dim2])
            for i in range(no2):
                for j in range(dim2):
                    if np.isnan(test[i,j]):
                        data_m2[i,j] = 0
         # Parameters
            _, dim2 = test.shape
            norm_data12 = test.copy()
            #  # Return norm_parameters for renormalization
            min_val2=norm_parameters2['min_val2']
            max_val2=norm_parameters2['max_val2']
            
              # For each dimension
            for i in range(dim2):
            
                norm_data12[:,i] = norm_data12[:,i] - min_val2[i]
            
                norm_data12[:,i] = norm_data12[:,i] / (max_val2[i] + 1e-6)   
            
            #  # Return norm_parameters for renormalization
            norm_data_test2 = np.nan_to_num(norm_data12, 0)
            #Z_mb = uniform_sampler(0, 0.1, no, dim) 
            #fix the parts that has no values 
            Z_mb2 = uniform_sampler(0,1,no2,dim2) # ,random_state = 0)
            M_mb2 = data_m2 #0, 1 matrix generate 
            X_mb2 = norm_data_test2          
            X_mb2 = M_mb2 * X_mb2 + (1-M_mb2) * Z_mb2 
            #
            
            imputed_data2 = sess2.run([G_sample2], feed_dict = {X2: X_mb2, M2: M_mb2})[0]
              
            imputed_data2 = data_m2 * norm_data_test2 + (1-data_m2) * imputed_data2
              
            # Renormalization
            imputed_data2 = renormalization2(imputed_data2, norm_parameters2)  
              
              
            # Rounding for final imputed data
            imputed_data2 = rounding2(imputed_data2, data_x2)  
            df = pd.DataFrame(data=imputed_data2)
            columns = [	"BMI01",	"ETHANL03",	"SMKSTA",	"MOVE",	"CARB",	"CHOL",	"DFIB",	"PROT",	"SFAT",	
                   "TFAT", "TOTCAL03","HDLSIU02",	"LDLSIU02",	"TCHSIU01",	"TRGSIU01",	"SBPA21",	"SBPA22",	"ANTA07A",	"ANTA07B",	
                   "ECGMA31",	"HMTA03",	"APASIU01",	"APBSIU01",	"LIPA08",	"CHMA09",	'HYPTMD01','ANTICOAGCODE01','ASPIRINCODE01','STATINCODE01',
                   "HYPTMDCODE01",	"HYPERT04",	"CIGTYR01",
                   "V1AGE01",	"RACEGRP",	"DIABTS02",	"ELEVEL01",	"GENDER",	"CIGT01",	"ANTA01",	"CHOLMDCODE01",	"CLASS"]
            df.columns = columns 
            ic_imput = pd.concat([ori,df],ignore_index=False, sort =False )
            X_test_out = ic_imput.drop(columns =['CLASS'], axis=1)
            X_test_output =pd.DataFrame()
            X_test_output['Risk'] =rf.predict_proba(X_test_out)[:, 1]
            rf_probs = pd.DataFrame(rf.predict_proba(X_test_out)[:, 1])
            ic_imput.index = X_test_output.index
            X_test_out2 = pd.concat([X_test_output,ic_imput],axis=1, sort=False )
            #X_test_out2.to_csv('IC_imputation_Rec'+str(z)+str(k)+'.csv')
            X_test = df.drop(columns =['CLASS'], axis=1)
            differences = pd.DataFrame()
            lambda_1 = pd.DataFrame()
            #to calculate the lamda the differences btw. recommended lifestyle and original 
            #is discritized to measure the differences
            dif_recommend =pd.DataFrame() 
            #dif_recommend = dif_recommend.append(recommend)
            dif_recommend = recommend.copy()
      
            dif_ori =pd.DataFrame() 
            dif_ori = dif_ori.append(ori)
     
            differences = pd.DataFrame(dif_ori.iloc[0,0:11]-dif_recommend.iloc[0,0:11])      
            #coefficients.index = columns[:10]
            
            #differences = pd.DataFrame(ori.iloc[0,0:10]-recommend.iloc[0,0:10])
            differences.index = [0,1,2,3,4,5,6,7,8,9,10]
            differences.columns = [0]
            lambda_1 = pd.DataFrame(coefficients[0]*differences[0])
            #new_lambda = 0
            new_lambda = 0
            for i in lambda_1.index:
                new_lambda = new_lambda + abs(lambda_1.iloc[i,0])
            
            rf_probs = rf.predict_proba(X_test)[:, 1]
            tot = 0
            for i in range(len(rf_probs)):   
                tot = tot + utility(rf_probs[i])
            #print("Mean value of the Utility: " , tot/len(rf_probs))
            expect = calc_Expectation(X_test)
            var = statistics.variance(rf_probs, xbar=None)
            # Display expectation of given array 
            uti_rec= tot/len(rf_probs)
            new_row = [uti_rec, expect, var, min(rf_probs),max(rf_probs),recommend.iloc[0,0],recommend.iloc[0,1],
                       recommend.iloc[0,2],recommend.iloc[0,3],recommend.iloc[0,4],recommend.iloc[0,5],recommend.iloc[0,6],
                       recommend.iloc[0,7],recommend.iloc[0,8],recommend.iloc[0,9],recommend.iloc[0,10],
                       df.iloc[:,11].mean(),df.iloc[:,12].mean(),df.iloc[:,13].mean(),df.iloc[:,14].mean(),df.iloc[:,15].mean(),
                       df.iloc[:,16].mean(),df.iloc[:,17].mean(),df.iloc[:,18].mean(),df.iloc[:,19].mean(),df.iloc[:,20].mean(),
                       df.iloc[:,21].mean(),df.iloc[:,22].mean(),df.iloc[:,23].mean(),df.iloc[:,24].mean(),recommend.iloc[0,25],
                       recommend.iloc[0,26],recommend.iloc[0,27],recommend.iloc[0,28],recommend.iloc[0,29],recommend.iloc[0,30],
                       recommend.iloc[0,31],recommend.iloc[0,32],recommend.iloc[0,33],recommend.iloc[0,34],recommend.iloc[0,35],
                       recommend.iloc[0,36],recommend.iloc[0,37],recommend.iloc[0,38],recommend.iloc[0,39],recommend.iloc[0,40]]#,df.iloc[:,10].mean()
            new_row = pd.DataFrame(new_row).T# .append(new_row,ignore_index=True)
            if expect_ori> expect:
                new1 = new1.append(new_row,ignore_index=True)
            #hypothesis testing
                f = X_test_out2.iloc[1:,:]
                rec_risk = f['Risk'].mean()
                ori_rec = f['Risk'].to_list()
                t_test, p = statsmodels.stats.weightstats.ztest(ori_rec, x2=None, value=expect_ori, alternative='smaller', usevar='pooled', ddof=1.0)
                t_list.append(t_test)
                p_list.append(p)
           
        if len(new1) > 0:
            cols = ['utility','expectation', 'variance','minimum','maximum',
                 'BMI01', 'ETHANL03', 'SMKSTA', 'MOVE', 'CARB', 'CHOL', 
                 'DFIB','PROT', 'SFAT', 'TFAT', 'TOTCAL03',"HDLSIU02",	"LDLSIU02",	"TCHSIU01",	"TRGSIU01",	"SBPA21",	"SBPA22",	"ANTA07A",	"ANTA07B",	
                   "ECGMA31",	"HMTA03",	"APASIU01",	"APBSIU01",	"LIPA08",	"CHMA09",'HYPTMD01','ANTICOAGCODE01','ASPIRINCODE01','STATINCODE01',
                   "HYPTMDCODE01",	"HYPERT04",	"CIGTYR01",
                   "V1AGE01",	"RACEGRP",	"DIABTS02",	"ELEVEL01",	"GENDER",	"CIGT01",	"ANTA01",	"CHOLMDCODE01",	"CLASS"] 
            new1.columns = cols
            #find the index of max utility
            rec_index = new1['utility'].idxmax() 
            p_selected.append(p_list[rec_index])
            
            #new1.to_csv('originalOrder'+str(z)+'.csv')
            new1 = new1.sort_values(by='utility',ascending=False)
            new2 = new1.copy()
            #new3 = pd.concat([values,new2], axis = 0)
            frames = [values,new2]
            new3 = pd.concat(frames,ignore_index=False, sort =False )
        
            recommendations = recommendations.append(new3,ignore_index=False)
            row = list()
            row.append('Original')
            #row.append(1)
            discrete = pd.DataFrame() 
            discrete = discrete.append(recommendations,ignore_index=False, sort = False)
            ii = 0
            for ii in range(1,discrete.shape[0]):
                str_ii = str(ii)
                num = 'R'+str_ii 
                
                row.append(num)
                
            discrete.index = row
            New_R0.append(expect_ori)
            New_R1.append(discrete.iloc[1,1])
            min_risk.append(discrete.iloc[1,3])
            max_risk.append(discrete.iloc[1,4])
            name_x = str(z)
            name1 = ("DC_Patx"+name_x+".csv")
            discrete.to_csv(name1)
            z = z +1
            print('RUN',z)
#            if len(New_R0) == 2:
#                break
        else: 
            sim = sim.append(pd.DataFrame(values))
##    
##whole_rec = pd.DataFrame()
##whole_rec = whole_rec.append(discrete,ignore_index=True)
##whole_rec.to_csv("Patients.csv")
##sim = pd.DataFrame(New_R0).T
##sim1 = sim.append(pd.DataFrame(New_R1).T)
##sim2 = sim1.append(pd.DataFrame(min_risk).T)
##sim3 = sim2.append(pd.DataFrame(max_risk).T)
sim.to_csv("NonRecommended39_2nd.csv")
##pd.DataFrame(New_R0).to_csv("Comparison_Lambda10.csv")
#     
with open('TO_New_R0.txt', 'w') as f:
    for item in New_R0:
        f.write("%s\n" % item)   
with open('TO_New_R1.txt', 'w') as f:
    for item in New_R1:
        f.write("%s\n" % item)
with open('TO_p_list.txt', 'w') as f:
    for item in p_selected:
        f.write("%s\n" % item)
with open('TO_t_list.txt', 'w') as f:
    for item in t_list:
        f.write("%s\n" % item)

