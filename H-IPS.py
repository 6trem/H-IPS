#!/usr/bin/env python
# coding: utf-8

# # H-IPS implementation
# ## Author : Leonardo Sestrem de Oliveira
# ## Date: 07/10/2021
# ## Indoor Positioning System combining Multilateration and Fingerprinting

# In[1]:


# Import section
import numpy as np
import pandas as pd
import scipy.optimize
from FP import FPest
import matplotlib.pyplot as plt
import KF_FP
import KF_MLT
from AOI import limitador


# In[2]:


# Function responsible to estimate the distance between TN and AP through the measured RSSI.
def find_distance(TNRSSIdbm):
    RSSI0dbm = -59.0    # RSSI in the d0 distance
    s = 0               # Shadowing value in dB
    n = 3.227           # Pathloss exponent value
    d0 = 1.0            # Reference distance
    largura = 9.775     # Width
    comprimento = 13.45 # Length

    dist_buffer = []
    
    for i in range(5):
        dist = 10**(-((TNRSSIdbm[i] - RSSI0dbm + s) / (10 * n)) + np.log10(d0))  # Compute the distance between TN and AP
        dist_buffer.append(dist)       
        
    return np.asarray(dist_buffer)


# In[3]:


# Calculates the euclidean distance between each AP 
# and the estimated position
def dist_fun(APsLoc, pos):
    res = []
    for i in range(len(APsLoc)):
        res.append(list((APsLoc[i, :] - pos) ** 2))

    res = np.sqrt(np.sum(np.asarray(res), axis=1))

    return res


# In[4]:


def multilateracao(APsLoc, TNRSSIdbm):
    # Calculates the distance through path-loss propagation model
    dist_final = find_distance(TNRSSIdbm) 
    dist_final = np.transpose(dist_final) 
    pos_est_mlt = [] # Vetor que contem a coordenada estimada
    APsLoc = APsLoc.to_numpy() # Vetor contendo a posicao de AP
    # Calulates the error between the estimated distance and the path-loss propagation model
    cost_fun = lambda pos: np.sum((dist_fun(APsLoc, pos)- dist_final) ** 2) 
    # Initial condition
    cond_init = np.array([0.0, 0.0])
    # Estimates the position minimizing the fuction cost_fun
    location = scipy.optimize.minimize(
    cost_fun,            # The error function
    cond_init,           # The initial guess
    method='l-bfgs-b',   # The optimisation algorithm
    options={
    'ftol':1e-5,          # Tolerance
    'maxiter':1e+7      # Maximum iterations
    })
    posicao = location.x
    pos_est_mlt.append(list(posicao))

    return np.asarray(pos_est_mlt)


# In[5]:


KNN = 4 # Number of k nearest neighbours
nAPS = 5 # Number of APs

# Database 1 - contains the average RSSI values of the environment
Database1 = pd.read_csv("DatabaseAVG.csv",delimiter=',',index_col=None)
Database1 = Database1.to_numpy()

# RSSI values measured along the projected path
rssi = pd.read_csv("TrajetoriaGeral.csv",delimiter=',',index_col=None)
TNRSSIdBm = rssi.to_numpy()
t = range(0,len(rssi))

# APs location
APsLoc = pd.DataFrame([[0.20, 1.32], [9.625, 1.64], [5.445, 6.07], [0.48, 12.77], [9.775, 10.89]], index=('APS1', 'APS2', 'APS3', 'APS4', 'APS5'), columns=['x', 'y'])

# Scenario dimensions
largura = 9.775
comprimento = 13.45


# In[6]:


pos_FP_buffer = [] # Estimating FP position vector

pos_MLT_buffer = [] # Estimating MLT position vector

# Estimating the locations using FP and MLT techniques
for i in range(len(TNRSSIdBm)):
    xestFP,yestFP = FPest(TNRSSIdBm[i,:], Database1, nAPS, KNN)
    pos_FP_buffer.append([xestFP, yestFP])

    pos_MLT = multilateracao(APsLoc,TNRSSIdBm[i,:])
    pos_MLT_buffer.append(pos_MLT)

# Transform the lists into numpy arrays and limit the estimated positions within the area of interest 
pos_FP_buffer = np.array(pos_FP_buffer)
pos_FP_buffer = limitador(pos_FP_buffer, largura, comprimento, len(pos_FP_buffer))

pos_MLT_buffer = np.asarray(pos_MLT_buffer).reshape((i+1,2))
pos_MLT_buffer = limitador(pos_MLT_buffer, largura, comprimento, len(pos_MLT_buffer))


# In[7]:


# Applies the KF over the FP and MLT estimates
pos_FP_KF,P_FP = KF_FP.KF(pos_FP_buffer,0.00877,0.1)
pos_MLT_KF,P_MLT = KF_MLT.KF(pos_MLT_buffer,0.667,0.1)

# TTF algorithm that combines the FP and MLT estimates
# resulting in a hybrid estimate position
pos_fusion = []

for k in range(len(TNRSSIdBm)):
    pos_fusion.append(pos_FP_KF[k+1,:].reshape(4,1) + P_FP[k+1].dot(np.linalg.inv(P_MLT[k+1]+P_FP[k+1])).dot((pos_MLT_KF[k+1,:].reshape(4,1)-pos_FP_KF[k+1,:].reshape(4,1))))


pos_fusion = np.array(pos_fusion)

pos_fusion = pos_fusion.reshape(pos_fusion.shape[0],pos_fusion.shape[1])
# Limit the area of interest of the estimated positions
pos_fusion = limitador(pos_fusion, largura, comprimento, len(pos_fusion))


# In[8]:


# Trajectory positions
pos_geral = np.array([[1.45, 11.71], [2.54, 11.71], [3.63, 11.71], [3.98, 11.18], [4.98, 11.18], 
                      [5.98, 11.18], [6.58, 10.66], [7.58, 10.66], [8.52, 10.77], [8.54, 9.77], 
                      [7.54, 9.78], [6.68, 9.685], [6.58, 8.595], [6.62, 7.555], [6.525, 6.675], 
                      [7.325, 6.675], [7.325, 5.675], [7.325, 4.675], [7.325, 3.675], [7.325, 2.675], 
                      [6.325, 2.675], [6.325, 1.875], [6.325, 1.875], [7.325, 2.675], [6.325, 2.675], 
                      [7.325, 3.675], [7.325, 5.675], [7.325, 4.675], [7.325, 6.675], [6.525, 6.675], 
                      [6.58, 8.595], [6.62, 7.555], [6.68, 9.685], [7.54, 9.78], [8.54, 9.77], 
                      [7.58, 10.66], [8.52, 10.77], [6.58, 10.66], [5.98, 11.18], [1.45, 11.71], 
                      [2.54, 11.71], [3.63, 11.71], [3.98, 11.18], [4.98, 11.18], [3.98, 11.18], 
                      [3.63, 11.71], [2.54, 11.71], [1.45, 11.71], [3.63, 11.71], [3.98, 11.18]])

dist_mlt=[]
dist_mlt_kf=[]
dist_fp=[]
dist_fp_kf=[]
dist_fusion=[]
# Calculate the mean square erro for each technique
# MLT standalone, FP standalone, MLT+KF, FP+KF, H-IPS
for i in range(len(pos_MLT_buffer)):
    dist_mlt.append(list((pos_MLT_buffer[i, :] - pos_geral[i]) ** 2))

for i in range(len(pos_MLT_KF)-1):
    dist_mlt_kf.append(list((pos_MLT_KF[i+1, 0:2] - pos_geral[i]) ** 2))

for i in range(len(pos_FP_buffer)):
    dist_fp.append(list((pos_FP_buffer[i, 0:2] - pos_geral[i]) ** 2))

for i in range(len(pos_FP_KF)-1):
    dist_fp_kf.append(list((pos_FP_KF[i+1, 0:2] - pos_geral[i]) ** 2))

for i in range(len(pos_fusion)):
    dist_fusion.append(list((pos_fusion[i, 0:2] - pos_geral[i]) ** 2))

dist_mlt = np.sqrt(np.sum(np.asarray(dist_mlt), axis=1))
dist_mlt_kf = np.sqrt(np.sum(np.asarray(dist_mlt_kf), axis=1))
dist_fp = np.sqrt(np.sum(np.asarray(dist_fp), axis=1))
dist_fp_kf = np.sqrt(np.sum(np.asarray(dist_fp_kf), axis=1))
dist_fusion = np.sqrt(np.sum(np.asarray(dist_fusion), axis=1))

# Print the average positioning error in meters for each technique
print(np.mean(dist_mlt))
print(np.mean(dist_mlt_kf))
print(np.mean(dist_fp))
print(np.mean(dist_fp_kf))
print(np.mean(dist_fusion))


# In[9]:


# Plots the test trajectory
get_ipython().run_line_magic('matplotlib', 'qt')
plt.figure(1)
plt.plot(APsLoc.x,APsLoc.y,'r*',label='AP',markersize=5)
plt.plot(pos_geral[:,0],pos_geral[:,1],'ko-',label='pos_true', markersize=5)

for a,b,c in zip(APsLoc.x, APsLoc.y, APsLoc.index):
    
    label = c
    
    plt.annotate(label, # this is the text
                 (a,b), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,5), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
    
plt.xticks(np.arange(0,14,2))
plt.yticks(np.arange(0,15,2))
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend()
plt.grid()
#plt.savefig('H-IPS_Trajectory.pdf',format='pdf',bbox_inches='tight')
plt.show()


# In[10]:


# Calculates the empirical CDF to FP, MLT and H-IPS methods
norm_HIPS = np.sort(dist_fusion)
cdf_HIPS = 1. * np.arange(len(dist_fusion)) / (len(dist_fusion) - 1)

norm_FP = np.sort(dist_fp)
cdf_FP = 1. * np.arange(len(dist_fp)) / (len(dist_fp) - 1)

norm_MLT = np.sort(dist_mlt)
cdf_MLT = 1. * np.arange(len(dist_mlt)) / (len(dist_mlt) - 1)

# Plots the CDF curve
plt.figure(2)
plt.plot(norm_HIPS,cdf_HIPS,'k*-',label='H-IPS',markersize=5)
plt.plot(norm_FP,cdf_FP,'b:o',label='FP',markersize=3)
plt.plot(norm_MLT,cdf_MLT,'rs--',label='MLT',markersize=3)
plt.title('Empirical CDF')
plt.xlabel('Error [m]')
plt.ylabel('CDF')
plt.legend()
plt.grid()
#plt.savefig('H-IPS_CDF.pdf',format='pdf',bbox_inches='tight')
plt.show()


# In[11]:


# Calculates the distance between the estimated position and the true position
all_POS = pos_geral
all_POS = np.append(all_POS,pos_FP_buffer,axis=1)
all_POS = np.append(all_POS,pos_MLT_buffer,axis=1)
all_POS = np.append(all_POS,pos_fusion[:,0:2],axis=1)

# FINGERPRINTING
vetorFP = np.array([all_POS[0,0], all_POS[0,1]]).reshape(1,2)
for i in range(0,len(all_POS)):
    vetorFP = np.append(vetorFP, np.array([all_POS[i,0],all_POS[i,1]]).reshape(1,2),axis=0)
    vetorFP = np.append(vetorFP, np.array([all_POS[i,2],all_POS[i,3]]).reshape(1,2),axis=0)
    vetorFP = np.append(vetorFP, np.array([all_POS[i,0],all_POS[i,1]]).reshape(1,2),axis=0)

# MULTILATERATION
vetorMLT = np.array([all_POS[0,0], all_POS[0,1]]).reshape(1,2)
for i in range(0,len(all_POS)):
    vetorMLT = np.append(vetorMLT, np.array([all_POS[i,0],all_POS[i,1]]).reshape(1,2),axis=0)
    vetorMLT = np.append(vetorMLT, np.array([all_POS[i,4],all_POS[i,5]]).reshape(1,2),axis=0)
    vetorMLT = np.append(vetorMLT, np.array([all_POS[i,0],all_POS[i,1]]).reshape(1,2),axis=0)

# H-IPS
vetorFUSION = np.array([all_POS[0,0], all_POS[0,1]]).reshape(1,2)
for i in range(0,len(all_POS)):
    vetorFUSION = np.append(vetorFUSION, np.array([all_POS[i,0],all_POS[i,1]]).reshape(1,2),axis=0)
    vetorFUSION = np.append(vetorFUSION, np.array([all_POS[i,6],all_POS[i,7]]).reshape(1,2),axis=0)
    vetorFUSION = np.append(vetorFUSION, np.array([all_POS[i,0],all_POS[i,1]]).reshape(1,2),axis=0)

# Plots the graphic result for each method (FP, MLT and H-IPS) along the trajectory 
plt.figure(3)
plt.plot(APsLoc.x,APsLoc.y,'r*',label='AP',markersize=5)
plt.plot(pos_geral[:,0],pos_geral[:,1],'ko-',label='pos_true', markersize=5)
plt.plot(pos_fusion[:,0],pos_fusion[:,1],'b*',label='H-IPS',markersize=5)
plt.plot(vetorFUSION[:,0],vetorFUSION[:,1],'b',linewidth=0.2)
plt.plot(pos_FP_buffer[1:,0],pos_FP_buffer[1:,1],'gs',label='FP',markersize=3)
plt.plot(vetorFP[:,0],vetorFP[:,1],'g',linewidth=0.2)
plt.plot(pos_MLT_buffer[1:,0],pos_MLT_buffer[1:,1],'mo',label='MLT',markersize=2)
plt.plot(vetorMLT[:,0],vetorMLT[:,1],'m',linewidth=0.2)

for a,b,c in zip(APsLoc.x, APsLoc.y, APsLoc.index):
    
    label = c
    
    plt.annotate(label, # this is the text
                 (a,b), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,5), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
    
plt.xticks(np.arange(0,15,2))
plt.yticks(np.arange(0,15,2))
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend()
plt.grid()
#plt.savefig('H-IPS_Results_v2.pdf',format='pdf',bbox_inches='tight')
plt.show()

