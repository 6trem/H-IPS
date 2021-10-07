# Author: Leonardo Sestrem de Oliveira
# Date: 07/10/2021
# Limit the area of interest
import numpy as np

def limitador(pos_est,largura, comprimento, Namostra):

    x = pos_est[:,0]
    y = pos_est[:,1]

    for i in range(Namostra):
        # This function ensures that the estimates
        # are within the environmental dimensions
        if x[i] < 0:
            x[i] = 0
    
        elif x[i] > largura:
            x[i] = largura
    
        if y[i] < 0:
            y[i] = 0
    
        elif y[i] > comprimento:
            y[i] = comprimento

    res = np.array([x,y])

    return res.transpose() 

    

