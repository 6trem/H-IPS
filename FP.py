# Author: Leonardo Sestrem de Oliveira
# Date: 07/10/2021
# FP positioning estimation code

import numpy as np
from scipy.spatial import ConvexHull

def FPest (TMRSSI,database,nAPS,KNN):
    # MSE values matrix
    erroMSE = np.zeros((len(database),nAPS))
    
    # Calculates the differece between the measured RSSI and the database value
    for j in range(0,nAPS):
        erroMSE[:,j] = TMRSSI[j] - database[:,3+j]
    
    # Computes the MSE and inserts in the database matrix
    erroMSE2 = erroMSE**2
    erroMSE2 = np.append(erroMSE2,(np.sum(erroMSE2,axis=1)).reshape(len(database),1),axis=1)
    erroMSE2 = np.append(erroMSE2,(erroMSE2[:,nAPS]/nAPS).reshape(len(database),1),axis=1)
    database = np.append(database,(erroMSE2[:,nAPS+1]).reshape(len(erroMSE2),1),axis=1)
    
    # Search in the database the first KNNs lower MSEs
    xy_hat = np.zeros((KNN,KNN))
    
    for b in range(0,KNN):
        # Seach the lower MSE 
        min_mse_value = min(database[:,nAPS+3])
        # Search the corresponding position
        min_mse_index = np.argmin(database[:,nAPS+3])
        # Coordinate x of the lower MSE
        xpos = database[min_mse_index,1]
        # Coordinate y of the lower MSE
        ypos = database[min_mse_index,2]
        # Possible estimates matrix
        xy_hat[:,b] = [min_mse_value,min_mse_index,xpos,ypos]
        # Replace the found value by 200
        database[min_mse_index,nAPS+3] = 200
    
    # Function which verifies if the points are collinear through 
    # the matrix rank (SVD method)
    pointsAreCollinear = lambda xy: np.linalg.matrix_rank(xy[1:,:] - xy[0,:]) == 1
    # Possible x values
    x1 = xy_hat[2,:]
    # Possible y values
    y1 = xy_hat[3,:]
    
    # Case the sum of the possible coordinates x and y are null,
    # the estiamte position will be the point (0,0)
    if sum(x1) == 0 and sum(y1) == 0:
        xest = 0
        yest = 0
        
    else:
        # Case the points are not collinear, the estimated position,
        # will be the centroid of the convex hull formed by the 
        # possible points
        pos = np.array([x1,y1]).transpose()
        if not pointsAreCollinear(pos):            
            hull = ConvexHull(pos)
            cx = np.mean(hull.points[hull.vertices,0])
            cy = np.mean(hull.points[hull.vertices,1])
            xest = cx
            yest = cy          
            
        
        # Case points are collinear, the estimate will be 
        # the average  of the possible coordinates
        else:
            xest = np.mean(x1)
            yest = np.mean(y1)        
        
    
    return xest,yest