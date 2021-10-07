# Author: Leonardo Sestrem de Oliveira
# Date: 07/10/2021
# Kalman Filter to the FP estimates

import numpy as np

def KF(pos_est, q, dt):
    # Model matrix, linear motion
    Phi = np.array([[1, 0, dt,0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
    # Measurement matrix
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    # Model covariance error matrix
    Q = q * np.array([[1,0,1,0],[0,1,0,1],[1,0,1,0],[0,1,0,1]])
    # Measurement covariance error matrix
    R = np.array([[0.0388, 0],[0, 0.0565]])
    # Measurement values
    z = pos_est
    
    # A posteriori state
    hat_x_ma = np.array([[0],[0],[0],[0]])
    # Estimate covariance error matrix
    P_ma = np.eye(4)
    # A priori state
    hat_x_me = hat_x_ma
    # Vector of covariance matrices that will be used on TTF algorithm
    P_buffer = [P_ma]

    for k in range(len(z)):
        
        if k == 0: # First iteration
            # Prediction
            hat_x_me = np.append(hat_x_me, Phi.dot(hat_x_ma[:, k].reshape(4, 1))+ np.random.normal(0,Q,1), axis=1)
            P_me = Phi.dot(P_ma.dot(np.transpose(Phi))) + Q
            # Correction
            K = P_me.dot(np.transpose(H).dot(np.linalg.inv(H.dot(P_me.dot(np.transpose(H))) + R)))
            hat_x_ma = np.append(hat_x_ma, (hat_x_me[:, k + 1].reshape(4, 1) + K.dot(z[k, :].reshape(2, 1) - H.dot(hat_x_me[:, k + 1].reshape(4, 1)))), axis=1)
            P_ma = P_me - K.dot(H.dot(P_me))
            P_buffer.append(P_ma)
        
        elif k>0:
            # Prediction
            hat_x_me = np.append(hat_x_me, Phi.dot(hat_x_ma[:, k].reshape(4, 1))+ np.random.normal(0,Q,1), axis=1)
            P_me = Phi.dot(P_ma.dot(np.transpose(Phi))) + Q
            # Correction
            K = P_me.dot(np.transpose(H).dot(np.linalg.inv(H.dot(P_me.dot(np.transpose(H))) + R)))
            hat_x_ma = np.append(hat_x_ma, (hat_x_me[:, k + 1].reshape(4, 1) + K.dot(z[k, :].reshape(2, 1) - H.dot(hat_x_me[:, k + 1].reshape(4, 1)))), axis=1)
            P_ma = P_me - K.dot(H.dot(P_me))
            P_buffer.append(P_ma)
        
    
    return hat_x_ma.transpose(), P_buffer

