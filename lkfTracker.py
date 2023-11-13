'''
lkfTracking.py
Linear Kalman filter for tracking of position and orientation motion.
MiS - Martin Sanfridson, April 2023
'''

import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag

class linearKalmanFilter():
    #This class represents the internal state of individual tracked objects observed as bbox.
    #update X_k+1 = F*X_k with  measurement Z = H*X_k
    #The states are x, y, c (2D plus orientation) and their derivatives:
    #[x,dx,y,dy,c,dc] --> need to add estimation of curvature that relates to x and y (and c?)
    #suggestion: [x,dx,y,dy,c,dc,w], where x_k+1 = x_k + dx + trig func --> non-linear kalman filter
    def __init__(self,x_init,samplingPeriod):
        #define constant velocity model
        self.dt = samplingPeriod
        self.kf = KalmanFilter(dim_x=6, dim_z=3) 
        self.kf.F = np.array([[1,self.dt,0,0,0,0],[0,1,0,0,0,0],[0,0,1,self.dt,0,0],[0,0,0,1,0,0],[0,0,0,0,1,self.dt],[0,0,0,0,0,1]])
        self.time_since_update = 0 #what init value to use?
        self.kf.R = np.diag([0.1,0.1,0.1]) #meas noise

        # self.kf.P[] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P = np.diag([1,1,1,1,1,1]) #covariance
        q_xy = Q_discrete_white_noise(dim=2,dt=self.dt,var=1.0)
        q_r = Q_discrete_white_noise(dim=2,dt=self.dt,var=10.0)
        self.kf.Q = block_diag(q_xy,q_xy,q_r) #process noise
        self.kf.H = np.array([[1,0,0,0,0,0],[0,0,1,0,0,0],[0,0,0,0,1,0]])

        #self.kf.x[:4] = convert_input_to_z(bbox) 
        self.kf.x = x_init

    def setMeasNoise(self,R):
        self.kf.R = R

    def update(self,meas):
        #Updates the state vector with observed bbox.
        self.time_since_update = 0
        #self.history = []
        #self.hits += 1
        #self.hit_streak += 1
        #self.kf.update(convert_input_to_z(bbox)) 
        self.kf.update(meas)

    def predict(self):
        self.kf.predict()
        #self.age += 1
        #if self.time_since_update > 0:
        #    self.hit_streak = 0
        self.time_since_update += 1
        #self.history.append(convert_x_to_bbox(self.kf.x)) 
        #return self.history[-1]
        return self.kf.x

    def getState(self):
        return self.kf.x

    def getCovariance(self):
        return self.kf.P

