'''
sizeEstim.py
Bounding box size estimation.
Alternative: could use some totally different method to improve estimate of sides, e.g. ICP
Built to be executed each time instant in casual order. Does not store history.

MiS - Martin Sanfridson, January 2023
'''


import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

class sizeEstim():
    #This class is for estimation of the sides of the box

    def __init__(self,x_init,samplingPeriod):
        #define constant velocity model
        self.dt = samplingPeriod
        self.kf = KalmanFilter(dim_x=2, dim_z=2) 
        self.kf.F = np.array([[1,0],[0,1]])
        
        self.kf.R = np.diag([0.1,0.1]) #meas noise
        
        self.kf.P = np.diag([10,10]) #covariance
        #self.kf.Q = Q_discrete_white_noise(dim=2,dt=self.dt,var=1.0)
        self.kf.Q = np.eye(2)
                
        self.kf.H = np.array([[1,0],[0,1]])

        self.kf.x = x_init

    def setMeasNoise(self,R):
        self.kf.R = R

    def setModelNoise(self,Q):
        self.kf.Q = Q

    def update(self,meas):
        self.kf.update(meas)

    def predict(self):
        self.kf.predict()

    def getState(self):
        return self.kf.x

    def getCovariance(self):
        return self.kf.P

    def getLengthAndWidth(self):
        return self.kf.x[0],self.kf.x[1]
