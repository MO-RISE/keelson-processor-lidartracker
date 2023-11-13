'''
motionEstim.py
Include all parts of the motion estimation for single object. 
Start by a linear Kalman filter, with future work aiming at IMM.

MiS - Martin Sanfridson, April 2023
'''

import numpy as np
import copy
import sizeEstim
import boxHandling
import boxHandlingUtils
import lkfTracker



class motionEstim:
    #This class is for only one tracker, but encapsulates the different calculations/estimations 
    count = 0

    def __init__(self,x_init,Lnom,Wnom,samplingPeriod):
        self.dt = samplingPeriod
        self.id = motionEstim.count
        motionEstim.count += 1
        self.bh = boxHandling.boxHandling(Lnom=Lnom,Wnom=Wnom)
        self.lkf = lkfTracker.linearKalmanFilter(x_init,self.dt) #motion state estimation filter
        #self.se = sizeEstim.sizeEstim(np.array([Lnom*0.9,Wnom*1.1]),samplingPeriod) #box size estimator


    def update(self,bbox):
        #update all parts
        measValid = self.bh.refineBBox(bbox) #what if bbox is not found
        if measValid:
            midBBox = self.bh.getNewMidpoint()
            oriBBox = self.bh.getOrientation()
            z = np.concatenate((midBBox,np.array([oriBBox])),axis=0)
            self.lkf.update(z.reshape((3,1)))
            #get V1 and V2 as measurement noise for position to inhibit updating without seeing the sides (degenerated case)
            #fsb.setMeasNoise(np.diag([1*(0.9*(bboxV1[k,0])),1*(0.9*(bboxV2[k,0])),1*(0.9*(bboxV1[k,0]))])) #based on V1 and V2, vice versa?

    def predict(self):
        self.lkf.predict()
        #xvecMotion = xvecMotion.append(self.lkf.getState())
        #PvecMotion = PvecMotion.append(self.lkf.getCovariance())


    def predictTraj(self,tick,predInterval,predSteps):
        #invoke once every interval to predict nPred steps ahead 
        xvec, tvec = list(), list()
        if tick % predInterval == 0: #or self.lkf.hits % predInterval == 0:
            kfPred = copy.deepcopy(self.lkf) #does not affect original instance
            for t in range(predSteps):
                xvec.append(kfPred.predict())
                tvec.append(self.dt*int(tick + t + 1))
        return xvec, tvec #empty if not a new sample
    


class multiMotionEstim:
    #This class handles a list of motionEstim, currently does not handle multiple sampling periods
    def __init__(self,samplingPeriod,predInterval,predSteps):
        #always match estimators with trackers
        self.tick = 0 #count time
        self.estTrk, self.estimators = list(), list()
        self.dt = np.array(samplingPeriod)
        self.predInterval = predInterval
        self.predSteps = predSteps

    def addEstimator(self,obb3D,th):
        #TODO: figure out more initial conditions
        #state vector is [x,dx,y,dy,c,dc]
        x = obb3D.center[0]
        y = obb3D.center[1]
        width = obb3D.extent[0:2].min()
        length = obb3D.extent[0:2].max()
        x_init = np.array([x,0.0,y,0.0,th,0.0]) 
        me = motionEstim(x_init,length,width,self.dt) 
        #print(f"Added estimator with ID {}")
        self.estimators.append(me)
        estLoc = len(self.estimators)-1 #nb, only correct if nothing is removed
        return me.id, estLoc
    
    def tickTack(self):
        self.tick += 1

    def measureUpdate(self,trkId,obb3D,th): 
        #if tracker not in list, then addEstimator()
        #TODO: could add objectPC to input list later
        matchEst = [m for m in self.estTrk if m['trkId'] == trkId] #first hit should be the only
        if len(matchEst) == 0:
            estId, estLoc = self.addEstimator(obb3D,th)
            self.estTrk.append({'trkId': trkId, 'estId': estId, 'estLoc': estLoc})
        else:
            self.estimators[matchEst[0]['estLoc']].update(obb3D)
            #self.estimators[match[0]['estLoc']].predict()

    def predictAll(self):
        for est in self.estTrk:
            self.estimators[est['estLoc']].predict()

    def removeEstimator(self):
        #garbage collection if not seen for a long while
        #pop both estimators and trackers
        pass

    def getStateWithCovariance(self,trkId):
        state, cov = None, None
        matchEst = [m for m in self.estTrk if m['trkId'] == trkId] #first hit should be the only
        if len(matchEst) > 0:
            state = self.estimators[matchEst[0]['estId']].lkf.getState()
            cov = self.estimators[matchEst[0]['estId']].lkf.getCovariance()
        return state, cov


    def getPredTrajWithTime(self,trkId):
        #returns list of state vector for tracker 'id'
        xvecMotionPred,tvecMotionPred = list(), list()
        matchEst = [m for m in self.estTrk if m['trkId'] == trkId] #first hit should be the only
        if len(matchEst) > 0:
            xvecMotionPred,tvecMotionPred = self.estimators[matchEst[0]['estId']].predictTraj(self.tick,self.predInterval,self.predSteps)
        return xvecMotionPred,tvecMotionPred
    
    
    def getCurrentTime(self):
        #multiEstim does not slip in time if missing objects
        return self.dt*self.tick


