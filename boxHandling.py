'''
boxHandling.py
Class of functions for handling bounding box for tracking using lidar which sees the object from one side. This
is to be used prior to motion tracking by e.g. Kalman or an IMM filter.

Input: a measured bounding box defined with four corners 
Output: a modified bounding box with middle, orientation (+-pi), length and width, and idea of their confidence
Feature: taking into account the direction from which the lidar measured the object, in order to improve measurements

Built to be executed each time instant in casual order. Does not store history.

MiS - Martin Sanfridson, January 2023

'''

import numpy as np
import boxHandlingUtils


class boxHandling():
    def __init__(self,Lnom,Wnom):
        self.firstTime = True #needed to know there's a history or not
        self.bbox = None
        self.eps = 1e-4 #could tailor to sensor's angular resolution
        self.angleToCorner = np.zeros((4,))
        #self.ang_prev = self.angleToCorner
        self.hiddenCorner = np.zeros((4,),dtype=np.int32)
        #TODO: need some initial value for s2_prev and c2_prev!
        self.s2_prev = 0
        self.c2_prev = 0
        #self.s2_prev = np.NaN
        #self.c2_prev = np.NaN
        
        self.S1 = np.NaN
        self.S2 = np.NaN
        self.V1 = np.NaN
        self.V2 = np.NaN
        self.mind = np.NaN
        self.bboxPoints = np.zeros((4,2))
        self.newBBoxPoints = np.zeros((4,2))
        self.midpoint = np.zeros((2,)) #c_x,c_y
        self.newMidpoint = np.zeros((2,)) 
        self.setLengthAndWidth(Lnom,Wnom)
        #self.Wnom = np.NaN
        #self.Lnom = np.NaN


    def setBBox(self,boxPoints):
        #NOTE: changed definition of boxPoints, need to convert from oriented 3D to array for 2D corners
        eightPoints = np.asarray(boxPoints.get_box_points())
        self.bboxPoints = boxHandlingUtils.bbox3DTo2D_fourPoints(eightPoints)
        #self.midpoint = np.mean(self.bboxPoints,axis=0)
        self.midpoint = boxPoints.get_center()

    def checkBBox(self):
        return len(np.unique(self.bboxPoints,axis=0)) == 4


    def setLengthAndWidth(self,Lnom,Wnom):
        self.Wnom = Wnom
        self.Lnom = Lnom


    def anglesAndHiddenCorners(self):

        def calc4Angles(boxPoints):
            ang = np.zeros((4,))
            for k in range(4):
                ang[k] = np.arctan2(boxPoints[k,1],boxPoints[k,0])
            return ang

        #TODO: find a more robust way of sorting this out
        #calc unwrapped angles to all corners
        ang0 = calc4Angles(self.bboxPoints)
        if self.firstTime: #wrap in space first time (NB: not debugged)
            #ang = np.unwrap(ang0)
            ang = np.unwrap(np.hstack((0,ang0)))[1:]
            #ang = np.unwrap(np.vstack(([0,0,0,0],ang0)),axis=0,period=np.pi/2)[1,:]
        else: #k > 0: #wrap in time
            #ang = np.unwrap(np.vstack((self.angleToCorner,ang)),axis=0)[1,:]
            #below uses previous value: self.angelToCorner, is that really necessary?
            #ang = np.unwrap(np.vstack((self.angleToCorner,ang0)),period=np.pi/2,axis=0)[1,:] #choice of period(?)
            ang = np.unwrap(np.hstack((0,ang0)),period=np.pi/2,axis=0)[1:] #choice of period(?)
        #reorder data
        ind = np.argsort(ang) #corner order in increasing angle, ind is a vector
        self.bboxPoints[:,0] = np.take(self.bboxPoints[:,0],ind,axis=0)
        self.bboxPoints[:,1] = np.take(self.bboxPoints[:,1],ind,axis=0)
        #find out what corners are hidden
        nm = np.linalg.norm(self.bboxPoints,axis=1)
        self.hiddenCorner[1] = nm[1] > nm[0] #if further away then hidden
        self.hiddenCorner[2] = nm[2] > nm[3] #d:o

        if np.all(self.hiddenCorner == 0):
            print("hiddenCorner is not valid")
            import matplotlib.pyplot as plt
            plt.plot(self.bboxPoints[(0,1,2,3,0),0],self.bboxPoints[(0,1,2,3,0),1])
            #for k in range(4):
            #    plt.gca().text(self.bboxPoints[k,0],self.bboxPoints[k,1],
            #    f"k {k}\nang {self.angleToCorner[k]} \nang0 {ang0[k]} \nprevAng {np.take(ang,ind)[k]} \nnm {nm[k]} \nhidden {self.hiddenCorner[k]}")
            plt.show()

        self.angleToCorner = np.take(ang,ind)

        #TODO: output should be checked since only a few combinations are valid


    def sidesAndOrientation(self):
        #figure out hidden points, c/heading

        def costhe(p1,p2,a12):
            d1 = np.linalg.norm(p1) #distance from the sensor in the origin
            d2 = np.linalg.norm(p2)
            return np.sqrt(d1*d1 + d2*d2 -2*d1*d2*np.cos(a12)) #TODO: runtime error of "invalid value"

        if np.array_equal(self.hiddenCorner,np.array([0,1,1,0])):
            p = np.take(self.bboxPoints,[0,3],axis=0)
            a = np.take(self.angleToCorner,[0,3],axis=0)
            v1 = 1/(a[1]-a[0]+self.eps) #times 2pi
            v2 = 1/self.eps
            s1 = costhe(p[0,:],p[1,:],a[1]-a[0])
            s2 = self.s2_prev #np.NaN "degenerated case"
            c1 = np.arctan2(p[0,1]-p[1,1],p[0,0]-p[1,0])
            #TODO: how to set initial value
            c2 = self.c2_prev #or add fix pi/2 with unwrap
            #c2 = c1 + np.pi/2
        elif np.array_equal(self.hiddenCorner,np.array([0,1,0,0])):
            p = np.take(self.bboxPoints,[0,2,3],axis=0)
            a = np.take(self.angleToCorner,[0,2,3],axis=0)
            v1 = 1/(a[1]-a[0]+self.eps)
            v2 = 1/(a[2]-a[1]+self.eps)
            s1 = costhe(p[0,:],p[1,:],a[1]-a[0])
            s2 = costhe(p[1,:],p[2,:],a[2]-a[1])
            c1 = np.arctan2(p[0,1]-p[1,1],p[0,0]-p[1,0])
            c2 = np.arctan2(p[1,1]-p[2,1],p[1,0]-p[2,0])
        elif np.array_equal(self.hiddenCorner,np.array([0,0,1,0])):
            p = np.take(self.bboxPoints,[0,1,3],axis=0)
            a = np.take(self.angleToCorner,[0,1,3],axis=0)
            v1 = 1/(a[1]-a[0]+self.eps)
            v2 = 1/(a[2]-a[1]+self.eps)
            s1 = costhe(p[0,:],p[1,:],a[1]-a[0])
            s2 = costhe(p[1,:],p[2,:],a[2]-a[1])
            c1 = np.arctan2(p[0,1]-p[1,1],p[0,0]-p[1,0])
            c2 = np.arctan2(p[1,1]-p[2,1],p[1,0]-p[2,0])
        else:
            print("firstBBox: unexpected case")
        #bumpless assignment of s1 and s2 (v1 and v2)
        self.s2_prev = s2 #resoving degenerated case by using previous estimation
        self.c2_prev = c2
        if self.firstTime:
            self.mind = 0 #or use external default to set initial choice
        else:
            #todo: should use something more persistent when finding direction
            self.mind = np.argmin([(self.S1-s1)**2 + (self.S2-s2)**2,(self.S1-s2)**2 + (self.S2-s1)**2])
            #self.mind = np.argmin([(self.Lnom-s1)**2 + (self.Wnom-s2)**2,(self.Lnom-s2)**2 + (self.Wnom-s1)**2])
        #print(k,mind,c1,c2,s1>s2)
        if self.mind == 0: #keep
            self.S1 = s1 #what if NaN
            self.S2 = s2
            self.V1 = v1
            self.V2 = v2
        else: #swap
            self.S1 = s2
            self.S2 = s1
            self.V1 = v2
            self.V2 = v1
        #heading-counterheading
        if self.firstTime:
            if s1 > s2:
                self.C = c1
            else:
                self.C = c2
        else:
            if s1 > s2:
                self.C = np.unwrap(np.stack((self.C,c1)),period=np.pi/2)[1] #choice of period to be studied
            else:
                self.C = np.unwrap(np.stack((self.C,c2)),period=np.pi/2)[1]
        
        

    def calcCentres(self):
        #calc new bbox centre, assume same heading of S and P
        #TODO: need to check that L and W are not switched!!
        bboxPointsP = boxHandlingUtils.boxToCorners2D(self.Lnom,self.Wnom,[0,0],self.C) 
        bboxPointsS = boxHandlingUtils.boxToCorners2D(self.S1,self.S2,self.midpoint,self.C) #is S1 == L guaranteed?

        #find closest point
        ind = np.argmin(np.linalg.norm(bboxPointsS,axis=0))
        diff = bboxPointsS[ind.astype(np.int32),:] - bboxPointsP[ind.astype(np.int32),:]
        #translate P
        bboxPointsP = boxHandlingUtils.shift2D(bboxPointsP,diff)
        self.newBBoxPoints = bboxPointsP
        self.newMidpoint = np.mean(bboxPointsP,axis=0) #middle point



    def getMidpoint(self):
        return self.midpoint

    def getNewMidpoint(self):
        return self.newMidpoint

    def getNewBoxPoints(self):
        return self.newBBoxPoints 

    def getVisibleBoxPoints(self):
        if np.array_equal(self.hiddenCorner,np.array([0,1,1,0])):
            p = np.take(self.bboxPoints,[0,3],axis=0)
        elif np.array_equal(self.hiddenCorner,np.array([0,1,0,0])):
            p = np.take(self.bboxPoints,[0,2,3],axis=0)
        elif np.array_equal(self.hiddenCorner,np.array([0,0,1,0])):
            p = np.take(self.bboxPoints,[0,1,3],axis=0)
        else:
            p = np.take(self.bboxPoints,[0,1,2,3],axis=0)
        return p #nb: vector of varying length

    def getLengthAndWidth(self):
        return self.S1, self.S2

    def getOrientation(self):
        return self.C

    def getV1AndV2(self):
        return self.V1, self.V2

    def refineBBox(self,bbox):
        #main function, calls for a batch run per bbox
        self.setBBox(bbox)
        if self.checkBBox():
            self.anglesAndHiddenCorners()
            self.sidesAndOrientation()
            self.calcCentres()
            self.firstTime = False
            return True
        else:
            return False






