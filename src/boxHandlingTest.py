'''
boxHandlingTest.py

Script file for testing boxHandling and Kalman filters.

pts has the shape (nBoxes,nCorners=4,nDimensions=2)

MiS - Martin Sanfridson, January 2023

'''

import numpy as np
import matplotlib.pyplot as plt
import copy
import motionEstim
import sizeEstim
import boxHandling
import boxHandlingUtils
from boxHandlingPlot import animateTopView1, animateTopView2, animateTopView3



def plotBoxes(inputBBoxMiddle,visiblePoints,labels=None):
    for boxPoints,label in zip(visiblePoints,labels):
        #plt.plot(boxPoints[:,0],boxPoints[:,1],'bo--')
        plt.plot(boxPoints[:,0],boxPoints[:,1])
        #plot rays from sensor
        rays = np.insert(boxPoints,list(range(1,len(boxPoints))),0,axis=0) 
        plt.plot(rays[:,0],rays[:,1],'k:',linewidth=0.5)
        if labels is not None:
            plt.text(boxPoints[0,0],boxPoints[0,1],label)
    plt.plot(inputBBoxMiddle[:,0],inputBBoxMiddle[:,1],'rx')
    plt.axis('equal')
    plt.xlabel('[m]')
    plt.ylabel('[m]')
    plt.title('Rays to sides of bounding box')


def bboxToBottom(eightPoints):
    #find bottom part of 3D bbox, i.e. a 2D bbox
    #NB: remove effect of wake??
    pts = np.zeros((len(eightPoints),4,2),dtype=float)
    for k, bbox in enumerate(eightPoints):
        ind = np.argsort(bbox[:,2]) 
        pts[k,:] = np.take(bbox[:,0:2],ind[4:],axis=0)
    return pts


#load data
#bbData = np.load("../Tracker/savedTrajectory.npz")
#bbData = np.load("../Tracker/traj20221111t1133_1.npz") #need to work harder on this one!

#the original
#Status 2023-04-25: se ok ut
if False:
    bbData = np.load("../Tracker/traj20221111t1133_2.npz") 
    x_init = [31.0,0.1,1.0,0,0,0] #[x,dx,y,dy,c,dc]
    Wnom = 2.0
    Lnom = 5.0

# #bbData = np.load("../Tracker/traj20221111t1133_3.npz") #need to work more on this one!
#bbData = np.load("../Tracker/traj20221111t1133_4.npz")

#MOT kungsbesök
#Status 2023-04-25: kör åt fel håll
if True:
    bbData = np.load("../Tracker/traj20221013t1356_1.npz") 
    x_init = [5,1.0,-55,0,3.14,0]
    Wnom = 8.0
    Lnom = 31.0

#bbData = np.load("../Tracker/traj20221111t1126_1.npz") #switzer
#TODO: check if it works if all zeros are removed first. Anglefailure or sequencefailure?

#idea of debugging: remove outliers - but they are too many!
#if np.all(np.equal(pts[1,...],np.zeros((8,3)))):
#    bbData2.append()


pts = bboxToBottom(bbData['p'])

dt = 0.1
nBoxes = len(pts) #nb of boxes (frames)
labels = np.linspace(1,nBoxes,nBoxes,dtype=np.int16)


#TODO: change to allow nStates = 7

#save result to vectors
inputBBoxMiddle = np.zeros((nBoxes,2))
newBBoxMiddle = np.zeros((nBoxes,2))
angleToCorners = np.zeros((nBoxes,4))
hiddenCorners = np.zeros((nBoxes,4))
bboxLength = np.zeros((nBoxes,1))
bboxWidth = np.zeros((nBoxes,1))
bboxS1 = np.zeros((nBoxes,1))
bboxS2 = np.zeros((nBoxes,1))
bboxV1 = np.zeros((nBoxes,1))
bboxV2 = np.zeros((nBoxes,1))
newBBoxOrientation = np.zeros((nBoxes,1))
visiblePoints = list()
xvecMotion = np.zeros((nBoxes,6))
zvecMotion = np.zeros((nBoxes,3))
PvecMotion = np.zeros((nBoxes,6,6))
xvecSize = np.zeros((nBoxes,2))
zvecSize = np.zeros((nBoxes,2))
PvecSize = np.zeros((nBoxes,2,2))
tvec = np.linspace(0,(nBoxes-1)*dt,nBoxes)
nPred = 25
xvecMotionPredCollection = np.zeros((nBoxes,nPred,6))
tvecPredCollection = np.zeros((nBoxes,nPred))

#--- main loop
fbb = boxHandling.boxHandling(Lnom,Wnom)
s_init = np.array([Lnom*0.9,Wnom*1.1])
fkb = motionEstim.motionEstim(x_init,dt) #motion state estimation filter
fsb = sizeEstim.sizeEstim(s_init,dt) #box size estimator
for k, bbox in enumerate(pts):
    measValid = fbb.refineBBox(bbox) #what if bbox is not found
    #print(k,fbb.hiddenCorner)
    if measValid:
        newBBoxMiddle[k,:] = fbb.getNewMidpoint()
        newBBoxOrientation[k] = fbb.getOrientation()
        #filter motion states
        z1 = np.array([newBBoxMiddle[k,0],newBBoxMiddle[k,1],newBBoxOrientation[k,0]]) 
        #get V1 and V2 as measurement noise for position to inhibit updating without seeing the sides (degenerated case)
        fsb.setMeasNoise(np.diag([1*(0.9*(bboxV1[k,0])),1*(0.9*(bboxV2[k,0])),1*(0.9*(bboxV1[k,0]))])) #based on V1 and V2, vice versa?
    fkb.predict()
    xvecMotion[k,:] = fkb.getState()
    PvecMotion[k,:] = fkb.getCovariance()

    if measValid:
        fkb.update(z1.reshape((3,1)))
        zvecMotion[k,:] = z1
        visiblePoints.append(fbb.getVisibleBoxPoints()) #jagged list
        bboxS1[k],bboxS2[k] = fbb.getLengthAndWidth()
        bboxV1[k],bboxV2[k] = fbb.getV1AndV2()

        #filter box size 
        z2 = np.array([bboxS1[k,0],bboxS2[k,0]]) #based on S1 and S2
        fsb.setMeasNoise(np.diag([1*(9*(bboxV1[k,0])),1*(9*(bboxV2[k,0]))])) #based on V1 and V2, vice versa?
        fsb.predict()
        #TODO: inhibit update if noise channel is to high
        fsb.update(z2.reshape((2,1)))
        zvecSize[k,:] = z2
        xvecSize[k,:] = fsb.getState()
        PvecSize[k,:] = fsb.getCovariance()
        bboxLength[k],bboxWidth[k] = fsb.getLengthAndWidth()

        #calc mid point
        fbb.setLengthAndWidth(bboxLength[k],bboxWidth[k])
        #fbb.setLengthAndWidth(Lnom,Wnom)
        inputBBoxMiddle[k,:] = fbb.getMidpoint()

    #test of prediction
    if k % 1 == 0:
        fkbPred = copy.deepcopy(fkb)
        #xvecMotionPred = np.zeros((nPred,6))
        tvecPredCollection[k,:] = tvec[k]+dt + np.linspace(0,(nPred-1)*dt,nPred)
        for q in range(nPred):
            fkbPred.predict()
            xvecMotionPredCollection[k,q,:] = fkbPred.getState()

    #print(k,fbb.mind,fbb.C)
    #print(k,fbb.hiddenCorner,fbb.angleToCorner,np.linalg.norm(fbb.bboxPoints,axis=1))
   
#TODO: write an Envelope function for the states (max and min for each time step), use "plt.fill_between"
#TODO: need state estimation of curvature (CT) to get a curvilinear prediction of motion


#--- plotting  ---
#TODO: structure the plotting functions better

if False:
    #plot predictions
    fig,axs = plt.subplots(2,1,sharex=True)
    fig.subplots_adjust(hspace=0)
    fig.suptitle("Prediction, position and orientation")
    #position and orientation
    #axs[0,0].plot(tvec,zvecMotion[:,(0,1)])
    #axs[0,0].plot(tvec,xvecMotion[:,(0,2)])
    #tvecTot = np.hstack((tvec,tvecPred))
    #xvecMotionTot = np.vstack((xvecMotion,xvecMotionPred))
    axs[0].plot(tvec,xvecMotion[:,0])
    for t,v in zip(tvecPredCollection,xvecMotionPredCollection):
        axs[0].plot(t,v[:,0],':')
    axs_right = axs[0].twinx()
    axs_right.plot(tvec,xvecMotion[:,2])
    for t,v in zip(tvecPredCollection,xvecMotionPredCollection):
        axs_right.plot(t,v[:,2],':')
    axs[0].set_ylabel("[m]")

    axs[1].plot(tvec,xvecMotion[:,4])
    for t,v in zip(tvecPredCollection,xvecMotionPredCollection):
        axs[1].plot(t,v[:,4],':')
    #axs_right = axs[1].twinx()
    #axs_right.plot(tvec,xvecMotion[:,5])
    #for t,v in zip(tvecPredCollection,xvecMotionPredCollection):
    #    axs_right.plot(t,v[:,5],':')
    axs[1].set_ylabel("[rad]")
    axs[1].set_xlabel("[s]")
    plt.show()


def bboxFromStateVector(vecMotion):
    #convert from result vector to corner points
    #TODO: use varying L and W, add vecSize as input --> L AND W switched!?!
    bbox = np.zeros((len(vecMotion),5,2)) #4 of x and y
    for k in range(len(vecMotion)):
        bbox[k,0:4,:] = boxHandlingUtils.boxToCorners2D(Wnom,Lnom,[vecMotion[k,0],vecMotion[k,2]],vecMotion[k,4])
        bbox[k,4,:] = bbox[k,0,:] 
    return bbox 

if True:
    #plot xy trace with predictions, NB: entries in collection-vectors can be zeros by purpose!
    fig,axs = plt.subplots(1,1,sharex=True)
    fig.subplots_adjust(hspace=0)
    fig.suptitle("Prediction")
    axs.plot(xvecMotion[:,0],xvecMotion[:,2],'k',linewidth=2.0)
    for vecMotion in xvecMotionPredCollection:
        axs.plot(vecMotion[:,0],vecMotion[:,2],'k:',linewidth=1.0)
    #plot bboxes
    #prop_cycle = plt.rcParams['axes.prop_cycle']
    #colors = prop_cycle.by_key()['color']
    plt.gca().set_prop_cycle(None)
    for vecMotion in bboxFromStateVector(xvecMotion[0::30,...]):
        plt.plot(vecMotion[:,0],vecMotion[:,1])
    plt.gca().set_prop_cycle(None)
    for vecMotionPred in xvecMotionPredCollection[0::30,...]:
        for vecMotion in bboxFromStateVector(vecMotionPred[24::1,...]):
            plt.plot(vecMotion[:,0],vecMotion[:,1],':',linewidth=0.5)
    plt.axis('equal')
    plt.show()
    xlim = axs.get_xlim()
    ylim = axs.get_ylim()

    #latvel chart input
    velMag = np.linalg.norm(xvecMotion[:,(1,3)],axis=1)
    ori = np.unwrap(np.arctan2(xvecMotion[:,3],xvecMotion[:,1]))
    velAng = ori-xvecMotion[:,4]
    velLat = np.sin(velAng)*velMag 
    velLong = np.cos(velAng)*velMag
    L = np.mean(bboxLength)
    velLat_bow = velLat + xvecMotion[:,5]*L/2
    velLat_stern = velLat - xvecMotion[:,5]*L/2

    animateTopView3(xvecMotion,xvecMotionPredCollection,xlim,ylim,velLong,velLat_bow,velLat_stern)



if False:
    #compare input and filtered estimation
    fig,axs = plt.subplots(3,2,sharex=True)
    fig.subplots_adjust(hspace=0)
    fig.suptitle("Position and orientation")
    #position and orientation
    #axs[0,0].plot(tvec,zvecMotion[:,(0,1)])
    #axs[0,0].plot(tvec,xvecMotion[:,(0,2)])
    axs[0,0].plot(tvec,zvecMotion[:,0])
    axs[0,0].plot(tvec,xvecMotion[:,0])
    axs_right = axs[0,0].twinx()
    axs_right.plot(tvec,zvecMotion[:,1])
    axs_right.plot(tvec,xvecMotion[:,2])
    axs[0,0].set_title("xy position (abs, diff, der)")
    axs[0,0].set_ylabel("[m],[rad]")
    axs[0,1].plot(tvec,zvecMotion[:,(2)])
    axs[0,1].plot(tvec,xvecMotion[:,(4)])
    axs[0,1].set_title("z orientation (abs, diff, der)")
    #comparison of pos and ori
    axs[1,0].plot(tvec,zvecMotion[:,(0,1)] - xvecMotion[:,(0,2)])
    axs[1,1].plot(tvec,zvecMotion[:,(2)] - xvecMotion[:,(4)])
    axs[1,0].set_ylabel("[m],[rad]")
    #rates (other states)
    axs[2,0].plot(tvec,xvecMotion[:,(1,3)])
    axs[2,1].plot(tvec,xvecMotion[:,(5)])
    axs[2,0].set_ylabel("[m/s],[rad/s]")

    plt.show()
    

if False:
    #compare bbox size filtering
    fig,axs = plt.subplots(3,1,sharex=True)
    fig.subplots_adjust(hspace=0)
    fig.suptitle("Length and width (abs, diff, var)")
    #filtering result
    axs[0].plot(tvec,zvecSize[:,0])
    axs[0].plot(tvec,zvecSize[:,1])
    axs[0].plot(tvec,xvecSize[:,0])
    axs[0].plot(tvec,xvecSize[:,1])
    axs[0].set_ylabel('[m]')
    #d:o difference
    axs[1].plot(tvec,zvecSize[:,0] - xvecSize[:,0])
    axs[1].plot(tvec,zvecSize[:,1] - xvecSize[:,1])
    axs[1].set_ylabel('[m]')
    #rates (other states)
    axs[2].plot(tvec,np.log10(bboxV1))
    axs[2].plot(tvec,np.log10(bboxV2))
    axs[2].set_ylabel('[log]')
    plt.show()

if False: 
    #plot covariance matrix
    fig,axs = plt.subplots(6,6,sharex=True)
    for k1 in range(6):
        for k2 in range(6):
            axs[k1,k2].plot(tvec,PvecMotion[:,k1,k2])
    plt.show()

if False: 
    #plot covariance matrix
    fig,axs = plt.subplots(2,2,sharex=True)
    for k1 in range(2):
        for k2 in range(2):
            axs[k1,k2].plot(tvec,PvecSize[:,k1,k2])
    plt.show()

if False:
    #velocity of centre in local coordinates
    fig,axs = plt.subplots(3,1,sharex=True)
    fig.subplots_adjust(hspace=0)
    fig.suptitle("Velocity of centre in local coord (magnitude, angle, (vlat,vlong))")
    #axs[0].plot(tvec,xvecMotion[:,(1,3)])
    velMag = np.linalg.norm(xvecMotion[:,(1,3)],axis=1)
    axs[0].plot(tvec,velMag)
    axs[0].set_ylabel("[m/s]")
    ori = np.unwrap(np.arctan2(xvecMotion[:,3],xvecMotion[:,1]))
    #axs[1].plot(tvec,ori)
    #axs[1].plot(tvec,xvecMotion[:,4])
    velAng = ori-xvecMotion[:,4]
    axs[1].plot(tvec,velAng) #speed vector seen from the vessel
    axs[1].set_ylabel("[rad]")
    axs[2].plot(tvec,np.sin(velAng)*velMag)
    axs[2].plot(tvec,np.cos(velAng)*velMag)
    axs[2].set_ylabel("[m/s]")
    axs[2].set_xlabel("[s]")
    plt.show()

if False:
    #lateral velocity of bow and stern
    fig,axs = plt.subplots(2,1,sharex=True)
    fig.subplots_adjust(hspace=0)
    fig.suptitle("Centre, bow and stern lateral velocity")
    velMag = np.linalg.norm(xvecMotion[:,(1,3)],axis=1)
    ori = np.unwrap(np.arctan2(xvecMotion[:,3],xvecMotion[:,1]))
    velAng = ori-xvecMotion[:,4]
    velLat = np.sin(velAng)*velMag 
    velLong = np.cos(velAng)*velMag
    L = np.mean(bboxLength)
    velLat_bow = velLat + xvecMotion[:,5]*L/2
    velLat_stern = velLat - xvecMotion[:,5]*L/2
    axs[0].plot(tvec,velLat)
    axs[0].set_ylabel("[m/s]")
    axs[1].plot(tvec,velLat_bow)
    axs[1].plot(tvec,velLat_stern)
    axs[1].set_ylabel("[m/s]")
    axs[1].set_xlabel("[s]")
    plt.show()

    animateTopView1(velLong,velLat_bow,velLat_stern)

if False:
    plotBoxes(inputBBoxMiddle[1::20,...],visiblePoints[1::20],labels=labels[1::20,...])
    plt.show()



