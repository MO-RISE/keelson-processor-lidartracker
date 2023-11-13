import ousterRead
import prefilterLive
import rectangleTracker
import motionEstim
import numpy as np
from tqdm import tqdm
from trackerPlot import plotPredictionWithBoxes,plotConning,anim2,anim3, plotTrajectory, plotTrackers, plotEstimatedStates, plotPredictedTrajectories


#MiS - Martin Sanfridson, RISE, April 2023


#'TRACKER_IDS' is the preferred index to access data

# 1. SPECIFY DATA SOURCE 
if 0:
    #seahorse doing a couple of turns in front of landkrabban
    args_metadataFilename = 'C:/Users/martinsa/OneDrive - RISE/2022-11-11 Nya varvet/2022-11-11-11-33-53_OS-2-128-992133000563-2048x10.json'
    args_pcapFilename = 'C:/Users/martinsa/OneDrive - RISE/2022-11-11 Nya varvet/2022-11-11-11-33-53_OS-2-128-992133000563-2048x10.pcap'
    if 0:
        #besvärlig början
        SCAN_RANGE = [50,349]
        TRAJ_NAME = 'traj20221111t1133_1.npz'
        PTS_NAME = 'pts20221111t1133_1.npz'
    if 0:
        #J-kroken som var första försökskaninen
        SCAN_RANGE = [350,550]
        #args_scanRange = [350,360]
        TRAJ_NAME = 'traj20221111t1133_2.npz'
        PTS_NAME = 'pts20221111t1133_2.npz'
        init_xypos = np.array([20,20])
        init_size = 10
    if 0:
        #besvärlig väldigt lång bort
        SCAN_RANGE = [551,900]
        TRAJ_NAME = 'traj20221111t1133_3.npz'
        PTS_NAME = 'pts20221111t1133_3.npz'
    if 1:
        #Ytterligare en J-kurva
        #SCAN_RANGE = [901,1300]
        SCAN_RANGE = [901,1300]
        TRACKER_IDS = [1] 
        MINDIST_M, MAXDIST_M, MINAZ_DEG, MAXAZ_DEG = 10.0, 90.0, 90, 230
        TRAJ_NAME = 'traj20221111t1133_4.npz'
        PTS_NAME = 'pts20221111t1133_4.npz'

if 1:
    #NB: not converted yet --> need to retune bbox finding, it doesn't work well in this scenario
    #lindholmen 13 okt
    #args_metadataFilename = 'C:/Users/martinsa/OneDrive - RISE/2022-10-13 LIDAR Lindholmen/2022-10-13-13-56-18_OS-2-128-992109000253-2048x10.json'
    #args_pcapFilename = 'C:/Users/martinsa/OneDrive - RISE/2022-10-13 LIDAR Lindholmen/2022-10-13-13-56-18_OS-2-128-992109000253-2048x10.pcap'
    args_metadataFilename = '/mnt/c/Users/martinsa/OneDrive - RISE/2022-10-13 LIDAR Lindholmen/2022-10-13-13-56-18_OS-2-128-992109000253-2048x10.json'
    args_pcapFilename = '/mnt/c/Users/martinsa/OneDrive - RISE/2022-10-13 LIDAR Lindholmen/2022-10-13-13-56-18_OS-2-128-992109000253-2048x10.pcap'
    if 1:
        #MOT: åtminstone två båtar som rör på sig
        #en av skärgårsbåtarna åker för bi och bort österut (roro åker in till Lindholmensbryggan)
        SCAN_RANGE = [170,200]
        #SCAN_RANGE = [170,170+250]
        TRAJ_NAME = 'traj20221013t1356_1.npz'
        PTS_NAME = 'pts20221013t1356_1.npz'
        TRACKER_IDS = [1,2] #expand to a list!
        MINDIST_M, MAXDIST_M, MINAZ_DEG, MAXAZ_DEG = 10.0, 150.0, 150, 360
    if 0:
        #snabbare skärgårdsbåten lång borta, från 100 m till ca 200 m
        SCAN_RANGE = [501,900]
        TRAJ_NAME = 'traj20221013t1356_2.npz'
        PTS_NAME = 'pts20221013t1356_2.npz'
    if 0:
        #roro backar ut från Lindholmsbryggan och snurrar runt
        SCAN_RANGE = [1400,1750]
        TRAJ_NAME = 'traj20221013t1356_3.npz'
        PTS_NAME = 'pts20221013t1356_3.npz'
    if 0:
        #roro parkerar på andra sidan
        SCAN_RANGE = [1400,1750]
        TRAJ_NAME = 'traj20221013t1356_4.npz'
        PTS_NAME = 'pts20221013t1356_4.npz'

if 0:
    #NB: not converted yet
    #seahorse joilting around
    args_metadataFilename = 'C:/Users/martinsa/OneDrive - RISE/2022-11-11 Nya varvet/2022-11-11-11-40-41_OS-2-128-992133000563-2048x10.json'
    args_pcapFilename = 'C:/Users/martinsa/OneDrive - RISE/2022-11-11 Nya varvet/2022-11-11-11-40-41_OS-2-128-992133000563-2048x10.pcap'
    if 0:
        SCAN_RANGE = [250,500]
        TRAJ_NAME = 'traj20221111t1140_1.npz'
        PTS_NAME = 'pts20221111t1140_1.npz'
    if 0:
        SCAN_RANGE = [501,760]
        TRAJ_NAME = 'traj20221111t1140_2.npz'
        PTS_NAME = 'pts20221111t1140_2.npz'


if 0:
    #swizer coming in to the key, is partly obscured at the end
    args_metadataFilename = 'C:/Users/martinsa/OneDrive - RISE/2022-11-11 Nya varvet/2022-11-11-11-26-34_OS-2-128-992133000563-2048x10.json'
    args_pcapFilename = 'C:/Users/martinsa/OneDrive - RISE/2022-11-11 Nya varvet/2022-11-11-11-26-34_OS-2-128-992133000563-2048x10.pcap'
    if 0:
        #ser aldrig hela båten, helt annan storlek, måste hitta bounding box med annan inställning
        SCAN_RANGE = [250,500]
        TRAJ_NAME = 'traj20221111t1126_1.npz'
        PTS_NAME = 'pts20221111t1126_1.npz'
    if 0:
        #OBS: har inte konverterat än - måste hitta bounding box med annan inställning
        #båten hinner inte lägga till innan den blir delvis skymd i fören
        SCAN_RANGE = [501,760]
        TRAJ_NAME = 'traj20221111t1126_2.npz'
        PTS_NAME = 'pts20221111t1126_2.npz'

samplingPeriod = 0.1 #seconds
#TODO: capitalize prediction options and move into above
predInterval = 1 #set interval==1 when plotting conning display
#predSteps = 25
predSteps = 50

# 2. READ DATA AND INIT MAIN LOOP
ouRead = ousterRead.ousterRead(args_metadataFilename,args_pcapFilename)
ouRead.pcapReadFrames(SCAN_RANGE)
scanDelta = SCAN_RANGE[1] - SCAN_RANGE[0]
preFlt = prefilterLive.prefilterLive(minDist_m=MINDIST_M,maxDist_m=MAXDIST_M,minAz_deg=MINAZ_DEG,maxAz_deg=MAXAZ_DEG) #90 to 270 is "forwards"
pcs, labels, obbMetaList, backgrounds, trackers = list(), list(), list(), list(), list()
#create tracker
mot_tracker = rectangleTracker.SimpleMOT(max_age=1,min_hits=2,iou_threshold=0.3) 
multiEst = motionEstim.multiMotionEstim(samplingPeriod,predInterval,predSteps)
motionRecord, motionPredict = list(), list() #hold last values and holds regularly spaced predictions

# 3. MAIN LOOP
for scanNb in tqdm(list(range(scanDelta))): #range(scanDelta):
    #TODO: if there are omitted scans, prediction should continue anyway
    #step 1, preprocess
    frame = ouRead.getFrame(scanNb)
    label, pc = preFlt.cropAndClusterFrame(frame=frame) 
    backgrounds.append(preFlt.convertToPC(frame["xyz"],frame["refl"]))
    pcs.append(pc) #just save
    labels.append(label) #just save
    
    #step 2, prepare candidate topview bounding boxes, to run IoU algorithm on
    detections = np.ndarray((label.max()+1,6),dtype=np.float64) #detections in the format [[x1,y1,x2,y2,score,label_id],...]
    for detId in range(label.max()+1):
        abb2D = preFlt.axleOrientedBoundingBoxes2d(pc,label == detId)
        detections[detId,0:4] = abb2D
        detections[detId,4] = 0 #score changes to idLabel
        detections[detId,5] = detId  #wrong type

    #step 3, update Kalman filters
    #matched[0] is index to Det, matched[1] is index to tracker, also: row number of 'detections' corresponds to id in 'tracker'
    tracker, matched, unmatched = mot_tracker.update(detections[:,:5]) #Kalman filter to find (IoU based) and track object
    trackers.append(tracker)
    #TODO: need to save states, prediction, covariance, obb3D with ID number
    if len(matched) > 0:
        obb3D_list = list() 
        idDets,idTrks = mot_tracker.getRecentDetectionList(TRACKER_IDS) #matching tuples
        for detId, trkId in zip(idDets,idTrks):
            obb3D, th = preFlt.orientedBoundingBoxes3d(pc,label == detId) #could get orientation too
            obb3D_list.append({'time': multiEst.getCurrentTime(), 'trkId': trkId, 'obb3D': obb3D}) 
            #TODO: get estimated size also
            multiEst.measureUpdate(trkId,obb3D,th) 
        obbMetaList.append(obb3D_list)

    #for all interesting ID:s, both matched and unmatched
    multiEst.predictAll() 
    
    #save generated result to lists
    state, cov = multiEst.getStateWithCovariance(trkId=TRACKER_IDS[0]) #estim_id
    if state is not None:
        motionRecord.append({'time': multiEst.getCurrentTime(), 'state':state, 'cov': cov})
    statePred, timePred = multiEst.getPredTrajWithTime(trkId=TRACKER_IDS[0])
    if len(statePred) > 0: #depends on 'predInterval'
        motionPredict.append({'timePred': timePred, 'statePred':statePred})
    multiEst.tickTack() #tick even if no trackers active


# 4. PLOT RESULTS
print("Plotting results")
#targetPlotId = [0,1,2,3,4,5] #select which tracked IDs to plot
#plotTrackers(trackers,targetPlotId)
#TODO: ändra limits till storlek på input
#plotEstimatedStates(motionRecord)
#plotPredictedTrajectories(motionRecord,motionPredict)
#TODO: need to correct when lenght and width do not match!
est = multiEst.estimators[0].bh.getLengthAndWidth()
estWidth, estLength = min(est), max(est)
#plotPredictionWithBoxes(motionRecord,motionPredict,predInterval,Width=estWidth,Length=estLength)
#anim3(pcs,obbMetaList,limits=[0, scanDelta-1],samplingTime=0.1) #TODO: need to extract point cloud from obbMetaList 
#TODO: check that motionRecord and motionPredict has same number of time stamps (predInterval==1)
#plotConning(motionRecord,motionPredict,predInterval,width=estWidth,length=estLength,xlim=[10,45],ylim=[-20,15],saveToFile=True)
plotConning(motionRecord,motionPredict,predInterval,predSteps,width=estWidth,length=estLength,xlim=[-20,150],ylim=[-100,70],saveToFile=False)
print("eof")
