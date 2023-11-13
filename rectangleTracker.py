import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

'''
Much of the base is borrowed from:
    SORT: A Simple, Online and Realtime Tracker, Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai
My own adpatations:
 - Update parallel Kalman filters that have different purpose to avoid complexity of sharing 
   the same state and matrices 
 - Only want to track large objects that are moving, use visual flow to sort out
'''

#NOTE: kan vara fördel att ha separata Kalmanfilter för tracking och positionsestimering och koppla ihop dessa på annat 
# sätt så de får samma ID. Tracking baseras på kameravy snarare än top down





class KalmanBoxTracker(object):
    #This class represents the internal state of individual tracked objects observed as bbox.
    #states: x = [u, v, s, r, u', v',s'] where u and v are (pixel) positions, s is scale and r area
    count = 0
    def __init__(self,bbox,detId):
        #define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        self.detId = detId
        KalmanBoxTracker.count += 1
        self.history = [] #NOTE: eventually consumes memory, need to restrict
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self,bbox,detId):
        #Updates the state vector with observed bbox.
        self.detId = detId
        self.time_since_update = 0 #new meas
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))

    def predict(self):
        #Advances the state vector and returns the predicted bounding box estimate.
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return self.convert_x_to_bbox(self.kf.x)

    def estimMotionStates(self):
    #TODO: estimate state that can be used as initial states for the more accurate tracker
    #[x,dx,y,dy,c,dc], width and length
        state = self.get_state()
        x = (state[0]+state[2])/2
        y = (state[1]+state[3])/2
        return self.kf.x_post #use this to estimate motion, than add rotation by looking at principal

    def convert_bbox_to_z(self,bbox):
    # Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    #     [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    #     the aspect ratio
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h    #scale is just area
        if np.nonzero(float(h)): #added guard for runtime errors
            r = w / float(h)
            #print(f"h: {h}")
        else:
            r = 1
        return np.array([x, y, s, r]).reshape((4, 1))


    def convert_x_to_bbox(self,x,score=None):
    # Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    #     [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
        #w = np.sqrt(x[2] * x[3])
        #h = x[2] / w
        w = np.sqrt(np.abs(x[2]) * np.abs(x[3]))
        h = np.abs(x[2]) / w
        #print(f"w: {w}")
        if(score==None):
            return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
        else:
            return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))




class SimpleMOT(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        #Sets key parameters for SORT
        self.max_age = max_age #to remove dead tracker
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def linear_assignment(self,cost_matrix):
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y))) #why not just stack!?

    def iou_batch(self,bb_test, bb_gt):
        #From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
        #returned elements is in [0,1]
        #NOTE: makes very little sense adapted to rotation, and possibly to 3D
        bb_gt = np.expand_dims(bb_gt, 0)
        bb_test = np.expand_dims(bb_test, 1)

        xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
        yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
        yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
        return(o)



    def associate_detections_to_trackers(self,detections,trackers,iou_threshold = 0.3):
    #   Assigns detections to tracked object (both represented as bounding boxes)
    #   Returns 3 lists of matches, unmatched_detections and unmatched_trackers
        if(len(trackers)==0):
            return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

        iou_matrix = self.iou_batch(detections, trackers)

        #print(f"Size {iou_matrix.shape}, max {iou_matrix.max()}, min nonzero {iou_matrix[np.nonzero(iou_matrix)].min()}")

        #match what is currently detected and currently tracked
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = self.linear_assignment(-iou_matrix)
        else:
            matched_indices = np.empty(shape=(0,2))

        #list (possibly new) detected but not currently tracked
        unmatched_detections = []
        for d, det in enumerate(detections):
            if(d not in matched_indices[:,0]):
                unmatched_detections.append(d)
        #list trackers without current detection        
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if(t not in matched_indices[:,1]):
                unmatched_trackers.append(t)
                #unmatched_trackers.append()

        #filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if(iou_matrix[m[0], m[1]] < iou_threshold):
                unmatched_detections.append(m[0]) #e.g. vaguely detected but not (yet) tracked
                unmatched_trackers.append(-m[1]) #m[1] e.g. vaguely detected but not (yet) dead
            else:
                matches.append(m.reshape(1,2))
        if(len(matches)==0):
            matches = np.empty((0,2),dtype=int)
        else:
            matches = np.concatenate(matches,axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)



    def update(self, dets=np.empty((0, 5))):
        # Params:
        #   dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        # Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        # Returns the a similar array, where the last column is the object ID.

        # NOTE: The number of objects returned may differ from the number of detections provided.
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets,trks, self.iou_threshold)

        # update matched trackers with assigned detections, call Kalman filter update
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :],m[0])

        # create and initialise new trackers for unmatched detections
        for k in unmatched_dets:
            trk = KalmanBoxTracker(dets[k,:],-1)
            self.trackers.append(trk)
            #print(f"{self.frame_count}: Added new tracker {trk.id+1}" )

        #construct list of currently tracked
        #TODO: not very phythonic
        i = len(self.trackers)
        for trk in reversed(self.trackers): #reverse since pop is used?
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits): #time_since_update = last meas
                ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) #possibly add +1 as MOT benchmark requires positive (shifts index)
            i -= 1
            # remove dead tracker
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
                #print(f"{self.frame_count}: Removed dead tracker {trk.id}")
        if(len(ret)>0):
            ret = np.concatenate(ret)
        else:
            ret = np.empty((0,5))

        return ret, matched, unmatched_trks


    def getDetectionId(self,targetId):
        #consider make this method obsolete
        detId = -1
        time_since_update = 0
        for trk in self.trackers: #run reverse?
            if targetId == trk.id:
                time_since_update = trk.time_since_update
                if time_since_update == 0:
                    detId = trk.detId
                else:
                    trk.detId = -1 #flag as old
                break
        return detId, time_since_update
    

    def getRecentDetectionList(self,targetIds):
        #List of recently updated with new meas
        trkIds, detIds = list(), list()
        for trk in self.trackers: 
            if trk.time_since_update == 0 and trk.id in targetIds:
                trkIds.append(trk.id)
                detIds.append(trk.detId)
            #else:
                #trk.detId = -1 #flag as old
        return detIds, trkIds