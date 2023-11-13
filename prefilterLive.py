import copy
from math import acos
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from tqdm import tqdm
import boxHandlingUtils
from colormaps import colorize

'''
MiS - Martin Sanfridson, RISE, April 2022

Data strategy: do NOT keep pcl data in the object
'''

#TODO: find more pythonic way of repeating array within class
 
class prefilterLive():

    def __init__(self,minDist_m,maxDist_m,minAz_deg,maxAz_deg) -> None:
        self.minDist = minDist_m
        self.maxDist = maxDist_m
        self.minAz = minAz_deg*np.pi/180
        self.maxAz = maxAz_deg*np.pi/180 #todo: set when default
        self.voxelSize = 0.5 #downsampling before segmentation
        self.eps = 2.0
        self.minPoints = 10
        self.nb_points = 10
        self.radius = 1.0

    def cropFieldOfView(self,frame):
        xyz = frame["xyz"].copy()
        refl = frame["refl"].copy()
        dist = frame["dist"].copy()
        #apply cropping
        frameShp = dist.shape
        azimuth_rad = np.arange(0,2*np.pi,2*np.pi/frameShp[1]).reshape((1,frameShp[1])) #assume scanning of the whole horizon
        mask = np.logical_or(azimuth_rad > self.maxAz,azimuth_rad < self.minAz)
        refl[:, mask.flatten()] = 0 #consider using NaN instead
        xyz[:, mask.flatten(), :] = 0
        xyz = xyz*(np.logical_and(dist[:, :, np.newaxis] > self.minDist*1000,dist[:, :, np.newaxis] < self.maxDist*1000))
        refl[np.logical_and(dist <= self.minDist*1000,dist >= self.maxDist*1000)] = 0
        return xyz, refl #fixed sizes given by sensor
    
    def convertToPC(self,xyz,refl):
        #convert to PC, start by removing points with zero reflectivity
        ind = np.nonzero(refl.flatten())
        refl2 = refl.reshape(-1,1)[ind]
        xyz2 = xyz.reshape((-1, 3))[ind]
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz2))  #xyz.reshape((-1, 3))
        pc.colors = o3d.utility.Vector3dVector(colorize(refl2).reshape(-1,3)) #colorize(refl).reshape(-1,3)
        return pc

    def downSamplePC(self,pc): #want this new pc to be a copy(?)
        return pc.voxel_down_sample(self.voxelSize)
        #print(f"Point cloud size from {s0} down to {s1}, which is {s1//s0*100}%")

    def removeOutliers(self,pc):
        return pc.remove_radius_outlier(self.nb_points, self.radius)
    
    def getSize(self,pc):
        return len(pc.points)

    def clusterDBSCAN(self,pc):
        label = np.array(pc.cluster_dbscan(self.eps,self.minPoints, print_progress=False))
        max_label = label.max()
        #print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(label / (max_label if max_label > 0 else 1))
        colors[label < 0] = 0
        pc.colors = o3d.utility.Vector3dVector(colors[:, :3])
        return label, pc


    def cropAndClusterFrame(self,frame):
        #main function that calls others in this class
        xyz,refl = self.cropFieldOfView(frame)
        pc0 = self.convertToPC(xyz,refl) #programming strategy: keep data in object
        pc1 = self.downSamplePC(pc0) #copy used?
        label, pc1 = self.clusterDBSCAN(pc1)
        return label, pc1


    def principal(self,pc):
        #obb
        pcOut = copy.deepcopy(pc) 
        m, c = pc.compute_mean_and_covariance()
        D, V = np.linalg.eig(c)
        ind = np.argmax(D)
        u = V[:,ind]
        th = np.pi/2 + np.arctan2(u[0],u[1]) #rotation angle
        R = pc.get_rotation_matrix_from_xyz((0,0,th)) #rotate pc
        #pcOut = copy.deepcopy(pc)
        #abb = pcOut.get_axis_aligned_bounding_box()
        pcOut.rotate(R,m)
        obb = pcOut.get_axis_aligned_bounding_box()
        obb = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(obb)
        R = pc.get_rotation_matrix_from_xyz((0,0,-th)) #rotate bb back
        obb.rotate(R,m)
        return obb, th


    def orientedBoundingBoxes3d(self,pc,index): #labels,id):
        #find principal and reorient cloud
        xyz = np.asarray(pc.points)[index] #[labels == id,:]
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        #self.removeVesselOutliers(ou,nb)
        #self.removeVesselsWaterline(ou,nb)
        obb, th = self.principal(pc)
        obb.color = (0, 1, 0)
        return obb, th
    


    def axleOrientedBoundingBoxes2d(self,pc,index):
        #axis aligned bb in format for "simpleMOT"
        #pcOut = copy.deepcopy(pc)
        xyz = np.asarray(pc.points)[index] #[labels == id,:]
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        abb = pc.get_axis_aligned_bounding_box()
        #abb.color = (1, 1, 0)
        cornerPts = np.asarray(abb.get_box_points())
        return boxHandlingUtils.bbox3DTo2D_twoPoints(cornerPts)


    def selectLabelInPc(self,pc,index):
        xyz = np.asarray(pc.points)[index] #[labels == id,:]
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        return pc

'''
    def findVessel(self,nb):
        #TODO: when initial position and size are unknown, how to not reject? THIS IS PART OF TRACKING ENGINE
        found = False
        points = np.asarray(self.PCs[nb].points)
        for ID in range(self.labels[nb].max()):
            pts = points[self.labels[nb] == ID,:]
            new_xypos = np.mean(pts,axis=0)[0:2]
            ptp = np.ptp(pts,axis=0)
            new_size = ptp[0]*ptp[1] #np.linalg.norm(np.ptp(pts,axis=0)[0:2])
            #TODO: this should be improved! magic numbers, also: want to continue tracking!
            if np.linalg.norm(new_xypos - self.xypos) < 30: #check if overlap in position
                if np.abs(new_size - self.size) < 100: #check if overlap in size
                    found = True
                    break
        if found:
            self.xypos = new_xypos
            self.size = new_size
            self.vessels.append(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))) 
        else:
            #pts = np.zeros((1,3)) #no good to insert zeros, should discard instead
            if len(self.vessels) > 0:
                self.vessels.append(self.vessels[-1])
            print(f"There might be a vessel at {new_xypos} of size {new_size}")
        return found
    


    def removeVesselOutliers(self,ou,nb):
        points = o3d.utility.Vector3dVector(ou.xyz[nb].reshape((-1, 3)))
        ind = self.obbs[nb].get_point_indices_within_bounding_box(points) 
        #_ , ind = self.vessels[nb].remove_radius_outlier(nb_points=20, radius=0.5)
        #self.vessels[nb].select_by_index(ind)
        self.vessels[nb] = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points)[ind,:]))
        self.vessels[nb], ind = self.vessels[nb].remove_radius_outlier(nb_points=20, radius=0.5)
        #TODO: remove statistically instead? or by application domain
        #pc, _ = self.PCs[nb].remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0)


    def removeVesselsWaterline(self,ou,nb):
        #assume z is vertical direction and remove lowest 10%
        points = o3d.utility.Vector3dVector(ou.xyz[nb].reshape((-1, 3)))
        ind1 = self.obbs[nb].get_point_indices_within_bounding_box(points) 
        z = np.asarray(points)[ind1,2] #find those with z > pct
        pct = np.percentile(z,20)
        ind2 = np.argwhere(z > pct)
        ind = np.asarray(ind1)[ind2]
        self.vessels[nb] = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.squeeze(np.asarray(points)[ind])))


        
                    #self.extractVessel(ID=0,nb=nb) 
            #TODO: code below should not really be in this method but be moved to Tracking algorithm "engine"
            #self.findVessel(nb=nb) #tracking approximately xy-pos
            #TODO: what if vessel is not found
            #print(nb,self.xypos,self.size)
            #self.orientedBoundingBox(nb)
            #if True: #try  move such that reselecting becomes unnecessary
                #self.removeVesselOutliers(ou,nb)
            #    self.removeVesselsWaterline(ou,nb)
            #    pc = self.vessels[nb]
            #    bb, th = self.principal(pc)
            #    bb.color = (0, 1, 0)
            #    self.obbs[nb] = bb

'''




'''
Move these to next higher level where data is aggregated

    def saveFrame(self,filename,nb):
        o3d.io.write_point_cloud(filename, self.PCs[nb])

    def saveObjectPointCloudToFile(self,filename,ou):
        #saves a list of xyz points for the object of interest
        arr_xyz = list()
        arr_refl = list()
        for xyz,refl,obb in zip(ou.xyz,ou.refl,self.obbs):
            points = o3d.utility.Vector3dVector(xyz.reshape((-1, 3)))
            ind = obb.get_point_indices_within_bounding_box(points)        
            arr_xyz.append(xyz.reshape((-1,3))[ind])
            arr_refl.append(refl.flatten()[ind])

        np.savez(filename,xyz=np.asarray(arr_xyz,dtype=object),refl=np.asarray(arr_refl,dtype=object))
'''

