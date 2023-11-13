from ouster import client, pcap
from colormaps import normalize
from itertools import islice
import argparse
import open3d as o3d
from tqdm import tqdm


#TODO: add function to save and read from hdf5 not having to keep all in memory

'''
MiS - Martin Sanfridson, RISE, October 2022
Read ouster pcap file. First step in a pipeline. Might be substiuted by reading directly from the sensor.
Save point cloud data in list, since size will vary.

'''

class ousterRead:

    def __init__(self,metadataFilename,pcapFilename) -> None:
        with open(metadataFilename, 'r') as f:
            self.metadata = client.SensorInfo(f.read())
        self.source = pcap.Pcap(pcapFilename, self.metadata) #problem if file too large?


    def pcapQuery(self) -> None:
        scans = iter(client.Scans(self.source))
        scan = next(scans)

        print(f"Number of scans {sum(1 for k in scans)}") #time consuming
        print("Available fields and corresponding dtype in LidarScan")
        for field in scan.fields:
            print('{0:15} {1}'.format(str(field), scan.field(field).dtype))


    def pcapReadFrames(self, scanRange) -> None:
        # get single scan by index
        #scan = nth(client.Scans(self.source), scanRange)
        scanRange[1] += 1 #add one since stop is before index, not at index
        scans = islice(iter(client.Scans(self.source)),scanRange[0],scanRange[1],1)
        self.xyz = list()
        self.refl = list()
        self.nearIR = list()
        self.signal = list()
        self.dist = list()
        for scan in tqdm(scans):
            #convert to point cloud etc using client.SensorInfo and client.LidarScan
            self.xyz.append(client.destagger(self.metadata,client.XYZLut(self.metadata)(scan))) #convert directly to open3d pointcloud type?
            self.refl.append(normalize(client.destagger(self.metadata,scan.field(client.ChanField.REFLECTIVITY))))
            nearIR = normalize(client.destagger(self.metadata,scan.field(client.ChanField.NEAR_IR)), percentile=0.01)
            #nearIR -= np.outer(nearIR.mean(axis=1),np.ones(nearIR.shape[1])) #is this good at all to do?
            self.nearIR.append(nearIR)
            self.signal.append(normalize(client.destagger(self.metadata,scan.field(client.ChanField.SIGNAL))))
            #self.signal.append(client.destagger(self.metadata,scan.field(client.ChanField.SIGNAL)))
            #self.dist.append(normalize(client.destagger(self.metadata,scan.field(client.ChanField.RANGE))))
            self.dist.append(client.destagger(self.metadata,scan.field(client.ChanField.RANGE)))
        self.count = len(self.dist)

    def getPC(self,nb):
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.xyz[nb].reshape((-1, 3))))  
        pc.paint_uniform_color([0,0,0])
        return pc
    
    def getFrame(self,nb):
        frame = {"xyz": self.xyz[nb],
                 "refl": self.refl[nb],
                 "dist": self.dist[nb]}
        return frame 

def main():
    #metadataFilename = 'C:/Users/martinsa/OneDrive - RISE/2022-10-13 LIDAR Lindholmen/2022-10-13-13-56-18_OS-2-128-992109000253-2048x10.json'
    #pcapFilename = 'C:/Users/martinsa/OneDrive - RISE/2022-10-13 LIDAR Lindholmen/2022-10-13-13-56-18_OS-2-128-992109000253-2048x10.pcap'
    #scanRange = [1983, 1984 + 1] #NB: stop index is not included
    #scanRange = [1, 2 + 1] #NB: stop index is not included
        
    parser = argparse.ArgumentParser()
    parser.add_argument("metadataFilename", help="Ouster json meta data")
    parser.add_argument("pcapFilename", help="Ouster pcap data file")
    parser.add_argument("startFrame", type=int, help="First frame number in the range")
    parser.add_argument("stopFrame", type=int, help="Last frame number in the range")
    args = parser.parse_args()
    scanRange = [args.startFrame, args.stopFrame]

    ou = ousterRead(args.metadataFilename,args.pcapFilename)
    ou.pcapQuery()
    ou.pcapReadFrames(scanRange)

#future use: cmd line    
if __name__ == "__main__":
    main()


