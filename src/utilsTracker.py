

import numpy as np
import pandas as pd

def box_center_to_corner3D(center,extension,rotation_matrix):
    l, w, h = extension[0], extension[1], extension[2]
    bounding_box = np.array([
        [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
        [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
        [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]])
    # Repeat the [x, y, z] eight times
    translate_points = np.tile(center, (8, 1))
    # Translate the rotated bounding box by the original center position to obtain the final box
    corner_box = np.dot(rotation_matrix, bounding_box) + translate_points.transpose()
    return corner_box.transpose()

def boxPointPopulation2D(l,w):
    bboxPointsTemplate = np.zeros((4,2))
    bboxPointsTemplate[0,0]= -w/2
    bboxPointsTemplate[0,1] = l/2
    bboxPointsTemplate[1,0] = w/2
    bboxPointsTemplate[1,1] = l/2
    bboxPointsTemplate[2,0] = w/2
    bboxPointsTemplate[2,1] = -l/2
    bboxPointsTemplate[3,0] = -w/2
    bboxPointsTemplate[3,1] = -l/2
    return bboxPointsTemplate

def rotate2D(boxPoints,ang):
    R = np.squeeze(np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]]))
    for k2 in range(4):
        boxPoints[k2,0:2,np.newaxis] = np.dot(R,boxPoints[k2,:,np.newaxis])
    return boxPoints

def shift2D(boxPoints,dist):
    boxPoints[:,0] += dist[0]
    boxPoints[:,1] += dist[1]
    return boxPoints

#combination of creating corners, rotate and translate for 2D case
def boxToCorners2D(L,W,midpoint,angle):
    bboxPoints = boxPointPopulation2D(L,W) #is S1 == L guaranteed?
    bboxPoints = rotate2D(bboxPoints,angle)
    bboxPoints = shift2D(bboxPoints,midpoint)
    return bboxPoints


def getID(trackers,targetId):
    df = pd.DataFrame(np.concatenate(trackers), columns=['x1', 'y1', 'x2', 'y2','id'])
    df["id"] = df["id"].astype(int)
    allIds = df["id"].unique()
    if len(targetId) > 0: #remove IDs not to display
        df = df[df["id"].isin(targetId)]
    ids = df["id"].unique()
    maxId = len(ids)
    print(f"Selected {maxId} tracks")
