import open3d as o3d
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import pandas as pd
import utilsTracker
from collections import deque

'''
MiS - Martin Sanfridson, RISE, November 2022
A collection of plot and animate functions.

'''

def anim(ou,op,limits,samplingTime):
    frameNb = 0
    prevNb = -1
    cloud = ou[frameNb] #list of o3d pointclouds
    if len(op) == 1:
        obb = op[0]
    else:
        obb = op[frameNb] #equally long list of o3d.geometry.OrientedBoundingBox
    pauseFlag = True
    t = time.time()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    viewParam = None
    viewParamsFilename = 'viewparams.json'


    def next_one(vis):
        nonlocal frameNb, prevNb, viewParam
        frameNb += 1
        frameNb = min(frameNb,limits[1]-1)
        viewParam = vis.get_view_control().convert_to_pinhole_camera_parameters()

    def prev_one(vis):
        nonlocal frameNb, prevNb, viewParam
        frameNb -= 1
        frameNb = max(frameNb,limits[0])
        viewParam = vis.get_view_control().convert_to_pinhole_camera_parameters()

    def animation_callback(vis): 
        nonlocal t, pauseFlag, frameNb, prevNb, cloud, obb, viewParam
        if not pauseFlag and (time.time()-t > samplingTime):
            next_one(vis)
            t = time.time()
            

        if frameNb != prevNb:
            #print(frameNb)
            prevNb = frameNb
            vis.remove_geometry(cloud,False) #doesnt work with "True"
            vis.remove_geometry(obb,False) 
            cloud = ou[frameNb]
            if len(op) == 1:
                obb = op[0]
            else:
                obb = op[frameNb] #equally long list of o3d.geometry.OrientedBoundingBox
            #obb = op[frameNb]
            obb.color = [1, 0, 0]
            vis.add_geometry(cloud,True)
            #mesh_box = o3d.geometry.TriangleMesh.create_box(width=30.0, height=30.0, depth=0.1)
            #mesh_box.paint_uniform_color([0.5,0,0])
            #vis.add_geometry(mesh_box)
            vis.add_geometry(obb,True)
            vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(5.0))
            if viewParam is None:
                viewParam = o3d.io.read_pinhole_camera_parameters(viewParamsFilename)
            vis.get_view_control().convert_from_pinhole_camera_parameters(viewParam)
    
        vis.update_geometry(cloud)
        vis.update_geometry(obb)
     


    def reset_sequence(vis):
        nonlocal frameNb, prevNb
        frameNb = 0
        prevNb = -1
        

    def pause_or_resume(vis):
        nonlocal pauseFlag, viewParam
        viewParam = vis.get_view_control().convert_to_pinhole_camera_parameters()
        pauseFlag = not pauseFlag

        
 
    # key_action_callback will be triggered when there's a keyboard press, release or repeat event
    vis.register_key_callback(ord("."), next_one)
    vis.register_key_callback(ord(","), prev_one)
    vis.register_key_callback(ord(" "), pause_or_resume)
    vis.register_key_callback(ord("M"), reset_sequence)
    

    # animation_callback is always repeatedly called by the visualizer
    vis.register_animation_callback(animation_callback)

    #vis.set_full_screen(True)
    vis.create_window()
    
    #read view params if any
    param = o3d.io.read_pinhole_camera_parameters(viewParamsFilename)
    vis.get_view_control().convert_from_pinhole_camera_parameters(param)

    vis.run()
    #save view params
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(viewParamsFilename,param)

    vis.destroy_window()


def plotTrajectory(obbs,ths):
    fig, axs = plt.subplots(3,1)
    centers = list()
    extents = list()
    angles = list()
    for item in obbs:
        centers.append(item.center)
        extents.append(item.extent)
        angles.append(item.R)

    centers = np.asarray(centers)
    axs[0].plot(centers[:,0],centers[:,1],'.-')
    axs[0].set_title('Travelled path')
    axs[1].plot(extents)
    axs[1].legend(['length','width','height'],loc='center')
    axs[1].set_title('Dimensions')
    axs[2].plot(np.asarray(ths)[:,4],marker='.')
    axs[2].set_title('Angle')
    plt.show()


def plotBBox(bbox,ths):
    fig, axs = plt.subplots(3,1)
    axs[0].plot(bbox['c'][:,0],bbox['c'][:,1])
    axs[0].set_title('Travelled path')
    axs[1].plot(bbox['e'])
    axs[1].legend(['length','width','height'],loc='center')
    axs[1].set_title('Dimensions')
    axs[2].plot(ths['th'],marker='.')
    axs[2].set_title('Angle')
    plt.show()

def plotAngles(ths):
    fig, axs = plt.subplots(1,1)
    axs.plot(ths,marker='.')
    axs.set_title('Angle')
    axs.grid(True)
    plt.show()

#---

def anim2(*args,limits,samplingTime):
    frameNb = 0
    prevNb = None
    pcs = list()
    for pc in args: #does not work if one arg is a list itself
        pcs.append(pc)
    pauseFlag = True
    t = time.time()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    viewParam = None
    viewParamsFilename = 'viewparams.json'


    def next_one(vis):
        nonlocal frameNb, prevNb, viewParam
        frameNb += 1
        frameNb = min(frameNb,limits[1]-1)
        viewParam = vis.get_view_control().convert_to_pinhole_camera_parameters()

    def prev_one(vis):
        nonlocal frameNb, prevNb, viewParam
        frameNb -= 1
        frameNb = max(frameNb,limits[0])
        viewParam = vis.get_view_control().convert_to_pinhole_camera_parameters()

    def animation_callback(vis): 
        nonlocal t, pauseFlag, frameNb, prevNb, pcs, viewParam
        if not pauseFlag and (time.time()-t > samplingTime):
            next_one(vis)
            t = time.time()
            

        if frameNb != prevNb:
            for pc in pcs:
                if prevNb is not None:
                    vis.remove_geometry(pc[prevNb],False)
            prevNb = frameNb
            for pc in pcs:
                vis.add_geometry(pc[frameNb],True)
            if viewParam is None:
                viewParam = o3d.io.read_pinhole_camera_parameters(viewParamsFilename)
            vis.get_view_control().convert_from_pinhole_camera_parameters(viewParam)
    
        for pc in pcs:
            vis.update_geometry(pc[frameNb])
     


    def reset_sequence(vis):
        nonlocal frameNb, prevNb, pcs
        for pc in pcs:
            if prevNb is not None:
                vis.remove_geometry(pc[prevNb],False)
        frameNb = 0
        prevNb = None
        

    def pause_or_resume(vis):
        nonlocal pauseFlag, viewParam
        viewParam = vis.get_view_control().convert_to_pinhole_camera_parameters()
        pauseFlag = not pauseFlag

        
 
    # key_action_callback will be triggered when there's a keyboard press, release or repeat event
    vis.register_key_callback(ord("."), next_one)
    vis.register_key_callback(ord(","), prev_one)
    vis.register_key_callback(ord(" "), pause_or_resume)
    vis.register_key_callback(ord("M"), reset_sequence)
    

    # animation_callback is always repeatedly called by the visualizer
    vis.register_animation_callback(animation_callback)

    #vis.set_full_screen(True)
    vis.create_window()
    
    #read view params if any
    param = o3d.io.read_pinhole_camera_parameters(viewParamsFilename)
    vis.get_view_control().convert_from_pinhole_camera_parameters(param)

    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(5.0))

    vis.run()
    #save view params
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(viewParamsFilename,param)

    vis.destroy_window()

#---


def plotICPresults(icps):
    #for icp in icps:
    #    print(f"Fitness {icp.fitness}, rmse {icp.inlier_rmse}, angle {icp.transformation[0,0]}")
    def fig1(icps):
        fig, axs = plt.subplots(2,1)
        axs[0].plot([item.fitness for item in icps])
        axs[0].set_title('fitness')
        axs[1].plot([item.inlier_rmse for item in icps])
        axs[1].set_title('rmse for inliers')

    def fig2(icps):
        fig, axs = plt.subplots(3,1)
        x = [item.transformation[0,3] for item in icps]
        y = [item.transformation[1,3] for item in icps]
        axs[0].plot(x,y,'.')
        axs[0].set_title('Travelled path')
        axs[1].plot(x,'.-')
        axs[1].plot(y,'-')
        axs[1].set_title('Coordinate in time')
        axs[2].plot([180/np.pi*np.arccos(item.transformation[0,0]) for item in icps])
        axs[2].set_title('Angle [deg]')

    fig1(icps)
    fig2(icps)
    plt.show()



# if __name__ == "__main__":
#     pcd_data = o3d.data.DemoICPPointClouds()
#     limits = [0,len(pcd_data.paths)]
    
#     anim(pcd_data,limits=limits,samplingTime=1.2)

# Allow for list of lists in pcd_data
#---

def anim3(*args,limits,samplingTime):
    #TODO: check that limits is less than input vectors
    frameNb = 0
    prevNb = None
    pcs = list()
    for pc in args: #does not work if one arg is a list itself
        pcs.append(pc)
    pauseFlag = True
    t = time.time()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    viewParam = None
    viewParamsFilename = 'viewparams.json'


    def next_one(vis):
        nonlocal frameNb, prevNb, viewParam
        frameNb += 1
        frameNb = min(frameNb,limits[1]-1)
        viewParam = vis.get_view_control().convert_to_pinhole_camera_parameters()

    def prev_one(vis):
        nonlocal frameNb, prevNb, viewParam
        frameNb -= 1
        frameNb = max(frameNb,limits[0])
        viewParam = vis.get_view_control().convert_to_pinhole_camera_parameters()

    def animation_callback(vis): 
        nonlocal t, pauseFlag, frameNb, prevNb, pcs, viewParam
        if not pauseFlag and (time.time()-t > samplingTime):
            next_one(vis)
            t = time.time()
            

        if frameNb != prevNb:
            for pc1 in pcs: #could even do it recursively...
                if prevNb is not None:
                    if isinstance(pc1[0],list):
                        for pc2 in pc1[prevNb]:
                            vis.remove_geometry(pc2,False)
                    else:
                        vis.remove_geometry(pc1[prevNb],False)
            prevNb = frameNb
            for pc1 in pcs:
                if isinstance(pc1[0],list):
                    for pc2 in pc1[frameNb]:
                        vis.add_geometry(pc2,True)
                else:
                    vis.add_geometry(pc1[frameNb],True)
            if viewParam is None:
                viewParam = o3d.io.read_pinhole_camera_parameters(viewParamsFilename)
            vis.get_view_control().convert_from_pinhole_camera_parameters(viewParam)
    
        for pc1 in pcs:
            if isinstance(pc1[0],list):
                for pc2 in pc1[frameNb]:
                    vis.update_geometry(pc2)
            else:
                vis.update_geometry(pc1[frameNb])


    def reset_sequence(vis):
        nonlocal frameNb, prevNb, pcs
        for pc1 in pcs:
            if prevNb is not None:
                if isinstance(pc1[0],list):
                    for pc2 in pc1[prevNb]:
                        vis.remove_geometry(pc2,False)
                else:
                    vis.remove_geometry(pc1[prevNb],False)
        frameNb = 0
        prevNb = None
        

    def pause_or_resume(vis):
        nonlocal pauseFlag, viewParam
        viewParam = vis.get_view_control().convert_to_pinhole_camera_parameters()
        pauseFlag = not pauseFlag

        
 
    # key_action_callback will be triggered when there's a keyboard press, release or repeat event
    vis.register_key_callback(ord("."), next_one)
    vis.register_key_callback(ord(","), prev_one)
    vis.register_key_callback(ord(" "), pause_or_resume)
    vis.register_key_callback(ord("M"), reset_sequence)
    

    # animation_callback is always repeatedly called by the visualizer
    vis.register_animation_callback(animation_callback)

    #vis.set_full_screen(True)
    vis.create_window()
    
    #read view params if any
    param = o3d.io.read_pinhole_camera_parameters(viewParamsFilename)
    vis.get_view_control().convert_from_pinhole_camera_parameters(param)

    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(5.0))

    vis.run()
    #save view params
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(viewParamsFilename,param)

    vis.destroy_window()

#---


def plotTrackers(trackers,targetId):
    df = pd.DataFrame(np.concatenate(trackers), columns=['x1', 'y1', 'x2', 'y2','id'])
    df["id"] = df["id"].astype(int)
    allIds = df["id"].unique()
    if len(targetId) > 0: #remove IDs not to display
        df = df[df["id"].isin(targetId)]
    ids = df["id"].unique()
    maxId = len(ids)
    #print(f"Selected {maxId} tracks")
    fig, axs = plt.subplots(maxId+1,1) #add an extra to use axs[] workaround
    for axNb in ids:
        axs[axNb].plot(df[df["id"] == ids[axNb-1]]["x1"],df[df["id"] == ids[axNb-1]]["y1"])
        axs[axNb].set_title(ids[axNb-1])
    axs[maxId].plot(allIds,'x')
    axs[maxId].set_title("All indices")

    plt.show()


def getStatesDataFrame(motionRecord):
    df0 = pd.DataFrame.from_dict(motionRecord)
    df = pd.DataFrame(np.hstack((df0['time'].values.reshape((-1,1)),np.vstack(df0['state'].values))),columns=['t','x','dx','y','dy','c','dc'])
    return df

def plotEstimatedStates(motionRecord):
    #updated version
    df = getStatesDataFrame(motionRecord)
    fig, axs = plt.subplots(4,1)
    axs[0].plot(df["x"],df["y"],'x-')
    axs[0].set_ylabel("xy-plot [m]")
    #axs[1].plot(np.hstack((df["dx"], df["dy"])),'x-')
    axs[1].plot(df['t'],df["dx"],'x-')
    axs[1].plot(df['t'],df["dy"],'x-')
    axs[1].set_ylabel("x and y speed [m/s]")
    axs[2].plot(df['t'],df["c"],'x-')
    axs[2].set_ylabel("rotation [rad]")
    axs[3].plot(df['t'],df["dc"],'x-')
    axs[3].set_ylabel("rot rate [rad/s]")
    axs[0].set_title("Plot of state vector")
    plt.show()


def plotPredictedTrajectories(motionRecord,motionPredict):
    #plot of evolving state and with periodic predictions
    df = getStatesDataFrame(motionRecord)
    dfp = pd.DataFrame.from_dict(motionPredict)
    fig,axs = plt.subplots(2,1,sharex=True)
    fig.subplots_adjust(hspace=0)
    fig.suptitle("Prediction, position and orientation")
    #position and orientation
    axs[0].plot(df['t'],df['x'])
    axs_right = axs[0].twinx()
    axs_right.plot(df['t'],df['y'])
    for index, row in dfp.iterrows():
        t = np.asarray(row['timePred'])
        v = np.vstack(row['statePred'])
        axs[0].plot(t,v[:,0],':')
        axs_right.plot(t,v[:,2],':')
    axs[0].set_ylabel("[m]")
    #next axis with rotation
    axs[1].plot(df['t'],df['c'],'r')
    axs_right = axs[1].twinx()
    axs_right.plot(df['t'],df['dc'],'b')
    for index, row in dfp.iterrows():
        t = np.asarray(row['timePred'])
        v = np.vstack(row['statePred'])
        axs[1].plot(t,v[:,4],'r:')
        axs_right.plot(t,v[:,5],'b:')
    axs[1].set_ylabel("[rad]")
    axs[1].set_xlabel("[s]")
    plt.show()
    

def bboxFromStateVector(vecMotion,W,L):
    #auxiliary, convert from result vector to corner points
    bbox = np.zeros((len(vecMotion),5,2)) #4 of x and y
    for k in range(len(vecMotion)):
        bbox[k,0:4,:] = utilsTracker.boxToCorners2D(W,L,[vecMotion[k,0],vecMotion[k,2]],vecMotion[k,4])
        bbox[k,4,:] = bbox[k,0,:] 
    return bbox 


def plotPredictionWithBoxes(motionRecord,motionPredict,predInterval,Width,Length):
    #plots xy trace with predictions, plots last bbox in prediction sequence
    #TODO: need to sync using time vector instead, slips might change result otherwise

    df = getStatesDataFrame(motionRecord)
    dfp = pd.DataFrame.from_dict(motionPredict)

    fig,axs = plt.subplots(1,1,sharex=True)
    fig.subplots_adjust(hspace=0)
    #fig.suptitle("Prediction")
    #plot trajectory
    axs.plot(df['x'],df['y'],'k',linewidth=2.0)
    #plot predicted state
    for _ , row in dfp.iterrows():
        v = np.vstack(row['statePred'])
        axs.plot(v[:,0],v[:,2],'k:',linewidth=1.0)
    
    #plot bboxes for trajectory at 'predInterval'
    axs.set_prop_cycle(None)
    v = np.asarray((df['x'],df['dx'],df['y'],df['dy'],df['c'],df['dc'])).T
    bbox = bboxFromStateVector(v,Width,Length)
    for k in range(1,len(v)):
        if k % predInterval == 0 or k == len(v)-1:
            axs.plot(bbox[k,:,0],bbox[k,:,1])
    
    #plot last predicted trajectories and final bboxes
    plt.gca().set_prop_cycle(None)
    for _ , row in dfp.iterrows():
        v = row['statePred'][-1].reshape((1,6))
        bbox = bboxFromStateVector(v,Width,Length)
        axs.plot(bbox[0,:,0],bbox[0,:,1],':',linewidth=0.5)

    plt.axis('equal')
    plt.grid(visible=True)
    axs.set_xlabel('[m]')
    axs.set_ylabel('[m]')
    plt.show()
    #xlim = axs.get_xlim()
    #ylim = axs.get_ylim()


def animateTopView3(xvecMotion,xvecMotionPredCollection,predInterval,xlim,ylim,velLong,velLat_bow,velLat_stern,width,length,saveToFile):

    #--- setup figure with a set of axes
    gs_kw = dict(width_ratios=[5.0, 2.0], height_ratios=[1, 2])
    fig, axd = plt.subplot_mosaic([['left', 'upper right'],
                                ['left', 'lower right']],
                                figsize=(8.0, 4.0),gridspec_kw=gs_kw)
    #fig.tight_layout()
  
    axd['upper right'].set_aspect('equal')
    axd['upper right'].set_axis_off()
    #axd['lower right'].set_xtick=[]
    #axd['lower right'].set_ytick=[]
    #axd['lower right'].set_axis_off()
    axd['lower right'].get_yaxis().set_visible(False)
    axd['lower right'].get_xaxis().set_visible(False)
    #--- setup the main plot
    n = len(xvecMotion)
    bboxes = bboxFromStateVector(xvecMotion,width,length)

    axd['left'].set_xlim(xlim)
    axd['left'].set_ylim(ylim)
    axd['left'].set_title("Animation of trace and prediction")
    axd['left'].set_xlabel("[m]")
    axd['left'].set_ylabel("[m]")
    history_x = deque(maxlen=n) #for making a trace!
    history_y = deque(maxlen=n) #for making a trace!
    trace, = axd['left'].plot([], [], 'ko', lw=0.5, ms=2)
    curRect, = axd['left'].plot([], [], 'b', lw=1, ms=2)
    pred, = axd['left'].plot([], [], 'r:', lw=1, ms=2)
    predRect0, = axd['left'].plot([], [], 'b:', lw=0.8, ms=2)
    predRect1, = axd['left'].plot([], [], 'b:', lw=0.4, ms=2)
    
    #--- setup lat vel chart
    v_bw = -velLat_bow #change of coord system
    v_st = -velLat_stern
    v_long = velLong
    #n = len(v_long)

    #position of parts of the plot
    y_bw = 1
    dy_bw = 0
    y_st = -1
    dy_st = 0
    x_sb = 1
    x_pt = 0
    x_long = (x_sb + x_pt)/2
    #y_long = (y_bw + y_st)/2
    y_long = y_bw
    v_lat_max = 2 #scaling factor for arrow
    v_long_max = 3.0

    #TODO: need to scale input to make arrows of appropriate lengths
    #TODO: make an init-funtion to clean up
    #TODO: make functions in order to clean up

    axd['upper right'].set_xlim((-2.0, 2.0))
    axd['upper right'].set_ylim((-2.0, 3.0))
    #axd['upper right'].set_facecolor('b')

    arrow_bw = mpatches.Arrow(x_sb, y_bw, v_bw[0]/v_long_max, dy_bw,width=0.5)
    arrow_st = mpatches.Arrow(x_pt, y_st, v_st[0]/v_lat_max, dy_st,width=0.5)
    #arrow_long = mpatches.Arrow(x_pt, y_st, v_long[0]/v_long_max, dy_st,width=0.5)
    arrow_long = mpatches.Arrow(x_pt, y_st, dy_st,v_long[0]/v_long_max,width=0.5)
    axd['upper right'].add_patch(arrow_bw)
    axd['upper right'].add_patch(arrow_st)
    axd['upper right'].add_patch(arrow_long)
    trace_bw, = axd['upper right'].plot([], [], 'ko', lw=1, ms=2)
    trace_st, = axd['upper right'].plot([], [], 'ko', lw=1, ms=2)
    trace_long, = axd['upper right'].plot([], [], 'ko', lw=1, ms=2)

    #--- setup text box
    #text0 = axd['lower right'].text(0.05, 0.9, '', transform=axd['upper right'].transAxes)
    text0 = axd['lower right'].annotate('test 1',xy=(0.05, 0.8), xycoords='axes fraction')
    text1 = axd['lower right'].annotate('test 2',xy=(0.05, 0.7), xycoords='axes fraction')
    text2 = axd['lower right'].annotate('test 3',xy=(0.05, 0.6), xycoords='axes fraction')
    text3 = axd['lower right'].annotate('test 4',xy=(0.05, 0.5), xycoords='axes fraction')

    history_bw = deque(maxlen=n) #for making a trace!
    history_st = deque(maxlen=n) #for making a trace!
    history_long = deque(maxlen=n) #for making a trace!

    #TODO: remove a lot of hardcoding of numbers

    def animate(k):
        #--- animate main 
        if k == 0:
            history_x.clear()
            history_y.clear()
        history_x.appendleft(xvecMotion[k,0])
        history_y.appendleft(xvecMotion[k,2])
        #plot vessel and trace
        curRect.set_data(bboxes[k,:,0],bboxes[k,:,1])
        trace.set_data(history_x,history_y)
        #plot prediction

        #TODO: use time to invoke this instead
        #NOTE: should change settings to predict at every time step, regardless of predInterval
        # if k % predInterval == 0:
        #     k2 = k/predInterval
        #     vecMotionPred = xvecMotionPredCollection[k2,...]
        #     pred.set_data(vecMotionPred[:,0],vecMotionPred[:,2])
        #     vecMotion = bboxFromStateVector(vecMotionPred,width,length) 
        #     predRect0.set_data(vecMotion[len(vecMotion)//2,:,0],vecMotion[len(vecMotion)//2,:,1]) 
        #     predRect1.set_data(vecMotion[-1,:,0],vecMotion[-1,:,1])
        vecMotionPred = xvecMotionPredCollection[k,...]
        pred.set_data(vecMotionPred[:,0],vecMotionPred[:,2])
        vecMotion = bboxFromStateVector(vecMotionPred,width,length) 
        predRect0.set_data(vecMotion[len(vecMotion)//2,:,0],vecMotion[len(vecMotion)//2,:,1]) 
        predRect1.set_data(vecMotion[-1,:,0],vecMotion[-1,:,1])

        #--- animate latvel
        if k == 0:
            history_bw.clear()
            history_st.clear()
            history_long.clear()

        #MiS 2023-10-17 need to empty patches, axd['upper right'].patches = [], new version of matplotlib
        for removeItem in axd['upper right'].patches:
            removeItem.remove()
        #bow
        if v_bw[k] > 0: #starboard
            arrow_bw = mpatches.Arrow(x_sb, y_bw, v_bw[k], dy_bw,width=0.5,color=(0,1,0))
            history_bw.appendleft(x_sb+v_bw[k])
        else: #port
            arrow_bw = mpatches.Arrow(x_pt, y_bw, v_bw[k], dy_bw,width=0.5,color=(1,0,0))
            history_bw.appendleft(x_pt+v_bw[k])
        #stern
        if v_st[k] > 0:
            arrow_st = mpatches.Arrow(x_sb, y_st, v_st[k], dy_st,width=0.5,color=(0,1,0))
            history_st.appendleft(x_sb+v_st[k])
        else:
            arrow_st = mpatches.Arrow(x_pt, y_st, v_st[k], dy_st,width=0.5,color=(1,0,0))
            history_st.appendleft(x_pt+v_st[k])

        arrow_long = mpatches.Arrow(x_long, y_long, 0, v_long[k]/v_long_max,width=0.5,color=(0,0,0))
        #plot hull and arrows
        axd['upper right'].add_patch(mpatches.Rectangle((x_pt,y_st),x_sb-x_pt,y_bw-y_st,alpha=0.5,color=(0,0,1)))
        axd['upper right'].add_patch(mpatches.Polygon([[x_pt,y_bw],[(x_pt+x_sb)/2,y_bw+0.3],[x_pt,y_st]],closed=True,alpha=0.5,color=(0,0,1)))
        axd['upper right'].add_patch(mpatches.Polygon([[x_sb,y_bw],[(x_pt+x_sb)/2,y_bw+0.3],[x_sb,y_st]],closed=True,alpha=0.5,color=(0,0,1)))
        axd['upper right'].add_patch(arrow_bw)
        axd['upper right'].add_patch(arrow_st)
        axd['upper right'].add_patch(arrow_long)
        if len(history_bw) > 2: #not correct way to do it
            trace_bw.set_data(history_bw[2],y_bw)
        if len(history_bw) > 1:
            trace_bw.set_data(history_bw[1],y_bw)
        trace_bw.set_data(history_bw[0],y_bw)
        if len(history_st) > 1:
            trace_st.set_data(history_st[1],y_st)
        else:
            trace_st.set_data(history_st[0],y_st)
        
        text0.set_text(f'SOG  {3600/1852*np.abs(v_long[k]):.1f} [kn]')
        text1.set_text(f'ROG  {180/np.pi*xvecMotion[k,5]:.1f} [deg/s]')
        text2.set_text(f'LSB  {3600/1852*v_bw[k]:.1f} [kn]')
        text3.set_text(f'LSS  {3600/1852*v_st[k]:.1f} [kn]')
        #text0.set_text(f'SOG  {np.abs(v_long[k]):.2f} [m/s]')
        #text1.set_text(f'ROG  {xvecMotion[k,5]:.2f} [rad/s]')
        return trace, curRect, pred, predRect0, predRect1, arrow_bw, arrow_st, arrow_long, trace_bw, trace_st, trace_long, text0, text1, text2, text3


    anim = animation.FuncAnimation(fig, animate, n, interval=100, blit=True) #interval is in [ms]
    if saveToFile:
        # saving to m4 using ffmpeg writer
        #writervideo = animation.FFMpegWriter(fps=30)
        #anim.save('anim1.mp4', writer=writervideo)
        writergif = animation.PillowWriter(fps=10) 
        anim.save("anim1.gif", writer=writergif)
        print("Animation saved to file!")
    else:
        plt.show()



def plotConning(motionRecord,motionPredict,predInterval,predSteps,width,length,xlim,ylim,saveToFile=False):
    #plots animated trajectory and conning display
    
    df = getStatesDataFrame(motionRecord)
    xvecMotion = np.asarray((df['x'],df['dx'],df['y'],df['dy'],df['c'],df['dc'])).T
    #latvel chart input
    velMag = np.linalg.norm(xvecMotion[:,(1,3)],axis=1)
    ori = np.unwrap(np.arctan2(xvecMotion[:,3],xvecMotion[:,1]))
    velAng = ori-xvecMotion[:,4]
    velLat = np.sin(velAng)*velMag 
    velLong = -1*np.cos(velAng)*velMag #NOTE: Change of direction here! Could cheat by abs()
    L = 5
    velLat_bow = velLat + xvecMotion[:,5]*L/2
    velLat_stern = velLat - xvecMotion[:,5]*L/2

    dfp = pd.DataFrame.from_dict(motionPredict)
    xvecMotionPredCollection = np.ndarray((len(dfp),predSteps,6)) #TODO: fix fixed
    for index, row in dfp.iterrows():
        xvecMotionPredCollection[index,...] = np.vstack(row['statePred'])

    animateTopView3(xvecMotion,xvecMotionPredCollection,predInterval,xlim,ylim,velLong,velLat_bow,velLat_stern,width=width,length=length,saveToFile=saveToFile)



#TODO: plot covariance circle of estimate
#TODO: plot obbMetaList to check contents