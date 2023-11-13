import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from collections import deque
import boxHandlingUtils

#MiS - Martin Sanfridson, January 2023

def animateTopView1(velLong,velLat_bow,velLat_stern):
    v_bw = -velLat_bow #change of coord system
    v_st = -velLat_stern
    v_long = velLong
    n = len(v_long)

    #position of parts of the plot
    y_bw = 1
    dy_bw = 0
    y_st = -1
    dy_st = 0
    x_sb = 1
    x_pt = 0
    x_long = (x_sb + x_pt)/2
    y_long = (y_bw + y_st)/2

    #TODO: need to scale input to make arrows of appropriate lengths
    #TODO: make an init-funtion to clean up
    #TODO: make functions in order to clean up

    fig = plt.figure(figsize=(5, 4)) 
    L = 1
    ax = fig.add_subplot(autoscale_on=False, xlim=(-L, 1 + L), ylim=(-2*L, 2*L))

    arrow_bw = mpatches.Arrow(x_sb, y_bw, v_bw[0], dy_bw,width=0.5)
    arrow_st = mpatches.Arrow(x_pt, y_st, v_st[0], dy_st,width=0.5)
    arrow_long = mpatches.Arrow(x_pt, y_st, v_st[0], dy_st,width=0.5)
    ax.add_patch(arrow_bw)
    ax.add_patch(arrow_st)
    ax.add_patch(arrow_long)
    trace_bw, = ax.plot([], [], 'ko', lw=1, ms=2)
    trace_st, = ax.plot([], [], 'ko', lw=1, ms=2)
    trace_long, = ax.plot([], [], 'ko', lw=1, ms=2)

    history_bw = deque(maxlen=n) #for making a trace!
    history_st = deque(maxlen=n) #for making a trace!
    history_long = deque(maxlen=n) #for making a trace!


    def animate(k):
        if k == 0:
            history_bw.clear()
            history_st.clear()
            history_long.clear()

        ax.patches = []
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

        arrow_long = mpatches.Arrow(x_long, y_long, 0, v_long[k],width=0.5,color=(1,1,0))
        #plot hull and arrows
        #[x_sb,x_sb,x_pt,x_pt,x_sb],[y_bw,y_st,y_st,y_bw,y_bw]
        ax.add_patch(mpatches.Rectangle((x_pt,y_st),x_sb-x_pt,y_bw-y_st,alpha=0.5))
        ax.add_patch(arrow_bw)
        ax.add_patch(arrow_st)
        ax.add_patch(arrow_long)
        if len(history_bw) > 1:
            trace_bw.set_data(history_bw[1],y_bw)
        else:
            trace_bw.set_data(history_bw[0],y_bw)
        if len(history_st) > 1:
            trace_st.set_data(history_st[1],y_st)
        else:
            trace_st.set_data(history_st[0],y_st)
        return arrow_bw, arrow_st, arrow_long, trace_bw, trace_st, trace_long

    ani = animation.FuncAnimation(fig, animate, n) #, n, interval=10, blit=True)

    plt.show()


def bboxFromStateVector(vecMotion,Wnom,Lnom):
    #convert from result vector to corner points
    #TODO: use varying L and W, add vecSize as input --> L AND W switched!?!
    bbox = np.zeros((len(vecMotion),5,2)) #4+1 of x and y
    for q in range(len(vecMotion)):
        bbox[q,0:4,:] = boxHandlingUtils.boxToCorners2D(Wnom,Lnom,[vecMotion[q,0],vecMotion[q,2]],vecMotion[q,4])
        bbox[q,4,:] = bbox[q,0,:]
    return bbox 


def animateTopView2(xvecMotion,xvecMotionPredCollection,xlim,ylim):

    n = len(xvecMotion)
    bboxes = bboxFromStateVector(xvecMotion,2.0,5.0)

    fig = plt.figure() #figsize=(5, 4)) 
    L = 40
    ax = fig.add_subplot(autoscale_on=False) #, xlim=(0*-L, L), ylim=(-L/2, L/2))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title("Animation of trace and prediction")
    ax.set_xlabel("[m]")
    ax.set_ylabel("[m]")
    history_x = deque(maxlen=n) #for making a trace!
    history_y = deque(maxlen=n) #for making a trace!
    trace, = ax.plot([], [], 'ko', lw=0.5, ms=2)
    curRect, = ax.plot([], [], 'b', lw=1, ms=2)
    pred, = ax.plot([], [], 'r:', lw=1, ms=2)
    predRect0, = ax.plot([], [], 'b:', lw=0.8, ms=2)
    predRect1, = ax.plot([], [], 'b:', lw=0.4, ms=2)
    
    def animate(k):
        if k == 0:
            history_x.clear()
            history_y.clear()
        history_x.appendleft(xvecMotion[k,0])
        history_y.appendleft(xvecMotion[k,2])
        #plot vessel and trace
        curRect.set_data(bboxes[k,:,0],bboxes[k,:,1])
        trace.set_data(history_x,history_y)
        #plot prediction
        vecMotionPred = xvecMotionPredCollection[k,...]
        pred.set_data(vecMotionPred[:,0],vecMotionPred[:,2])
        vecMotion = bboxFromStateVector(vecMotionPred,2.0,5.0)
        predRect0.set_data(vecMotion[len(vecMotion)//2,:,0],vecMotion[len(vecMotion)//2,:,1]) 
        predRect1.set_data(vecMotion[-1,:,0],vecMotion[-1,:,1])

        return trace, curRect, pred, predRect0, predRect1


    ani = animation.FuncAnimation(fig, animate, n, interval=100, blit=True) #interval is in [ms]
    #plt.axis('equal')
    plt.show()


#TODO: save animation to video file

def animateTopView3(xvecMotion,xvecMotionPredCollection,xlim,ylim,velLong,velLat_bow,velLat_stern):

    #--- setup figure with a set of axes
    gs_kw = dict(width_ratios=[5.0, 2.0], height_ratios=[1, 2])
    fig, axd = plt.subplot_mosaic([['left', 'upper right'],
                                ['left', 'lower right']],
                                figsize=(6.0, 3.0),gridspec_kw=gs_kw)
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
    bboxes = bboxFromStateVector(xvecMotion,2.0,5.0)

    L = 40
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
    y_long = (y_bw + y_st)/2
    v_lat_max = 2 #scaling factor for arrow
    v_long_max = 4
    

    #TODO: need to scale input to make arrows of appropriate lengths
    #TODO: make an init-funtion to clean up
    #TODO: make functions in order to clean up

    L = 1
    axd['upper right'].set_xlim((-2.0, 2.0))
    axd['upper right'].set_ylim((-2.0, 3.0))
    #axd['upper right'].set_facecolor('b')

    arrow_bw = mpatches.Arrow(x_sb, y_bw, v_bw[0]/v_long_max, dy_bw,width=0.5)
    arrow_st = mpatches.Arrow(x_pt, y_st, v_st[0]/v_lat_max, dy_st,width=0.5)
    arrow_long = mpatches.Arrow(x_pt, y_st, v_long[0]/v_long_max, dy_st,width=0.5)
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
        vecMotionPred = xvecMotionPredCollection[k,...]
        pred.set_data(vecMotionPred[:,0],vecMotionPred[:,2])
        vecMotion = bboxFromStateVector(vecMotionPred,2.0,5.0)
        predRect0.set_data(vecMotion[len(vecMotion)//2,:,0],vecMotion[len(vecMotion)//2,:,1]) 
        predRect1.set_data(vecMotion[-1,:,0],vecMotion[-1,:,1])

        #--- animate latvel
        if k == 0:
            history_bw.clear()
            history_st.clear()
            history_long.clear()

        axd['upper right'].patches = []
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

        arrow_long = mpatches.Arrow(x_long, y_long, 0, v_long[k],width=0.5,color=(0,0,0))
        #plot hull and arrows
        axd['upper right'].add_patch(mpatches.Rectangle((x_pt,y_st),x_sb-x_pt,y_bw-y_st,alpha=0.5,color=(0,0,1)))
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
    if False:
        # saving to m4 using ffmpeg writer
        #writervideo = animation.FFMpegWriter(fps=30)
        #anim.save('anim1.mp4', writer=writervideo)
        writergif = animation.PillowWriter(fps=10) 
        anim.save("anim1.gif", writer=writergif)
    else:
        plt.show()
