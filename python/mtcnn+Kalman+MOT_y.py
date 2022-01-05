#tutorial: https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/#download-the-code
#yolo keras implementation step-by-step https://github.com/experiencor/keras-yolo2/blob/master/Yolo%20Step-by-Step.ipynb
# weights retrieved from darknet site: https://pjreddie.com/darknet/yolo/#demo
# config + label file retrieved from official yolo github: https://github.com/pjreddie/darknet
# https://github.com/nilesh0109/PedestrianTracking/blob/master/main.py
# cmd: wget directory/to/raw/file -O saveasfilename
import cv2
import numpy as np
import argparse
import os
import imutils
import time
from tracker import Tracker
from detector_mtcnn import get_mtcnn_face
import motmetrics as mm
from reading_gt_yodaway import dist_from_gt,draw_bbox_gt
#from utils import *

#instead of argparse fixed values  --output output/wildrack_MOT.avi --yolo yolo-coco\venvYodaway\code\yolo>
inputvid =  "../data/vid/TestOnePlus1.mp4" #cam1_5s.mp4"  annotated_seq1
output_dir = "./output"
yolo_dir = "../yolov3"
confidence = 0.7 #default 0.5
threshold = 0.3 #default 0.3
tracker = Tracker(160,30,5,0)
skip_frame_count = 0
track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (0, 255, 255), (255, 0, 255), (255, 127, 255),
                (127, 0, 255), (127, 0, 127),(100, 100, 100),
                (160, 135, 10), (12, 20, 0), (100, 30, 0),
                (100, 200, 355),(134, 20, 0), (0, 0, 1)]
            # -> write something like : len(detections) = track_colors > [fill random(0,255)] but only for first frame 
pause = False
person_id = 14


#create accumulator that will be updated during each frame
acc = mm.MOTAccumulator(auto_id=True)

#load coco class labels 
labelsPath = os.path.sep.join([yolo_dir, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")


#initilize colors for labels
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")


#init of videostream 
vs = cv2.VideoCapture(inputvid)
writer = None
(H,W) = (None, None)
frame_cnt = 0

# #count total nr of frames in video
# try:
#     prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
#         else cv2.CAP_PROP_FRAME_COUNT
#     total = int(vs.get(prop))
#     print("INFO: {} total frames in video").format(total)
# except:
#     print("INFO: could not determine number of frames")
#     print("INFO: no apporx complection time can be provided")
#     toatal = -1

# frameloop
while True: 
    # read next frame from file
    (next, frame) = vs.read()

    if not next:
        break
    
    # check frame dimensions are provided
    if W is None or H is None:
        (H,W) = frame.shape[:2]

    # init of bboxes, ..
    end = time.time()
    centroids = [] 
    hypothesis_xy = [] #[[x1,y1],[x2,y2]] # distances from object classID to detected objects
    hypothesis_ids = [] 

    # Extract mtcnn bounding boxes and centroids 
    bboxes = get_mtcnn_face(frame)
    print("bboxes", len(bboxes))
    if len(bboxes) > 0:        
        # loop over indexes 
        
        for box in bboxes: 
            # extract bbox coordinates
            (x, y, w, h) = [int(v) for v in box]
            # extract centroid positions
            cx = int(x + ((x+w)-(x))/2)
            cy = int((y+h) - ((y+h)-y)/2)
            #print(cx,cy)
            # include box centroids into array that will be passed to KalmanFilter
            centroids.append(np.round(np.array([[cx], [cy]])))
            centroidArray = np.array([[cx],[cy]])
            

            #different colours - not necessary if classID only 1 (person)
            #color = [int(c) for c in COLORS[classIDs[i]]]
            color = (0,200,200)
            # draw bbox rectange and label on img
            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)

            # draw bbox centroid on img
            cv2.circle(frame, (cx,cy), 1, (0,0,250),5)

            # if only person then not necessary:
            #txt = "{}: {:.4f}".format(LABELS[classIDs[i]],confs[i])
            #txt = "person: {}".format(round(confs[i],2))
            txt = "person"
            cv2.putText(frame, txt, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    temp_id = []
    temp_centr2 = []
    temp_centr = [] 
    centr_frame = 0      
    if (len(centroids) >0):
        tracker.Update(centroids)
        for i in range(len(tracker.tracks)):
            if (len(tracker.tracks[i].trace) > 1):
                for j in range(len(tracker.tracks[i].trace)-1):
                    # trace line draw
                    #11 new updates
                    x11 = tracker.tracks[i].trace[j][0][0]
                    y11 = tracker.tracks[i].trace[j][1][0]
                    #22 old tracker points
                    x22 = tracker.tracks[i].trace[j+1][0][0]
                    y22 = tracker.tracks[i].trace[j+1][1][0]
    
                    clr = tracker.tracks[i].track_id % 15
                    #cv2.line(frame, (int(x11), int(y11)), (int(x22),int(y22)), track_colors[clr], 4)
                    
                    #cv2.rectangle(frame, (int(x22)-50, int(y22)-70), (int(x22)+50, int(y22)+70), track_colors[clr], 1)
                    pplcount = str(i)
                    #cv2.putText(frame, pplcount, (int(x22), int(y22)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.putText(frame, ".", (int(x11), int(y11)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 6)

                    if (x11>x22):
                        cv2.putText(frame, ".", (int(x22), int(y22)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,210), 6)
                    if (x22>x11):
                        cv2.putText(frame, ".", (int(x22), int(y22)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 6) 
                    
                    centr_frame = [x22,y22]
                    temp_centr.append(centr_frame)
                    
                temp_centr2.append(temp_centr[0])
                temp_id.append(tracker.tracks[i].track_id)
                temp_centr = []

    hypothesis_ids.append(temp_id)
    hypothesis_xy = temp_centr2
    hypothesis_ids = hypothesis_ids[0]
    
    cv2.putText(frame, "Enter: Green", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1) #B
    cv2.putText(frame, "Exit: Yellow", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,210),1) #G
    
    #visualization of the ground truth
    draw_bbox_gt(frame,frame_cnt)

    # application of the MOT metrics by calculating the distances btw hypothesis and gt
    distances, gt_ids = dist_from_gt(frame_cnt, hypothesis_xy)
    # populating accumulator for later evaluation
    frameid = acc.update(gt_ids, hypothesis_ids, distances)
    #print(acc.mot_events)

    cv2.imshow('frame', frame)

    
    frame_cnt +=1

    if frame_cnt == 999:
        mh = mm.metrics.create()
       
        summary = mh.compute_many(  [acc, acc.events.loc[0:1]],
                                    metrics=mm.metrics.motchallenge_metrics,
                                    names=['full', 'part'])

        strsummary = mm.io.render_summary(  summary,
                                            formatters=mh.formatters,
                                            namemap=mm.io.motchallenge_metric_names)
        print(strsummary)
        break
    

    # #check writer
    # if writer is None:
    #     #init video writer
    #     fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    #     writer = cv2.VideoWriter(output_dir, fourcc, 30, (frame.shape[1], frame.shape[0]), True)

    #     #single frame processing
    #     if total>0:
    #         elap = (end-start)
    #         print("INFO: single frame took {:.4f} seconds".format(elap))
    #         print("INFO: estimated toatal time to finish {:.4f}".format(elap*total))

    # writer.write(frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("INFO: clean up...")
#writer.release()
vs.release()

#need to return centers