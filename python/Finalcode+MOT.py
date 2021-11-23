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
import pdb
import motmetrics as mm
import time
from Kalmanfilter_remake import KalmanFilter
from tracker import Tracker
from numpy.lib.type_check import _nan_to_num_dispatcher
from precision_calc import calc_dist_points
from reading_gt import dist_from_gt,draw_bbox_gt
#from utils import *

#instead of argparse fixed values  --output output/wildrack_MOT.avi --yolo yolo-coco\venvYodaway\code\yolo>
inputvid =  "../data/vid/annotated_seq1.mp4" #cam1_5s.mp4"
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

#paths to YOLO weights and model config
weightsPath = os.path.sep.join([yolo_dir, "yolov3.weights"])
configPath = os.path.sep.join([yolo_dir, "yolov3.cfg"])

# load YOLO detector trained on COCO dataset
print("Loading YoloV3")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# dermine output layer names needed from YOLO
layerNames = net.getLayerNames()
layerNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]

#init of videostream 
vs = cv2.VideoCapture(inputvid)
writer = None
(H,W) = (None, None)
frame_cnt = 0

# frameloop
while True: 
    # read next frame from file
    (next, frame) = vs.read()

    if not next:
        break
    
    # check frame dimensions are provided
    if W is None or H is None:
        (H,W) = frame.shape[:2]

    # blob for input frame and forward pass to YOLO object detector, with bounding boxes + probabilities
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(layerNames)
    end = time.time()

    # init of bboxes, confidences(=probabilities) and class IDs
    bboxes = []
    confs = []
    classIDs = [] #real classIDs
    centroids = [] #real class 
    hypothesis_xy = [] #[[x1,y1],[x2,y2]] # distances from object classID to detected objects
    hypothesis_ids = [] 
    
    #loop over each output layer 
    for output in layerOutputs:
        #loop over each detection
        for detection in output:

            #extract classID and confidence of current detection
            scores = detection[5:]
            classID = np.argmax(scores)

            conf = scores[classID]

            
            # - there apparently is no way to exclude the detection itself
            # which could make the detector obviously slow and unfit for real time (only) human detection
            # Either yolo needs to be retrained only on humans or another detector will be used.
            # exclude all the other classes so only person is shown 
            if (classID != 0):
                conf = 0

            #print(classID)
            # filter weak predictions
            # probability greater than minimum probability
            if conf > confidence:
                #scale bbox coordinates back relative to img siye
                #Yolo returns center (x,y)-coords of bbox enhanced by box width and height
                box = detection[0:4] * np.array([W,H,W,H])
                (centerX, centerY, width, height) = box.astype("int")

                #use center coordinates to derive top and left box corner
                x = int(centerX-(width/2))
                y = int(centerY-(height/2))

                #update our list of bboxes, confidences, class IDs
                bboxes.append([x,y,int(width), int(height)])
                confs.append(float(conf))
                classIDs.append(classID)
    
    # non-maxima suppression=filtering out unnecessary boxes
    idxs = cv2.dnn.NMSBoxes(bboxes, confs, confidence, threshold)
    #print(len(idxs))
    # ensure that detection exists
    if len(idxs) > 0:        
        # loop over indexes 
        for i in idxs.flatten():
            # extract bbox coordinates
            (x,y) = (bboxes[i][0], bboxes[i][1])
            (w,h) = (bboxes[i][2], bboxes[i][3])

            # extract centroid positions
            cx = int(x + ((x+w)-(x))/2)
            cy = int((y+h) - ((y+h)-y)/2)
            #print(cx,cy)
            # include box centroids into array that will be passed to KalmanFilter
            centroids.append(np.round(np.array([[cx], [cy]])))
            centroidArray = np.array([[cx],[cy]])

            #different colours - not necessary if classID only 1 (person)
            color = [int(c) for c in COLORS[classIDs[i]]]
            color = (0,200,200)
            # draw bbox rectange and label on img
            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)

            # draw bbox centroid on img
            cv2.circle(frame, (cx,cy), 1, (0,0,250),5)

            # if only person then not necessary:
            txt = "{}: {:.4f}".format(LABELS[classIDs[i]],confs[i])
            txt = "person: {}".format(round(confs[i],2))
            cv2.putText(frame, txt, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    temp_id = []
    temp_centr2 = []
    temp_centr = [] 
    centr_frame = 0       
    if (len(centroids) > 0):
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
                    #define the color 
                    clr = tracker.tracks[i].track_id % 15 
                    #cv2.line(frame, (int(x11), int(y11)), (int(x22),int(y22)), track_colors[clr], 4)
                    #cv2.rectangle(frame, (int(x22)-50, int(y22)-70), (int(x22)+50, int(y22)+70), track_colors[clr], 1)                    
                    #cv2.putText(frame, ".", (int(x11), int(y11)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 6)

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
    hypothesis_xy = hypothesis_xy#[0]
        #pplcount = str(i)
        #cv2.putText(frame, pplcount, (int(x22), int(y22)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
    cv2.putText(frame, "Enter: Green", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1) #B
    cv2.putText(frame, "Exit: Yellow", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,210),1) #G

    #dist_h_gt = dist_from_gt(hypothesis_xy)
    print("Hypothesis_ids",hypothesis_ids)
    print("Hypothesis_xy",hypothesis_xy)
    draw_bbox_gt(frame,frame_cnt)
    distances, o_ids = dist_from_gt(frame_cnt, hypothesis_xy)
    print("distances: ", distances)
    print("original id:", o_ids)
    #acc.update(o_ids, hypothesis_ids, distances)

    cv2.imshow('frame', frame) 
    frame_cnt +=1
    #check writer
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