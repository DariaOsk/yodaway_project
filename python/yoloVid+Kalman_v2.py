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
from Kalmanfilter_remake import KalmanFilter
from tracker import Tracker
from numpy.lib.type_check import _nan_to_num_dispatcher
#from utils import *

#instead of argparse fixed values  --output output/wildrack_MOT.avi --yolo yolo-coco\venvYodaway\code\yolo>
inputvid =  "../data/vid/cam1_5s.mp4"
output_dir = "./output"
yolo_dir = "../yolov3"
confidence = 0.7 #default 0.5
threshold = 0.3 #default 0.3
#KF = KalmanFilter(0.005, 1, 1, 1, 0.1,0.1)
                #(0.1, 1, 1, 1, 0.1, 0.1)
tracker = Tracker(160,30,5,100)
skip_frame_count = 0
track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (0, 255, 255), (255, 0, 255), (255, 127, 255),
                (127, 0, 255), (127, 0, 127)]
pause = False
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
print("[INFO] load Yolo from disk..")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# dermine output layer names needed from YOLO
layerNames = net.getLayerNames()
layerNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]

#init of videostream 
vs = cv2.VideoCapture(inputvid)
writer = None
(H,W) = (None, None)

#count total nr of frames in video
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("INFO: {} total frames in video").format(total)
except:
    print("INFO: could not determine number of frames")
    print("INFO: no apporx complection time can be provided")
    toatal = -1

# frameloop
while True: 
    # read next frame from file
    (next, frame) = vs.read()

    if not next:
        break
    
    # check frame dimensions are provided
    if W is None or H is None:
        (H,W) = frame.shape[:2]

    # Make copy of original frame
    #orig_frame = copy.copy(frame)
    
    # blob for input frame and forward pass to YOLO object detector, with bounding boxes + probabilities
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(layerNames)
    end = time.time()

    # init of bboxes, confidences(=probabilities) and class IDs
    bboxes = []
    confs = []
    classIDs = []
    centroids = []

    #loop over each output layer 
    for output in layerOutputs:
        #loop over each detection
        for detection in output:
            #extract classID and confidence of current detection
            scores = detection[5:]
            classID = np.argmax(scores)
            conf = scores[classID]

            # filter weak predictions
            # probability greater than minimum probability
            if conf > confidence:
                #scale bbox coordinates back relative to img siye
                # Yolo returns center (x,y)-coords of bbox enhanced by box width and height
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
            #print(centroidArray[0], centroidArray[1])
            #different colours - not necessary if classID only 1 (person)
            #color = [int(c) for c in COLORS[classIDs[i]]]
            color = (0,200,200)
            # draw bbox rectange and label on img
            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)

            # draw bbox centroid on img
            cv2.circle(frame, (cx,cy), 1, (0,0,250),5)

            # if only person then not necessary:
            #txt = "{}: {:.4f}".format(LABELS[classIDs[i]],confs[i])
            txt = "person: {}".format(round(confs[i],2))
            cv2.putText(frame, txt, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
    #Applying Kalmanfilter to yolo detector    /  store centroids in dict  safe pred in blob object
    # if (len(centroids) > 0):
    #         for i in centroids:
            # Predict
            #(x_, y_) = KF.predict()
            
            # Draw a rectangle as the predicted object position
            #cv2.rectangle(frame, (int(x_-30), int(y_-50)), (int(x_+30), int(y_+50)), (255, 0, 0), 2)
            
            # Update Kalman Trajectory
            #(x1, y1) = KF.update([centroidArray[0],centroidArray[1]])#KF.update(centers[0])
            #print(centroidArray)
            # Draw a rectangle as the estimated object position
            #cv2.rectangle(frame, ( int(x1-30), int(y1-50)), (int(x1+30), int(y1+50)), (0, 0, 255), 2)
            #cv2.putText(frame, "Estimated Position", (int(x1+30), int(y1+50)), 0, 0.5, (0, 0, 255), 2)
            #cv2.putText(frame, "Predicted Position", (int(x_+30), int(y_)), 0, 0.5, (255, 0, 0), 2)
            #cv2.putText(frame, "Measured Position", (int(centroidArray[0]) + 15, int(centroidArray[1]) - 15), 0, 0.5, (0,191,255), 2)

    if (len(centroids) >0):
        tracker.Update(centroids)

        for i in range(len(tracker.tracks)):
            if (len(tracker.tracks[i].trace) > 1):
                for j in range(len(tracker.tracks[i].trace)-1):
                    # trace line draw
                    x11 = tracker.tracks[i].trace[j][0][0]
                    y11 = tracker.tracks[i].trace[j][1][0]
                    x22 = tracker.tracks[i].trace[j+1][0][0]
                    y22 = tracker.tracks[i].trace[j+1][1][0]
    
                    clr = tracker.tracks[i].track_id % 9
                    cv2.line(frame, (int(x11), int(y11)), (int(x22),int(y22)), track_colors[clr], 2)

    
    cv2.imshow('frame', frame)

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