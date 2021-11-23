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
import copy
import imutils
import time
from Kalmanfilter import Kalmanfilter
from Detector_trash import Detectors

from numpy.lib.type_check import _nan_to_num_dispatcher

#instead of argparse fixed values  --output output/wildrack_MOT.avi --yolo yolo-coco\venvYodaway\code\yolo>
inputvid =  "../data/vid/cam1_5s.mp4"#video_ball.avi"
output_dir = "./output"
yolo_dir = "../yolov3"
confidence = 0.5 #default 0.5
threshold = 0.3 #default 0.3
KF = Kalmanfilter(0.1, 1, 1, 1, 0.1,0.1)
detector = Detectors()


#init of videostream 
vs = cv2.VideoCapture(inputvid)

#include Masking for Drawings
frame_o = vs.read()

# frameloop
while True: 
    # read next frame from file
    (next, frame) = vs.read()
    # Make copy of original frame
    orig_frame = copy.copy(frame)
    
    if not next:
        break

    frame = imutils.resize(frame, width = 800)
    centroids = detector.Detect_yolo(frame)
    mask = np.zeros_like(frame)
    #mask = cv2.putText(frame, "X", (int(400),int(400)), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 1, cv2.LINE_AA)
    #Applying Kalmanfilter to yolo detector    /  store centroids in dict  safe prediction in blob object

    if (len(centroids) > 0):
        for i in range(len(centroids)):
            # Predict
            (x_, y_) = KF.predict()
            # Draw a rectangle as the predicted object position
            cv2.rectangle(frame, (int(x_-30), int(y_-50)), (int(x_+30), int(y_+50)), (255, 0, 0), 2)
            
            # Update Kalman Trajectory
            (x1, y1) = KF.update(centroids[0])

            # Draw a rectangle as the estimated object position
            #cv2.rectangle(frame, ( int(x1-30), int(y1-50)), (int(x1+30), int(y1+50)), (0, 0, 255), 2)
            #cv2.putText(frame, "Estimated Position", (int(x1+30), int(y1+50)), 0, 0.5, (0, 0, 255), 2)
            #cv2.putText(frame, "Predicted Position", (int(x_+30), int(y_)), 0, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, "Measured Position", (int(centroids[0][0]) + 15, int(centroids[0][1]) - 15), 0, 0.5, (0,191,255), 2)
            mask = cv2.line(mask,
                            (int(centroids[0][0]), int(centroids[0][1])), 
                            (int(x1), int(y1)),
                            (55,0,255), 2)

    #display
    img = cv2.add(frame,mask)
    cv2.imshow('frame', img)

    #frame_o = frame.copy()

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