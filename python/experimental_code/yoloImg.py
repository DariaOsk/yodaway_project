#tutorial: https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/#download-the-code
#yolo keras implementation step-by-step https://github.com/experiencor/keras-yolo2/blob/master/Yolo%20Step-by-Step.ipynb
# weights retrieved from darknet site: https://pjreddie.com/darknet/yolo/#demo
# config + label file retrieved from official yolo github: https://github.com/pjreddie/darknet
# cmd: wget directory/to/raw/file -O saveasfilename
import cv2
import numpy as np
import argparse
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-y", "--yolo", required=True, help=("base path to YOLO directory"))
ap.add_argument("-c", "--confidence", type=float, default=0.5, help=("min probability - detection filter"))
#non-maxima suppression = ignores redundant and overlapping bounding boxes
ap.add_argument("-t", "--threshold", type=float, default=0.3, help=("threshold for non-maxima suppression")) 
args = vars(ap.parse_args())

#load coco class labels 
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

#initilize colors for labels
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

#paths to YOLO weights and model config
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load YOLO detector trained on COCO dataset
print("[INFO] load Yolo from disk..")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

#load input img
img = cv2.imread(args["image"])
(H,W) = img.shape[:2] 

# dermine output layer names needed from YOLO
layerNames = net.getLayerNames()
layerNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]

# construct blob from input img and perfom forward pass of yolo detector
# -> gives us bounding boxes and associated probabilities
# blobFromImage - preprocessing for better classification - mean subtraction (against illumination changes) + scaling + channel swapping
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416,416), swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(layerNames) #pass forward of sought layer names
end = time.time()

#inference time info for YOLO = from start of running live data points into an ML model until the first output
print("INFO: Yolo took {:.6f} seconds".format(end-start))

# init of bboxes, confidences(=probabilities) and class IDs
bboxes = []
confs = []
classIDs = []

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
        if conf > args["confidence"]:
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

# non-maxima suppression
idxs = cv2.dnn.NMSBoxes(bboxes, confs, args["confidence"],args["threshold"])

# ensure that detection exists
if len(idxs) > 0:
    # loop over indexes 
    for i in idxs.flatten():
        # extract bbox coordinates
        (x,y) = (bboxes[i][0], bboxes[i][1])
        (w,h) = (bboxes[i][2], bboxes[i][3])

        # draw bbox rectange and label on img
        color = [int(c) for c in COLORS[classIDs[i]]]

        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        txt = "{}: {:.4f}".format(LABELS[classIDs[i]],confs[i])
        cv2.putText(img, txt, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# show 
cv2.imshow("Image", img)
cv2.waitKey(0)