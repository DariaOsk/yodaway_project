import numpy as np
import cv2
import time
import imutils
import os
# set to 1 for pipeline images
debug = 0


class Detectors(object):
    """Detectors class to detect objects in video frame
    Attributes:
        None
    """
    def __init__(self):
        """Initialize variables used by Detectors class
        Args:
            None
        Return:
            None
        """
        self.fgbg = cv2.createBackgroundSubtractorMOG2()

    def Detect(self, frame):
        """Detect objects in video frame using following pipeline
            - Convert captured frame from BGR to GRAY
            - Perform Background Subtraction
            - Detect edges using Canny Edge Detection
              http://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html
            - Retain only edges within the threshold
            - Find contours
            - Find centroids for each valid contours
        Args:
            frame: single video frame
        Return:
            centers: vector of object centroids in a frame
        """

        # Convert BGR to GRAY
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if (debug == 1):
            cv2.imshow('gray', gray)

        # Perform Background Subtraction
        fgmask = self.fgbg.apply(gray)

        if (debug == 0):
            cv2.imshow('bgsub', fgmask)

        # Detect edges
        edges = cv2.Canny(fgmask, 50, 190, 3)

        if (debug == 1):
            cv2.imshow('Edges', edges)

        # Retain only edges within the threshold
        ret, thresh = cv2.threshold(edges, 127, 255, 0)

        # Find contours #killed hirarchy
        #_, contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if (debug == 0):
            cv2.imshow('thresh', thresh)

        centers = []  # vector of object centroids in a frame
        # we only care about centroids with size of bug in this example
        # recommended to be tuned based on expected object size for
        # improved performance
        blob_radius_thresh = 20#8
        # Find centroid for each valid contours
        for cnt in contours:
            print(cnt)
            try:
                # Calculate and draw circle
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                centeroid = (int(x), int(y))
                radius = int(radius)
                if (radius > blob_radius_thresh):
                    cv2.circle(frame, centeroid, radius, (0, 255, 0), 2)
                    b = np.array([[x], [y]])
                    centers.append(np.round(b))
            except ZeroDivisionError:
                pass

        # show contours of tracking objects
        # cv2.imshow('Track Bugs', frame)

        return centers





    def Detect_yolo(self, frame):
        #instead of argparse fixed values  --output output/wildrack_MOT.avi --yolo yolo-coco\venvYodaway\code\yolo>
        inputvid =  "../data/vid/cam1_5s.mp4"
        output_dir = "./output"
        yolo_dir = "../yolov3"
        confidence = 0.7 #default 0.5
        threshold = 0.3 #default 0.3
        #KF = KalmanFilter(0.005, 1, 1, 1, 0.1,0.1)
                        #(0.1, 1, 1, 1, 0.1, 0.1)
        
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

        # check frame dimensions are provided
        (H,W) = (None, None)
        
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
                
        return centroids
