# run !pip install MTCNN  to get the library 

from mtcnn import MTCNN

def get_mtcnn_face(frame):
    """
    Runs detector on current fame and returns a list of bounding boxes (also list), each bounding box has four numbers. 
    It also returns a list of centroids per bounding box. 
    x = x coordinate of the top left point of the bounding box
    y = y coordinate of the top left point of the bounding box
    w = width of the bounding box 
    h = height of the bounding box 
    cx = x coordinate of centroid 
    cy = y coordinate of centroid 
    
    Args: 
        - frame: current frame of the video feed  
 
    Returns: 
        - bboxes: List of bounding boxes
        - centroids: List of centroids 
    """
    mtcnn = MTCNN()
    detections=mtcnn.detect_faces(frame)
    if detections == []:
        bboxes = []
        centroids =[]
    else: 
        bboxes = []
        centroids =[]
        for detection in detections:
            x,y,w,h=detection['box']
            # calculate centroids
            # this formula is from yoloVid+Kalman_v2.py lines 130-132
            cx = int((x+w)/2)
            cy = int((y+h)/2)
            #cx = int(x + ((x+w)-(x))/2)
            #cy = int((y+h) - ((y+h)-y)/2)
            bboxes.append([x, y, w, h])
            #centroids.append(([[cx], [cy]]))
            
    return  bboxes#, centroids