"""
I use MTCNN (the default face detector used by facenet). I got it just by running pip install MTCNN.
To implement it using c++ I've seen repositories that may help...
https://github.com/happynear/MTCNN_face_detection_alignment
https://github.com/OAID/FaceDetection
"""
from mtcnn import MTCNN

def get_mtcnn_face(frame):
    """
    Runs detector on current fame and returns a list of bounding boxes (also list), each bounding box has four numbers. 
    x = x coordinate of the top left point of the bounding box
    y = y coordinate of the top left point of the bounding box
    w = width of the bounding box 
    h = height of the bounding box 
    
    Args: 
        - frame: current frame of the video feed  
 
    Returns: 
        - bboxes: List of bounding boxes
    """
    mtcnn = MTCNN()
    detections=mtcnn.detect_faces(frame)
    if detections == []:
        bboxes = []
    else: 
        bboxes = []
        for detection in detections:
            x,y,w,h=detection['box']
            #if 160<w<800 and 160<h<800: 
                #bboxes.append([x, y, w, h])
            bboxes.append([x, y, w, h])
    return  bboxes