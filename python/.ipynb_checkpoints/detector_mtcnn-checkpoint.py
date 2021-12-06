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

    
    Args: 
        - frame: current frame of the video feed  
 
    Returns: 
        - bboxes: List of bounding boxes
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
            bboxes.append([x, y, w, h])
            
    return  bboxes