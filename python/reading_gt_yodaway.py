# https://github.com/cheind/py-motmetrics
# reading ground truth files
import fileinput
import os
from os import read, write
import sys
import numpy as np
import pandas as pd
import motmetrics as mm
import csv
import pdb #python3 -m pdb code.py -> pdb.set_trace() as breakpoint
import cv2

bb_in_frame = []
folderpath = "../data/TestOnePlus1_data/"
files = os.listdir(folderpath)
ano_txt = "../data/img/tud_cross_gt.txt"
max_distance = 10000.  #maximal allowed distance between gt and hypothesis centroids
test = [[25.0, 74.0], [-28.5, -88.0], [-28.0, -82.0], [29.0, 83.0], [-29.0, -95.5], [29.5, -81.0], [-27.5, -73.5]]
test2 = [[364.0, 204.0], [427.0, 237.0], [140.0, 210.0], [166.0, 243.0], [201.0, 215.0], [237.0, 250.0]]
# def replaceAll(file,searchExp,replaceExp):
#     for line in fileinput.input(file, inplace=1):
#         if searchExp in line:
#             line = line.replace(searchExp,replaceExp)
#         sys.stdout.write(line)

# replaceAll(ano_txt)


def read_gt():
    obj_ids_frame=[]
    txtlines = []
    total_obj_ids = []
    obj_centroids_frame = []
    total_obj_centroids = []
    gt_data = []
    total_gt_data = []
    temp_obj_centr = []
    temp_obj_ids = []
    temp_data = []
    frame = 0
    for file in files:
        f = open(os.path.join(folderpath, file),'r')
        lines = f.readlines()
        for line in lines:
            if len(line.split(" ")) == 8:
                frameid,objectid,x,y,x2,y2,_,_= line.split(" ")
                objectstr = str(objectid)+" "+str(frame)+x+y+x2+y2+"\n"
                x = float(x) 
                x2 = float(x2)
                y = float(y)
                y2 = float(y2)
                w = abs(x-x2)
                h = abs(y-y2)
                cx = int(x + ((x+w)-(x))/2)
                cy = int((y+h) - ((y+h)-y)/2)
                centroid = [cx,cy]
                data = frame, objectid,centroid,x,y,w,h
                obj_centroids_frame.append(centroid)
                obj_ids_frame.append(objectid)
                txtlines.append(objectstr)
        frame +=1
        temp_obj_centr.append(obj_centroids_frame)
        temp_obj_ids.append(obj_ids_frame)
        temp_data.append(data)
        obj_centroids_frame = []
        obj_ids_frame = []
        data = []
    total_obj_ids.append(temp_obj_ids)
    total_obj_centroids.append(temp_obj_centr)
    total_gt_data.append(gt_data)

    return total_obj_centroids[0], total_obj_ids[0] , total_gt_data[0]

def write_new_gt():
    data = read_gt()
    with open(r'../yodaway_gt.txt', 'w') as outfile:

        frameid,_,x,y,x2,y2,_,_ = data.split(" ")
        data = frameid,x,y,x2,y2
        outfile.write(data)

#<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z> 
#The world coordinates x,y,z
def draw_bbox_gt(frame_vid, frame_cnt_vid):
    obj_centroids, object_ids, all_data = read_gt() 
    for data in range(len(all_data)):
        frame,objectid,centroid,x,y,w,h = data
        #print(frame, objectid,x,y,w,h)
        if (frame_cnt_vid == frame):
            x = int(x)
            y = int(y)
            h = round(float(h))
            w = round(float(w))
            p1 = (x,y)
            p2 = (x+w,y+h)
            cx = int(x + ((x+w)-(x))/2)
            cy = int((y+h) - ((y+h)-y)/2)
            print("objid: ", objectid, "centroid: ", centroid )
            cv2.putText(frame_vid, ".", (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10,10,10),4)
            cv2.rectangle(frame_vid, p1, p2, (13,10,10), 2)
        else: continue
        
# def dist_from_gt(frame_cnt_vid, hypothesis_c):
#     data = read_gt()
#     for i in data:
#         frame,objectid,centroid,_= data
#         if frame_cnt_vid == frame:    
#             o = np.array(centroid)
#             h = hypothesis_c
#             original_ids = objectid # check if the lengths of object_centroids and obj_ids are same?
#             print("Originals:",centroid)
#             print("Hypothesis:",hypothesis_c)
#             C = mm.distances.norm2squared_matrix(o,h)#, max_d2=max_distance)
#             return C, original_ids

def dist_from_gt(frame_cnt_vid, hypothesis_c):
    obj_centroids,object_ids,_ = read_gt() 
    for i in range(len(obj_centroids)):
        if (frame_cnt_vid == i):
            o = np.array(obj_centroids[i])
            h = hypothesis_c
            original_ids = object_ids[i] # check if the lengths of object_centroids and obj_ids are same?
            print("Originals:",obj_centroids[i])
            print("Hypothesis:",hypothesis_c)
            C = mm.distances.norm2squared_matrix(o,h)#, max_d2=max_distance)
            return C, original_ids

def rerewrite_new_gt():
    data = read_gt()
   # _,_,txtlines = coord_from_gt()
    with open(r'../data/yodaway_gt.txt', 'w') as outfile:
        for i in data:
            outfile.write(i)

#if __name__ == "__main__":
    #read_gt()
    #rerewrite_new_gt()
    #coord_from_gt()
    #dist_from_gt(test)
    #rerewrite_new_gt()
    #write_pd()
    #draw_bbox_gt(10, 100)

def write_pd():
    df = pd.read_csv('../data/img/new_tud_cross_gt3.txt', sep=" ", header=None, names=["Frame", "Object_id", "X", "X2", "Y", "Y2"])
    print(df)
    np.savetxt(r'../data/img/new_tud_cross_gt4.txt', df.values, fmt='%d')

# def numpy_euclidian_distance(point_1, point_2):
#     array_1, array_2 = np.array(point_1), np.array(point_2)
#     squared_distance = np.sum(np.square(array_1 - array_2))
#     distance = np.sqrt(squared_distance)
#     return distance