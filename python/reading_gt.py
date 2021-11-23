# https://github.com/cheind/py-motmetrics
# reading ground truth files
import fileinput
from os import write
import sys
import numpy as np
import pandas as pd
import motmetrics as mm
import csv
import pdb #python3 -m pdb code.py -> pdb.set_trace() as breakpoint
import cv2

bb_in_frame = []

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
    with open(r'../data/img/new_tud_cross_gt2.txt','r') as infile:
        data = infile.read()
    return data

def write_new_gt():
    data = read_gt()
    with open(r'../data/img/new_tud_cross_gt.txt', 'w') as outfile:
        data = data.replace("-1","")
        data = data.replace("(","")
        data = data.replace(")","")
        data = data.replace("DaSide0811-seq7-","")
        data = data.replace("DaSide0811-seq","")
        data = data.replace(".png","")
        data = data.replace(":,",":")
        data = data.replace(":;",":")
        id = data[17:20]
        outfile.write(data)

def coord_from_gt():
    with open(r'../data/img/new_tud_cross_gt.txt','r') as file:

        row = file.readline()
        obj_ids_frame=[]
        txtlines = []
        total_obj_ids = []
        obj_centroids_frame = []
        total_obj_centroids = []
        temp_obj_centr = []
        temp_obj_ids = []
        for k in enumerate(row):
            myline = file.readline()
            frameid = myline[1:4]
            splitLine = myline.split(sep=":")
            o=0
            for i in splitLine:
                if len(i)>5:
                    x,y,x2,y2 = i.split(",")
                    objectstr = str(frameid)+" "+str(o)+x+y+x2+y2+"\n"
                    x = int(x) 
                    x2 = int(x2)
                    y = int(y)
                    y2 = int(y2)
                    w = abs(x-x2)
                    h = abs(y-y2)
                    cx = int(x + ((x+w)-(x))/2)
                    cy = int((y+h) - ((y+h)-y)/2)
                    centroid = [cx,cy]
                    obj_centroids_frame.append(centroid)
                    obj_ids_frame.append(o)
                    o+=1
                    txtlines.append(objectstr)
            temp_obj_centr.append(obj_centroids_frame)
            temp_obj_ids.append(obj_ids_frame)
            obj_centroids_frame = []
            obj_ids_frame = []
        total_obj_ids.append(temp_obj_ids)
        total_obj_centroids.append(temp_obj_centr)
            
        #pdb.set_trace()
            #obj_ids_frame = []
            #obj_centroids = []
            #print(total_obj_ids)
        #with open(r'../data/img/new_tud_cross_gt3.txt', 'w') as outfile:
            #for i in txtlines:
        #    outfile.writelines(txtlines)#"%s\n" % t for t in txtlines)

    return total_obj_centroids[0], total_obj_ids[0], txtlines

#<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z> 
#The world coordinates x,y,z
def draw_bbox_gt(frame_vid, frame_cnt_vid):
    with open(r'../data/img/det.txt','r') as file:
        row = file.readline()
        #print("rowlength: {}",len(row))
        for k in enumerate(row):
            myline = file.readline()
            frame,obj_id,x,y,w,h,conf,x_,y_,z_ = myline.split(",") #frame,obj_id,x,y,x2,y2 = myline.split(",")
            frame_cnt_vid = str(frame_cnt_vid)
            #frame = frame.zfill(3)
            #frame_cnt = frame_cnt_vid.zfill(3)#rjust(3, '0')
            #print(frame_cnt_vid,frame)
            if frame_cnt_vid == frame:
                x = int(x)
                y = int(y)
                h = round(float(h))
                w = round(float(w))
                #x2 = int(x2)
                #y2 = int(y2)
                p1 = (x,y)
                p2 = (x+w,y+h)
                #w = abs(x-x2)
                #h = abs(y-y2)
                cx = int(x + ((x+w)-(x))/2)
                cy = int((y+h) - ((y+h)-y)/2)
                cv2.putText(frame_vid, ".", (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10,10,10),4)
                cv2.rectangle(frame_vid, p1, p2, (13,10,10), 2)
                #return frame,obj_id,p1,p2
            else: continue
        
# def draw_bbox_gt2():
#     with open(r'../data/img/new_tud_cross_gt.txt','r') as file:

#         row = file.readline()
#         for k in enumerate(row):
#             print("len: {}",len(row))
#             myline = file.readline()
#             print(myline)


def dist_from_gt(frame_cnt_vid, hypothesis_c):
    lst_dist = []
    obj_centroids,object_ids,_ = coord_from_gt() 
    for i in range(len(obj_centroids)):
        if (frame_cnt_vid == i):
            o = np.array(obj_centroids[i])
            h = hypothesis_c
            original_ids = object_ids[i] # check if the lengths of object_centroids and obj_ids are same?
            print("Originals:",obj_centroids[i])
            print("Hypothesis:",hypothesis_c)
            #print(object_ids[i])

    #max_d2 : float
    #    Maximum tolerable squared Euclidean distance. Object / hypothesis points
    #    with larger distance are set to np.nan signalling do-not-pair. Defaults
    #    to +inf
            C = mm.distances.norm2squared_matrix(o,h)#, max_d2=max_distance)
            return C, original_ids
    #for i in range(len(object_ids)):
    #    if (i == frame_cnt_vid):
    #        for id in object_ids:
    #            print("obj_centroid", obj_centroids[i], "hypothesis_centroid",hypothesis_c)
    #            o = np.array(obj_centroids[i])

    #        for centr in hypothesis_c:
    #            h = np.array(centr)
    #            C = mm.distances.norm2squared_matrix(o,h, max_d2=max_distance)
    #            lst_dist.append(C)

        #return lst_dist #, id
   
def rerewrite_new_gt():
    #data = read_gt()
    _,_,txtlines = coord_from_gt()
    with open(r'../data/img/new_tud_cross_gt3.txt', 'w') as outfile:
        for i in txtlines:
            outfile.write(i)

#if __name__ == "__main__":
    #coord_from_gt()
    #dist_from_gt(test)
    #rerewrite_new_gt()
    #write_pd()
    #draw_bbox_gt2()

#def write_pd():
#     df = pd.read_csv('../data/img/new_tud_cross_gt3.txt', sep=" ", header=None, names=["Frame", "Object_id", "X", "X2", "Y", "Y2"])
#     print(df)
#     np.savetxt(r'../data/img/new_tud_cross_gt4.txt', df.values, fmt='%d')

# def numpy_euclidian_distance(point_1, point_2):
#     array_1, array_2 = np.array(point_1), np.array(point_2)
#     squared_distance = np.sum(np.square(array_1 - array_2))
#     distance = np.sqrt(squared_distance)
#     return distance