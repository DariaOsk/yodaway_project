import numpy as np
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
import os
import tqdm
from multiprocessing import Process, Manager
#from tqdm import tqdm
#import seaborn as sns



# def get model_scores(pred_boxes):
   
#     #Creates a dictionary of from model_scores to image ids.
#     #Args: pred_boxes (dict): dict of dicts of 'boxes' and 'scores'
#     #Returns: dict: keys are model_scores and value are image ids (usually filenames)

#     model_score={}
#     for img_id, val in pred_boxer.items():
#         for score in val['scores']:
#             if score not in model_score.keys():
#                 model_score[score]=[img_id]
#             else:
#                 model_score[score].append(img_id)
#     return model_score

#manager = Manager()
#return_dict = manager.dict()

ano_txt = "./images/CrowdHuman/annotation_train.txt"
ano_csv = "./images/CrowdHuman/annotation_train.csv"
ano_odgt = "./images/CrowdHuman/annotation_train.odgt"

def read_ano(ano_path):
    with open(os.path.expanduser(ano_path)) as f:
        #num_lines = int(f.readline())
        mapping = []
        for k in enumerate(f.readline()):
            row = f.readline()
            for i in enumerate(row):
                myline = f.readline()
                print(myline['ID'])
    
def readmap (map):
    for i in len(map):
        print(i)

def ano2df(ano_path):
    df = pd.read_csv(ano_path, sep=',',delimiter=None)
    #df = pd.read_fwf(ano_path)
    df.head()

def reading_ano(annotation_path):
    
    with open(os.path.expanduser(annotation_path)) as f:
        lines = []
        max_iter = 500
        for i, line in tqdm(enumerate(f)):
            #return_dict[i] = line
            print(line)

#ano2df(ano_csv)
#ano2df(ano_txt)
#map = read_ano(ano_txt)
#readmap(map)
if __name__ == '__main__':
    #reading_ano(ano_odgt)
    read_ano(ano_odgt)