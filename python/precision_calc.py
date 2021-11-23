
# MOT metrics taken from https://github.com/cheind/py-motmetrics
import motmetrics as mm
import numpy as np

#Calculate distances between groundtruth and hypothesis
def calc_dist_points(groundtruth, hypothesis ):
    o = np.array( groundtruth)  #[[x1,y1],[x2,y2]])    #ground truth
    h = np.array( hypothesis )  #[[xh1,yh1][xh2,yh2]])  
    # pass an array of arrays of points

    C = mm.distances.norm2squared_matrix(o, h, max_d2=5.)

#Create accumulator that will be updated during each frame
def init_acc():
    acc = mm.MOTAccumulator(auto_id=True)

    # Call update once for per frame. For now, assume distances between
    # frame objects / hypotheses are given.
    acc.update(
        [1, 2],                     # Object IDs of Ground truth objects in this frame
        [1, 2, 3],                  # Object IDs of Detector hypotheses in this frame
        [
            [0.1, np.nan, 0.3],     # Distances from object 1 to hypotheses 1, 2, 3
            [0.5,  0.2,   0.3]      # Distances from object 2 to hypotheses 1, 2, 3
        ]
    )

    print(acc.events)
# if __name__ == '__main__':
#     mh = mm.metrics.create()
#     print(mh.list_metrics_markdown())

