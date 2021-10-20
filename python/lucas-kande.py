#https://learnopencv.com/optical-flow-in-opencv/

import cv2
import numpy as np
import time
videopath = "../../research/cam4.mp4"
time_elapsed = 0

def main():
    lk(videopath)

def lk(videopath):

    vid = cv2.VideoCapture(videopath)
    time_elapsed = 5
    #ShiTomasi corner detection - most used method to implement in optical flow
    f_params = dict( 
                    maxCorners=100, #100
                    qualityLevel=0.3, #0.3
                    minDistance=5, 
                    blockSize=7
                    )
    #parameters for canny corner detection 
    c_params = dict(
                    threshold1=50,
                    threshold2=190,
                    edges=3,
                    )

    #Lucas Kanade optical flow
    lk_params = dict(
                    winSize = (7,7), #winsize
                    maxLevel=2,
                    criteria = (cv2.TERM_CRITERIA_EPS | 
                                cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                     )
    color = np.random.randint(0, 100,(100,3))

    #corner detection from first frame
    ret, frame_o = vid.read()
    #gray_o = cv2.cvtColor(frame_o, cv2.COLOR_BGR2GRAY)
    gray_o = cv2.cvtColor(frame_o, cv2.COLOR_BGR2HSV)
    # Edge detection using Canny function
    gray_o = cv2.Canny(gray_o, **c_params)
    p0 = cv2.goodFeaturesToTrack(gray_o, mask= None, **f_params)
    
    #mask for drawing
    mask = np.zeros_like(frame_o)

    while True:
        ret,frame = vid.read()

        if ret:
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.putText(frame, 'Enter: green', (50,50), cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 1, cv2.LINE_AA)
            frame = cv2.putText(frame, 'Exit: red', (50,90), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 1, cv2.LINE_AA)
                
            #count time elapsed - if period longer than 10s renew edges
            t2 = time.perf_counter()

            # iniciate calculate optical flow
            # calcOpticalFlowPyrLK(prev_img, next_img, prev_pts, next_pts[, winSize[,maxLevel[,citeria]]])
            # returns: 
            # vector of next points (p1), 
            # output status vector (st) (sets each element to 1 if flow for feature found, sonst 0), 
            # output errors vector (err)
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                            gray_o, 
                            gray, 
                            p0, 
                            None, 
                            **lk_params
                        )

            #select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            #draw tracks i
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                (a,b) = new.ravel()
                (c,d) = old.ravel() #ravel returns flattened vector of 1D
                mask = cv2.line(mask,
                                (int(a),int(b)), #need to cast into int()
                                (int(c),int(d)),
                                color[i].tolist(), 2)

                #x direction
                #when exiting (r->l a<c)
                if a < c:
                    mask = cv2.putText(mask, "X", (int(a),int(b)), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 1, cv2.LINE_AA)
                else:
                    mask = cv2.putText(mask, "O", (int(a),int(b)), cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 1, cv2.LINE_AA)
                
                frame = cv2.circle(frame,
                                (int(a),int(b)),
                                2,
                                color[i].tolist(), -1)
                
            #display
            img = cv2.add(frame,mask)
            cv2.imshow("LUCAS-KANADE", img)
            
            #Update prev frame and points
            gray_o = gray.copy()
            p0 = good_new.reshape(-1,1,2)

            #repeat finding of good features
            if time_elapsed > 15: #> 5 and time_elapsed < 5.3:

                p0 = cv2.goodFeaturesToTrack(gray_o, mask= None, **f_params)
                time_elapsed = 0
                print("now")
                continue
            
            k = cv2.waitKey(1) 
            if k == ord('q'):
                break    
            
            time_elapsed += 0.1 #time_elapsed #t2-t
            print(time_elapsed)

        else:
            break
    
    vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()