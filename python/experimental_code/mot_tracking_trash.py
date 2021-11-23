import cv2
import copy
from Detector_trash import Detectors
from tracker import Tracker
import imutils



def main():
    inputvid =  "../data/vid/cam1_5s.mp4" #video_ball.avi"  

    # Create opencv video capture object
    cap = cv2.VideoCapture(inputvid)

    # Create Object Detector
    detector = Detectors()

    # Create Object Tracker: dist_thresh, max_skipped_frames, max_trace_len, trackIdcount
    tracker = Tracker(160, 10, 5, 1) #self made tracker

    # Variables initialization
    skip_frame_count = 0
    track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]
    pause = False

    # Infinite loop to process video frames
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Make copy of original frame
        orig_frame = copy.copy(frame)
        frame = imutils.resize(frame, width = 800)

        # Detect and return centeroids of the objects in the frame
        centroids = detector.Detect_yolo(frame)
        #print(len(centroids))

        # If centroids are detected then track them
        if (len(centroids) > 0):

            # Track object using Kalman Filter
            tracker.Update(centroids)

            # For identified object tracks draw tracking line
            # Use various colors to indicate different track_id
            for i in range(len(tracker.tracks)):
                if (len(tracker.tracks[i].trace) > 1):
                    for j in range(len(tracker.tracks[i].trace)-1):
                        # Draw trace line
                        #print(len(tracker.tracks[i].trace))
                        x1 = tracker.tracks[i].trace[j][0][0]+0.5
                        y1 = tracker.tracks[i].trace[j][1][0]+0.5
                        #print(x1,y1)
                        x2 = tracker.tracks[i].trace[j+1][0][0]
                        y2 = tracker.tracks[i].trace[j+1][1][0]
                        #print(x2,y2)
                        clr = tracker.tracks[i].track_id % 9
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                 track_colors[clr], 2)

            # Display the resulting tracking frame
            cv2.imshow('Tracking', frame)
        
        # Slower the FPS
        #cv2.waitKey(20)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # execute main
    main()