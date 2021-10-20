import cv2
import numpy as np


face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
file_name = "kalman.txt"
videopath = "../../research/cam4.mp4"

def main():
    kalman_filter_tracker()
def detect_one_face(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    if len(faces) == 0:
        return (0, 0, 0, 0)
    return faces[0]

def kalman_filter_tracker():
    # Open output file
    # output_name = "../../code/" + file_name #sys.argv[3] + file_name
    # output = open(output_name, "w")
    frameCounter = 0
    v = cv2.VideoCapture(videopath)

    # read first frame
    ret, frame = v.read()
    if not ret:
        return

    # detect face in first frame
    c, r, w, h = detect_one_face(frame)

    # Write track point for first frame
    pt = (0,   c + w/2.0, r + h/2.0)
    #output.write("%d,%d,%d\n" % pt)  # Write as 0,pt_x,pt_y
    frameCounter += 1

    kalman = cv2.KalmanFilter(4, 2, 0)

    state = np.array([c + w / 2, r + h / 2, 0, 0], dtype='float64')  # initial position
    kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                        [0., 1., 0., .1],
                                        [0., 0., 1., 0.],
                                        [0., 0., 0., 1.]])
    kalman.measurementMatrix = 1. * np.eye(2, 4)
    kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
    kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
    kalman.errorCovPost = 1e-1 * np.eye(4, 4)
    kalman.statePost = state
    while (1):
        # use prediction or posterior as your tracking result
        ret, frame = v.read()  # read another frame
        if not ret:
            break

        img_width = frame.shape[0]
        img_height = frame.shape[1]

        def calc_point(angle):
            return (np.around(img_width / 2 + img_width / 3 * np.cos(angle), 0).astype(int),
                    np.around(img_height / 2 - img_width / 3 * np.sin(angle), 1).astype(int))

        # e.g. cv2.meanS    hift, cv2.CamShift, or kalman.predict(), kalman.correct()

        # use the tracking result to get the tracking point (pt):
        # if you track a rect (e.g. face detector) take the mid point,
        # if you track particles - take the weighted average
        # the Kalman filter already has the tracking point in the state vector

        prediction = kalman.predict()
        print("HELLOOO")
        pos = 0
        c, r, w, h = detect_one_face(frame)
        if w != 0 and h != 0:
            state = np.array([c + w / 2, r + h / 2, 0, 0], dtype='float64')
            # kalman.statePost = state
            measurement = (np.dot(kalman.measurementNoiseCov, np.random.randn(2, 1))).reshape(-1)
            measurement = np.dot(kalman.measurementMatrix, state) + measurement
            posterior = kalman.correct(measurement)
            pos = (posterior[0], posterior[1])
        else:
            measurement = (np.dot(kalman.measurementNoiseCov, np.random.randn(2, 1))).reshape(-1)
            measurement = np.dot(kalman.measurementMatrix, state) + measurement
            pos = (prediction[0], prediction[1])

        # display_kalman3(frame, pos, (c, r, w, h))
        process_noise = np.sqrt(kalman.processNoiseCov[0, 0]) * np.random.randn(4, 1)
        state = np.dot(kalman.transitionMatrix, state) + process_noise.reshape(-1)

        pt = (frameCounter, pos[0], pos[1])
        if frameCounter != 256:
            print("Clown A")
            #output.write("%d,%d,%d\n" % pt) # Write as frame_index,pt_x,pt_y
        else:
            print("Clown B")
            #output.write("%d,%d,%d" % pt)  # Write as frame_index,pt_x,pt_y
        frameCounter += 1
    # output.close()

if __name__ == "__main__":
    main()