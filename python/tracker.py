# https://github.com/srianant/kalman_filter_multi_object_tracking/blob/master/tracker.py

from cv2 import distanceTransform
import numpy as np
from Kalmanfilter_remake import KalmanFilter
#from common import dprint
from scipy.optimize import linear_sum_assignment

class Track(object):
	"Tracker for every object detected"

	def __init__(self, prediction, trackIdCount):
		'''
		prediction = predicted centroids to be tracked
		trackIdCount = identification of each track object
		'''
		self.track_id = trackIdCount
		self.KF = KalmanFilter() #kf instance to track this object
		self.prediction = np.asarray(prediction) # predicted centroids(x,y)
		self.skipped_frames = 0 
		self.trace = [] # tracing path

	


class Tracker(object):
	"Tracker class updatding track vectors of tracked objects"

	def __init__(self, dist_thresh, max_skipped_frames, max_trace_length, trackIdCount):
		"""
		dist_thresh: distance threshold. When exceeds the threshold,
                         track will be deleted and new track is created
        max_skipped_frames: maximum allowed frames to be skipped for
                                the track object undetected
        max_trace_lenght: trace path history length
        trackIdCount: identification of each track object
		"""

		self.dist_thresh = dist_thresh
		self.max_skipped_frames = max_skipped_frames
		self.max_trace_length = max_trace_length
		self.tracks = []
		self.trackIdCount = trackIdCount

	def Update(self, detections):
		"""
		Updating tracking vector:
		1. create tracks if no track vector found
		2. calculate cost using sum of square distance btw prediction and detection centroids
		3. Hungarian Algorithm to assign correct detections to predicted tracks
		4. Identify tracks with no assignment
		5. If tracks not detected for certain amount of time - remove them
		6. look for un-assigned detections
		7. start new tracks
		8. update kalman state, last results and tracks trace
		"""

		#create tracks if no track vector found
		if(len(self.tracks)==0):
			for i in range(len(detections)): 
				track = Track(detections[i], self.trackIdCount) #creates an Track object for each detection ->
				self.trackIdCount += 1
				self.tracks.append(track)

		#print(len(detections))
		#print(len(self.tracks)) # 17 in this case

		# Calculate cost using sum of square distance
		# predicted vs detected centroids
		N = len(self.tracks)
		M = len(detections)
		cost = np.zeros(shape=(N,M))
		for i in range(len(self.tracks)):
			for j in range(len(detections)):
				try:
					diff = self.tracks[i].prediction - detections[j]
					dist = np.sqrt(diff[0][0]*diff[0][0]+diff[1][0]*diff[1][0])
					cost[i][j] = dist
				except:
					pass


		# average squared error
		cost = (0.5)*cost
		#print(cost) # prints propper cost matrix for each detection

		#Hungarian Algorithm for correct detection assignement to predicted tracks
		assignment = [] 
		for _ in range(N):
			assignment.append(-1)
		row_ind, col_ind = linear_sum_assignment(cost) 
		for j in range(len(row_ind)):
			assignment[row_ind[j]]= col_ind[j]

		#print(assignment) #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

		#Identify tracks with no assignment, if any
		un_assigned_tracks = []
		for i in range(len(assignment)):
			if (assignment[i] != -1):
				# check for cost distance threshold
				# if cost too high, un_assing(delete) track
				if (cost[i][assignment[i]] > self.dist_thresh):
					assignment[i] = -1
					un_assigned_tracks.append(i)
				pass
			else:
				self.tracks[i].skipped_frames +=1 # if cost not too high, add a skipped frame to that tracks record
		#print(un_assigned_tracks)

		#if tracks not detected for long time - remove them
		del_tracks = []
		for i in range(len(self.tracks)):
			if (self.tracks[i].skipped_frames > self.max_skipped_frames):
				del_tracks.append(i)
		if len(del_tracks) > 0:
			for id in del_tracks:
				if id < len(self.tracks):
					del self.tracks[id]
					del assignment[id]
				else:
					print("ERROR: id is greater than length of tracks")
			
		# look for un-assigned detections
		un_assigned_detects = []
		for i in range(len(detections)):
			if i not in assignment:
				un_assigned_detects.append(i)

		#start new track for unassigned detection 
		if (len(un_assigned_detects) != 0):
			for i in range(len(un_assigned_detects)):
				track = Track(detections[un_assigned_detects[i]], self.trackIdCount)
			self.trackIdCount +=1
			self.tracks.append(track)
		
		#update kalmanfilter state, lastResult and tracks trace
		for i in range(len(assignment)):
			self.tracks[i].KF.predict()

			if (assignment[i] != -1):
				self.tracks[i].skipped_frames = 0
				self.tracks[i].prediction = self.tracks[i].KF.correct(detections[assignment[i]],1)
			
			else:
				self.tracks[i].prediction = self.tracks[i].KF.correct(np.array([[0][0]]),0)

			if (len(self.tracks[i].trace) > self.max_trace_length):
				for j in range(len(self.tracks[i].trace)- self.max_trace_length):
					del self.tracks[i].trace[j]

			self.tracks[i].trace.append(self.tracks[i].prediction)
			self.tracks[i].KF.lastResult = self.tracks[i].prediction

