# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import glob
import time
import pandas as pd
import matplotlib.pyplot as plt

from darkflow.net.build import TFNet
from sort import *

print("[INFO] loading YOLO from disk...")
options = {"pbLoad": "built_graph/yolo.pb", "metaLoad": "built_graph/yolo.meta", "threshold": 0.5,"gpu":1.0}
tfnet = TFNet(options)

counter = 0
person = 0
bike = 0
other = 0
# line = [(1090,936),(103,579)]
line = [(0,600),(1359,703)]
# 0,600,1359,703
# 0,536,1359,639
tracker = Sort()
# memory = {}
# line = [(986,353),(621,300)]
# new_frame = None

class Count():
	def __init__(self,file_path,line_in):
		self.file_path = file_path
		self.line_in = line_in
		# self.graph_path = graph_path
		# self.counter = counter
		# self.bike = bike
		# self.person = person
		# self.other = other
		self.memory = {}

	# Return true if line segments AB and CD intersect
	def intersect(self,A,B,C,D):
		return self.ccw(A,C,D) != self.ccw(B,C,D) and self.ccw(A,B,C) != self.ccw(A,B,D)

	def ccw(self,A,B,C):
		return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


#  THis method takes the list of predictions and and if the confidence of that prediction is greater than 40% then it draws a bounding box around that
#  and classifies that object
	def boxing(self,original_img, predictions):
		boxes = []
		confidences = []
		classIDs = []
		dets = []

		newImage = np.copy(original_img)

		for result in predictions:
			top_x = result['topleft']['x']
			top_y = result['topleft']['y']

			btm_x = result['bottomright']['x']
			btm_y = result['bottomright']['y']

			confidence = result['confidence']
			label = result['label'] + " " + str(round(confidence, 3))

			if confidence > 0.4:
				newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255,0,0), 3)
				newImage = cv2.putText(newImage, label, (top_x, top_y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.8, (0, 230, 0), 1, cv2.LINE_AA)

				boxes.append([top_x,top_y,int(btm_x-top_x),int(btm_y-btm_x)])
				confidences.append(float(confidence))
				classIDs.append(result['label'])
				dets.append([top_x,top_y,btm_x,btm_y,confidence])

		return newImage, boxes, confidences, classIDs, dets

# This method processes the vidoe and also keeps the count of different vehicles and displays it
	def vehicle(self):
		# print("[INFO] loading YOLO from disk...")
		# options = {"pbLoad": "built_graph/yolo.pb", "metaLoad": "built_graph/yolo.meta", "threshold": 0.5,"gpu":1.0}
		# # options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.4,"gpu":1.0}
		# tfnet = TFNet(options)

		# initialize the video stream, pointer to output video file, and frame dimensions
		cap = cv2.VideoCapture(self.file_path)
		width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   
		height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
		# cap.set(1280,720)
		(W, H) = (None, None)

		fourcc = cv2.VideoWriter_fourcc(*'DIVX')
		out = cv2.VideoWriter('output.avi',fourcc, 20.0, (int(width), int(height)))

		coo = 0
		counter = 0
		bike = 0
		person = 0
		other = 0
		while True:
			(grabbed, frame) = cap.read()
			print("DAta read")
			if not grabbed:
				break
			
			if W is None or H is None:
				(H, W) = frame.shape[:2]

			coo+=1
			# if (coo % 3 == 0):
			# resize = cv2.resize(frame, (720, 480), interpolation = cv2.INTER_LINEAR)
			frame = np.asarray(frame)
			results = tfnet.return_predict(frame)

			new_frame,boxes, confidences, classIDs, dets = self.boxing(frame, results)
			# global dets, tracker, counter, bike, person, other, memory, line
			dets = np.asarray(dets)
			tracks = tracker.update(dets)
			
			boxes = []
			indexIDs = []
			c = []
			previous = self.memory.copy()
			print("Previous",previous)
			self.memory = {}
		
			for track in tracks:
				boxes.append([track[0], track[1], track[2], track[3]])
				indexIDs.append(int(track[4]))
				self.memory[indexIDs[-1]] = boxes[-1]

			if len(boxes) > 0:
				i = int(0)
				for box in boxes:
					# extract the bounding box coordinates
					(x, y) = (int(box[0]), int(box[1]))
					(w, h) = (int(box[2]), int(box[3]))
					p0 = (int(x + (w-x)/2), int(y + (h-y)/2))

					if indexIDs[i] in previous:
						previous_box = previous[indexIDs[i]]
						(x2, y2) = (int(previous_box[0]), int(previous_box[1]))
						(w2, h2) = (int(previous_box[2]), int(previous_box[3]))
						p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
						# cv2.line(new_frame, p0, p1, (255,255,255), 6)

						if self.intersect(p0, p1, line[0], line[1]): 
							if classIDs[i] == 'car':
								counter += 1
								cv2.line(new_frame, p0, p1, (0,0,204), 6)
							elif classIDs[i] == 'motorbike':
								print("BIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
								bike += 1
								cv2.line(new_frame, p0, p1, (255,255,0), 6)
							elif classIDs[i] == 'person':
								print("BUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU")
								person += 1
								cv2.line(new_frame, p0, p1, (51,0,0), 6)
							else :
								print("OTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT")
								other += 1
								cv2.line(new_frame, p0, p1, (255,0,0), 6)
					i += 1

			# draw line
			cv2.rectangle(new_frame, line[0], line[1], (0, 255, 255), cv2.FILLED)

			# draw counter
			cv2.putText(new_frame, "Cars "+str(counter), (100,200), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 255, 255), 5)
			cv2.putText(new_frame, "Bike "+str(bike), (100,250), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 100, 100), 5)
			cv2.putText(new_frame, "person  "+str(person), (100,300), cv2.FONT_HERSHEY_DUPLEX, 2.0, (200, 200, 255), 5)
			cv2.putText(new_frame, "Other "+str(other), (100,350), cv2.FONT_HERSHEY_DUPLEX, 2.0, (100, 100, 255), 5)

			print("yooooooooooooo")
			out.write(new_frame)
			cv2.imshow('frame',new_frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			
		print("[INFO] Cleaning Up...",self.line_in,"frames ",coo)
		# print(classIDs)
		cap.release()
		out.release()
		cv2.destroyAllWindows()
		cc = {'car':counter,'person':person,'bike':bike,'other':other}
		print("cc ",cc)
		return cc

# var = Count('du.mp4',[(986,353),(621,300)])
# start_time = time.time()
# countt = var.vehicle()
# print(countt)
# print("--- %s seconds ---" % (time.time() - start_time))
