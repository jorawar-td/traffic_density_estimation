# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import glob

from darkflow.net.build import TFNet

# Path to the directory of output file
files = glob.glob('output/*.png')
for f in files:
	os.remove(f)

# Adding the sorting algorithm
from sort import *
tracker = Sort()
memory = {}
line = [(986,353),(621,300)]
counter = 0
person = 0
bike = 0
other = 0

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# load the COCO class labels our YOLO model was trained on
# labelsPath = os.path.sep.join(["yolo-coco", "coco.names"])
# LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
# np.random.seed(42)
# COLORS = np.random.randint(0, 255, size=(200, 3),dtype="uint8")

# THis is used to load the pre-trained TFmodel that is saved in the disk
print("[INFO] loading YOLO from disk...")
options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.5,'gpu':1.0}
# options = {"pbLoad": "built_graph/yolo.pb", "metaLoad": "built_graph/yolo.meta", "threshold": 0.5,"gpu":1.0}
tfnet = TFNet(options)

# initialize the video stream, pointer to output video file, and
# frame dimensions
cap = cv2.VideoCapture('du.mp4')
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
(W, H) = (None, None)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (int(width), int(height)))


def boxing(original_img, predictions):
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

	return newImage

cc = 0
while True:
	(grabbed, frame) = cap.read()

	if not grabbed:
		break
	
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	boxes = []
	confidences = []
	classIDs = []
	dets = []
	cc += 1
	frame = np.asarray(frame)
	results = tfnet.return_predict(frame)

	new_frame = boxing(frame, results)

	dets = np.asarray(dets)
	tracks = tracker.update(dets)
	
	boxes = []
	indexIDs = []
	c = []
	previous = memory.copy()
	print("Previous",previous)
	memory = {}

	for track in tracks:
		boxes.append([track[0], track[1], track[2], track[3]])
		indexIDs.append(int(track[4]))
		memory[indexIDs[-1]] = boxes[-1]

	if len(boxes) > 0:
		i = int(0)
		for box in boxes:
			# extract the bounding box coordinates
			(x, y) = (int(box[0]), int(box[1]))
			(w, h) = (int(box[2]), int(box[3]))

			# ! Isse do bounding box aa rhe hai, galat hai
			# color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
			# cv2.rectangle(new_frame, (x, y), (w, h), color, 2)

			if indexIDs[i] in previous:
				previous_box = previous[indexIDs[i]]
				(x2, y2) = (int(previous_box[0]), int(previous_box[1]))
				(w2, h2) = (int(previous_box[2]), int(previous_box[3]))
				p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
				p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
				cv2.line(new_frame, p0, p1, (255,255,255), 6)

				if intersect(p0, p1, line[0], line[1]): 
					if classIDs[i] == 'car':
						counter += 1
					elif classIDs[i] == 'motorbike':
						print("BIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
						bike += 1
					elif classIDs[i] == 'person':
						print("BUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU")
						person += 1
					else :
						print("OTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT")
						other += 1

			i += 1

	# draw line
	cv2.line(new_frame, line[0], line[1], (0, 255, 255), 5)

	# draw counter
	cv2.putText(new_frame, "Cars "+str(counter), (100,200), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 255, 255), 5)
	cv2.putText(new_frame, "Bike "+str(bike), (100,250), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 100, 100), 5)
	cv2.putText(new_frame, "person  "+str(person), (100,300), cv2.FONT_HERSHEY_DUPLEX, 2.0, (200, 200, 255), 5)
	cv2.putText(new_frame, "Other "+str(other), (100,350), cv2.FONT_HERSHEY_DUPLEX, 2.0, (100, 100, 255), 5)

	print("yooooooooooooo")
	# Display the resulting frame
	out.write(new_frame)
	cv2.imshow('frame',new_frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cc = {'cars':counter,'person':person,'bike':bike,'other':other}
print("[INFO] Cleaning Up...",cc)
print(classIDs)
cap.release()
out.release()
cv2.destroyAllWindows()
