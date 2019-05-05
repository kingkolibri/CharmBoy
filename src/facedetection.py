# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages

import numpy as np
import imutils
from imutils import face_utils
import dlib
import cv2


def detect_smile(img_mouth, detector_smile):
	smile = detector_smile.detectMultiScale(img_mouth,
											scaleFactor=1.7,
											minNeighbors=22,
											minSize=(25, 25))
	for (ex, ey, ew, eh) in smile:
		cv2.rectangle(img_mouth, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)

	cv2.imshow("smile", img_mouth)
	cv2.waitKey(0)
	if len(smile) > 0:
		smile = 10
		print('smile')
	else:
		smile = 0
		print('no smile')

	smile = {
		"smile": smile
	}

	return smile


def checkeyecolor(img, landmarks):
	'''
		This method gets a picture of an eye as input and detects its eye color by:
		:return: color: vector [brown, blue, green]
	'''
	'''
		:param img: hsv image
		:return:
		'''
	img_smooth = cv2.blur(img, (5, 5))
	img_hsv = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2HSV)

	lower_blue = np.array([100, 50, 100])
	upper_blue = np.array([140, 255, 255])

	mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)

	lower_green = np.array([40, 50, 100])
	upper_green = np.array([80, 255, 255])

	mask_green = cv2.inRange(img_hsv, lower_green, upper_green)

	lower_brown = np.array([10, 20, 70])
	upper_brown = np.array([45, 255, 200])

	mask_brown = cv2.inRange(img_hsv, lower_brown, upper_brown)

	output_brown = cv2.bitwise_and(img, img, mask=mask_brown)
	ratio_brown = np.round((cv2.countNonZero(mask_brown) / (img.size / 3)) * 100, 2)
	brown = 0

	output_green = cv2.bitwise_and(img, img, mask=mask_green)
	ratio_green = np.round((cv2.countNonZero(mask_green) / (img.size / 3)) * 100, 2)
	green = 0

	output_blue = cv2.bitwise_and(img, img, mask=mask_blue)
	ratio_blue = np.round((cv2.countNonZero(mask_blue) / (img.size / 3)) * 100, 2)
	blue = 0

	if ratio_brown > ratio_green and ratio_brown > ratio_blue and ratio_brown > 10:
		brown = 10

	if ratio_green > ratio_brown and ratio_green > ratio_blue and ratio_green > 10:
		green = 10

	if ratio_blue > ratio_green and ratio_blue > ratio_brown and ratio_blue > 10:
		blue = 10

	color = {
		"blue": blue,
		"brown": brown,
		"green": green
	}

	return color


def isolate_roi(img, landmarks, detector_smile):
	'''
	This method isolates the different regions of interests (ROI).
	The ROIs are depending on the following landmarks:
		- jaw: 0-17
		- right_eyebrow: 17-22
		- left_eyebrow: 22-27
		- nose: 27-35
		- right_eye: 36-42
		- left_eye: 42-48
		- mouth: 48 - 68
	'''

	features = {
		"blue": 0,
		"brown": 0,
		"green": 0,
		"smile": 0
	}

	for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
		# clone the original image so we can draw on it, then
		# display the name of the face part on the image
		clone = img.copy()
		cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
					0.7, (0, 0, 255), 2)
		# loop over the subset of facial landmarks, drawing the specific face part
		for (x, y) in landmarks[i:j]:
			cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

		# extract the ROI of the face region as a separate image
		(x, y, w, h) = cv2.boundingRect(np.array([landmarks[i:j]]))

		# Particular check for right and left eye for color detection
		if name == 'right_eye':
			roi = img[y:y + h, int((x + w)/2-h):int((x + w)/2+h)]
			color = checkeyecolor(roi, landmarks[i:j])
			features.update(color)

		if name == 'left_eye':
			roi = img[y:y + h, int((x + w)/2-h):int((x + w)/2+h)]
			color = checkeyecolor(roi, landmarks[i:j])
			features.update(color)

		if name == 'mouth':
			smile = detect_smile(img, detector_smile)
			features.update(smile)
		# mouthright_eyebrowleft_eyebrowright_eyeleft_eyenosejaw




	return img, features


def facedetection(img, detector, predictor, detector_smile):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# detects faces in the grayscale image (multiple!)
	rects = detector(gray, 1)

	rect = rects[0]
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)

	img, features = isolate_roi(img, shape, detector_smile)



	# convert dlib's rectangle to a OpenCV-style bounding box
	# [i.e., (x, y, w, h)], then draw the face bounding box
	(x, y, w, h) = face_utils.rect_to_bb(rect)
	cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

	img_face = img[y:(y + h), x:(x + w)]


	# loop over the (x, y)-coordinates for the facial landmarks and draw them on the image
	for (x, y) in shape:
		cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

	return img, img_face, features


if __name__ == '__main__':



	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('../models/shape_predictor_68_face_landmarks.dat')
	smile_cascade = cv2.CascadeClassifier('../models/smile.xml')

	# Load Video
	cap = cv2.VideoCapture('../data/testvideo_smile.mp4')

	while True:
		# load the input image, resize it, and convert it to grayscale
		ret, frame = cap.read()

		frame, img_face, features = facedetection(frame, detector, predictor, smile_cascade)
		print(features)
