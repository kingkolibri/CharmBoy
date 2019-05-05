# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages

import numpy as np
import imutils
from imutils import face_utils
import dlib
import cv2
from scipy import stats


# def checkeyecolor(img):

# check validität


def isolateROI(img, landmarks):
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
        roi = img[y:y + h, x:x + w]
        roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
        # show the particular face part
        cv2.imshow("ROI", roi)
        cv2.imshow("Image", clone)

        # Particular check for right and left eye for color detection
        # if name =='right_eye' | name == 'left_eye'
        #	checkeyecolor(img)
        # mouthright_eyebrowleft_eyebrowright_eyeleft_eyenosejaw
    cv2.waitKey(0)

    return img


def facedetection(img, detector, predictor):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detects faces in the grayscale image (multiple!)
    rects = detector(gray, 1)

    # loop over the face detections
    # for (i, rect) in enumerate(rects):

    rect = rects[0]
    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    img = isolateROI(img, shape)

    # convert dlib's rectangle to a OpenCV-style bounding box
    # [i.e., (x, y, w, h)], then draw the face bounding box
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

    img_face = img[y:(y + h), x:(x + w)]

    # show the face number (only needed if multiple faces are detected)
    # cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # loop over the (x, y)-coordinates for the facial landmarks and draw them on the image
    for (x, y) in shape:
        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    return img, img_face, shape


def is_facing_camera(img, landmarks, threshold=25):

    face_points = landmarks[0:17]
    x_points = face_points[:, 0]

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_points, face_points[:,1])
    y_points = slope * x_points + intercept
    line = np.array([x_points, y_points]).transpose()


    d = []
    for point in face_points:
        d.append(np.linalg.norm(np.cross(line[-1] - line[0], line[0] - point)) / \
            np.linalg.norm(line[-1] - line[0]))

    d = np.array(d)
    i = 0
    j = 16
    symmetrie = 0
    while j-i > 1:
        symmetrie = symmetrie + d[j] - d[i]
        i = i+1
        j = j-1

    if symmetrie < threshold:
        return True
    else:
        return False


if __name__ == '__main__':

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('../models/shape_predictor_68_face_landmarks.dat')
    cap = cv2.VideoCapture('../data/testvideo.mp4')
    count = 1

    while True:
        # load the input image, resize it, and convert it to grayscale
        ret, frame = cap.read()

        frame, img_face, landmarks = facedetection(frame, detector, predictor)
        if is_facing_camera(img=img_face, landmarks=landmarks, threshold=25):
            print("Person looking at us! Panic!!")
        else:
            print("No person facing us detected.")

        # show the output image with the face detections + facial landmarks
        cv2.imshow('img_face', img_face)
        cv2.imshow('frame', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
