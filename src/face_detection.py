import cv2
import dlib
import matplotlib.pyplot as plt
import tensorflow as tf
from CharmBoy.models.mtcnn import detect_face
from CharmBoy.models.mtcnn.align_dlib import AlignDlib


class FaceDetector:

    def __init__(self):

        align = AlignDlib('../models/dlib/shape_predictor_68_face_landmarks.dat')
        self.detector = dlib.get_frontal_face_detector()

        # Initialize model
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess, None)
        self.minsize = 20  # minimum size of face
        self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        self.factor = 0.709  # scale factor

    def detect_face_dlib(self, img):
        bbs = self.detector(img, 1)
        tuples = []
        for r in bbs:
            tuples.append((r.left(), r.top(), r.right(), r.bottom()))
        return tuples

    def detect_face_and_landmarks_mtcnn(self, img):
        img = img[:, :, 0:3]
        bbs, lms = detect_face.detect_face(img, self.minsize, self.pnet, self.rnet, self.onet, self.threshold,
                                           self.factor)
        boxes = []
        landmarks = []
        face_index = 0
        for r in bbs:
            r = r.astype(int)
            points = []
            for i in range(5):
                points.append((lms[i][face_index], lms[i + 5][face_index]))
            landmarks.append(points)
            boxes.append((r[0], r[1], r[2], r[3]))
            # boxes.append(r[:4].astype(int).tolist())
            face_index += 1
        return boxes, landmarks

    def draw_landmarks(self, image, points):
        result = image.copy()
        for point in points:
            cv2.circle(result, point, 3, (0, 255, 0), -1)
        return result


if __name__ == '__main__':

    cap = cv2.VideoCapture('/home/kingkolibri/Videos/Webcam/2019-05-04-191747.mp4')

    detector = FaceDetector()

    if not cap.isOpened():
        print("Maybe opencv VideoCapture can't open it")
        exit(0)

    print("Correctly opened resource, starting to show feed.")

    rval, frame = cap.read()
    while rval:
        # cv2.imshow('frame', frame)
        rval, frame = cap.read()

        cv2.imshow('frame', frame)

        bbs, lm = detector.detect_face_and_landmarks_mtcnn(frame)
        cv2.imshow('frame', detector.draw_landmarks(frame, lm[0]))

        key = cv2.waitKey(20)
        # print "key pressed: " + str(key)
        # exit on ESC, you may want to uncomment the print to know which key is ESC for you
        if key == 27 or key == 1048603:
            break

    cap.release()
    cv2.destroyAllWindows()
