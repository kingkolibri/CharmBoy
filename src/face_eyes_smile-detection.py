import cv2
import numpy as np


def get_circle(img, minR, maxR, p1, p2):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=p1, param2=p2, minRadius=minR, maxRadius=maxR)
    circles = np.uint8(np.around(circles))

    # return the first circle
    return circles[0][0]


def print_img(img, x, y, w, h):
    if w != 0 and h != 0:
        cv2.imshow('detected circles', img[x:x + w, y:y + h])
        cv2.imwrite('app-1/1.jpg', img[x:x + w, y:y + h])
    else:
        cv2.imshow('detected circles', img)
        cv2.imwrite('app-1/1.jpg', img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


def reject_out(img, xc, yc, r):
    row = len(img)
    col = len(img[0])

    for x in range(0, row):
        for y in range(0, col):
            res = (x - xc) * (x - xc) + (y - yc) * (y - yc)
            if res > r * r:
                img[x][y] = 0


def reject_in(img, xc, yc, r):
    xs = xc - r
    ys = yc - r

    for x in range(xs, xs + 2 * r):
        for y in range(ys, ys + 2 * r):
            res = (x - xc) * (x - xc) + (y - yc) * (y - yc)
            if res < r * r:
                img[x][y] = 0


def extract_iris(img):
    NoneType = type(None)
    if type(img) == NoneType:
        return False

    blurimg = cv2.medianBlur(img, 5)
    grayimg = cv2.cvtColor(blurimg, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grayimg, 100, 200)

    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=0, maxRadius=100)
    if type(circles) == NoneType:
        return False

    circles = np.uint16(circles)
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
    cv2.imshow('detected circles', img)

    # edges = cv2.Canny(grayimg, 100, 200)
    # cord = get_circle(edges, 35, 0, 50, 40)
    # draw the outer circle
    # cv2.circle(img,(cord[0],cord[1]),cord[2],(0,255,0),2)
    # draw the center of the circle
    # cv2.circle(img,(cord[0],cord[1]),2,(0,0,255),3)

    # h = 2*cord[2]
    # w = 2*cord[2]
    # x = cord[1]-cord[2]
    # y = cord[0]-cord[2]
    # nimg = img[x:x+w,y:y+h]

    # reject_out(nimg, h/2, w/2, h/2)
    # print_img(cimg,0,0,0,0)

    # nimg = cv2.cvtColor(nimg, cv2.COLOR_GRAY2BGR)
    # cord    = get_circle(nimg, 0, cord[2]-1, 50, 30)
    # print (nimg[0:w][cord[0]])
    # draw the outer circle
    # cv2.circle(nimg,(cord[0],cord[1]),cord[2],(0,255,0),2)
    # draw the center of the circle
    # cv2.circle(nimg,(cord[0],cord[1]),2,(0,0,255),3)
    # reject_in(nimg, cord[1], cord[0], cord[2])
    # print_img(nimg,0,0,0,0)
    # print (nimg[0:w][cord[0]])


if __name__ == '__main__':

    face_cascade = cv2.CascadeClassifier('../models/frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('../models/eye.xml')
    smile_cascade = cv2.CascadeClassifier('../models/smile.xml')

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('../temp/testvid.mp4')
    count = 1

    while True:

        ret, frame = cap.read()
        # Blur frame
        # frame_blur = cv2.GaussianBlur(frame,(15,15),0)
        frame_blur = frame
        frame_gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)

        # Display resulting frame
        cv2.imshow('frame', frame_gray);
        faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            roi_gray = frame_gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            eyes = eye_cascade.detectMultiScale(roi_gray)

            smile = smile_cascade.detectMultiScale(roi_gray,
                                                   scaleFactor=1.7,
                                                   minNeighbors=22,
                                                   minSize=(25, 25),
                                                   )

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                # Extract eyes as region of interest (use of gray, blurred one):
                roi_eye = roi_gray[ey: ey + eh, ex: ex + ew]
                roi_eye_col = roi_color[ey: ey + eh, ex: ex + ew]

                # Extract contours
                contours, hierarchy = cv2.findContours(roi_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                if len(contours) != 0:
                    cv2.drawContours(roi_eye_col, contours, -1, 255, 3)
                    cnt = max(contours, key=cv2.contourArea)

                    M = cv2.moments(cnt)

                    if (M['m00'] != 0):
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        cv2.circle(roi_eye_col, (cx, cy), int(2), [-1, 255, 3], 3)

                roi_eye_hsv = cv2.cvtColor(roi_eye_col, cv2.COLOR_BGR2HSV)

                cv2.imshow('processed_frame', roi_eye_col)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                # extract_iris(roi_eye)

                # Pupil radius and iris detection is both naive and (somewhat) novel. Both detection mechanisms search for circles originating at the pupil center. Pupil radius detection iteratively searches for the largest, darkest contour in a given search space.
                #
                # Iris detection generates possible choices and decides which radius is best given the fact that if the search space extends too wide, the resulting circle will begin to accumulate white pixels since the sclera of the eye is white in color.
                #
                # Eye color detection uses the OpenCV k-nearest neighbors (kNN) algorithm.

            for (ex, ey, ew, eh) in smile:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)

                # write here to database if with smile = +1 the longer it is

        cv2.imshow('frame', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
