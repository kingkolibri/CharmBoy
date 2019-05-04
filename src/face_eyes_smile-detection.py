import cv2
import numpy as np

if __name__ == '__main__':

    face_cascade = cv2.CascadeClassifier('../models/frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('../models/eye.xml')
    smile_cascade = cv2.CascadeClassifier('../models/smile.xml')
    glass_cascade = cv2.CascadeClassifier('../models/glasses_cascade.xml')

    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('../data/testvideo2.mp4')
    count = 1

    while True:

        ret, frame = cap.read()
        # Blur frame
        frame_blur = cv2.GaussianBlur(frame,(15,15),0)
        # frame_blur = frame
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
            print('Number of eyes:'+str(len(eyes)))
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
                        #cv2.circle(roi_eye_col, (cx, cy), int(2), [-1, 255, 3], 3)
                        r = 3;
                        rectX = int((cx - r))
                        rectY = int((cy - r))
                        crop_img = roi_eye_col[rectX:(rectX + 2 * r), rectY:(rectY + 2 * r)]
                        crop_hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
                        lower_yellow = np.array([30, 100, 100])
                        upper_yellow = np.array([30, 255, 170])
                        mask = cv2.inRange(crop_hsv, lower_yellow, upper_yellow)

                        cv2.imshow('ROI EYE COLOR', crop_img)

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

            edges = cv2.Canny(roi_gray, 100, 200)
            glass = glass_cascade.detectMultiScale(roi_gray, 1.04, 5)

            for (gx, gy, gw, gh) in glass:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.rectangle(roi_color, (gx, gy), (gx + gw, gy + gh), (255, 255, 0), 2)
                cv2.putText(roi_color, 'glass', (gx, gy - 3), font, 0.5, (11, 255, 255), 2, cv2.LINE_AA)

                # write here to database if with smile = +1 the longer it is

        cv2.imshow('frame', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
