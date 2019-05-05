
import cv2
import numpy as np

#cap = cv2.VideoCapture('./testvideo.mp4')
cap = cv2.VideoCapture(0)


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']


def initialize_caffe_model():
    print ('Loading models...')
    age_net = cv2.dnn.readNetFromCaffe(
        "pretrained_models/deploy_age.prototxt",
        "pretrained_models/age_net.caffemodel")
    gender_net = cv2.dnn.readNetFromCaffe(
        "pretrained_models/deploy_gender.prototxt",
        "pretrained_models/gender_net.caffemodel")
    print('Models successfully loaded')
    return (age_net, gender_net)


def capture_loop(age_net, gender_net):

    font = cv2.FONT_HERSHEY_SIMPLEX
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Operations on frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('../pretrained_models/haarcascade_frontalface_alt.xml')
        #test = face_cascade.load('../pretrained_models/haarcascade_frontalface_default.xml')
        #print (test)
        faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
        # Draw a rectangle around every found face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame_gray, (x, y), (x + w, y + h), (255, 255, 0), 2)
            face_img = frame_gray[y:y + h, x:x + w].copy()
            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            # Predict gender
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()] # gender prediction
            # Predict age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()] #age prediction
            overlay_text = "%s, %s" % (gender, age)
            cv2.putText(frame_gray, overlay_text, (x, y), font, 2, (255, 255, 255), 2, cv2.LINE_AA)

        #Display frame
        cv2.imshow("Image", frame_gray)

        if cv2.waitKey(1) & 0xFF == ord ('q'):
            break


if __name__ == '__main__':
    age_net, gender_net = initialize_caffe_model()
    capture_loop(age_net, gender_net)