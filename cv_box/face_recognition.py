import cv2
import numpy as np
import pickle

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('./mytrainer.xml')

labels = {}

with open('label.pickle', 'rb') as f:
   origin_labels =  pickle.load(f)
   labels = {v:k for k, v in origin_labels.items()}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            gray_roi = gray[y:y+h, x:x+w]
            id_, conf = recognizer.predict(gray_roi)
            if conf >= 60:
                print(labels[id_])
                cv2.recognition(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)

                cv2.putText(frame, str(labels[id_]), (x, y - 15),cv2.FONT_HERSHE_SIMPLEX,2, (255, 0, 0),2)
            cv2.imshow('Result', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
