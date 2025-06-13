import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

count = 0

while True:
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade,detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
       
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            count += 1
            cv2.imwrite('dataset/' + str(count) + '.jpg', gray[y:y+h,x:x+w])

        
        cv2.imshow('frame', frame)
        cv2.imshow('gray', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif count == 50:
        break

cap.release()
cv2.destroyAllWindows()



