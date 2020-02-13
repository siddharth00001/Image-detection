import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os
cap = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model = KNeighborsClassifier()
data = np.load('faces.npy')
X,Y = data[:,1:].astype(np.int),data[:,0]
model.fit(X,Y)
while True:
    ret, frame = cap.read()
    if ret:
        face = classifier.detectMultiScale(frame)
        if len(face) > 0:
            np_face = np.array(face)
            best_face = np.product(np_face[:, 2:], axis=1).argmax()
            x, y, w, h = face[best_face]
            crop = frame[y:y + h, x:x + w]
            face_img = cv2.resize(crop, (100, 100))
            face_gray = cv2.cvtColor(face_img,cv2.COLOR_RGB2BGR)
            face_flat = face_gray.flatten()
            fin_face = model.predict([face_flat])
            image = cv2.putText(frame,fin_face[0],(x, y), cv2.FONT_HERSHEY_SIMPLEX,fontScale= 3,color= (255, 0, 0),thickness= 4)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
        cv2.imshow('Main video', frame)

    key = cv2.waitKey(1)
    if ord('q') == key:
        break


cap.release()
cv2.destroyAllWindows()