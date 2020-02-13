import cv2
import numpy as np
import os
cap = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
name = input('please enter your name: ')
pics = int(input('Provide no of pics: '))
list_faces = []
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
            cv2.imshow('face', face_img)
        cv2.imshow('Main video', frame)
    key = cv2.waitKey(1)
    if ord('q') == key:
        break
    if ord('c') == key:
        if ret and len(face) > 0:
            list_faces.append(face_flat)
            print('capture face',len(list_faces))
            if len(list_faces) == pics:
                break

X = np.array(list_faces)
y = np.full((pics, 1), name)
data = np.hstack([y, X])
if os.path.exists('faces.npy'):
    old = np.load('faces.npy')
    total = np.vstack([old, data])
    np.save('faces.npy', total)
else:
    np.save('faces.npy', data)
cap.release()
cv2.destroyAllWindows()