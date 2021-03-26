"""
    Using OPENCv for live face detection
    """

import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haar.xml')

capture = cv2.VideoCapture(0)

while True:
    _, img = capture.read()  # reading the image
    img = cv2.flip(img, 1)  # flipping the image cause I like mirrored captures

    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # flipping the grayscaled version too, else the face detection box won't come in the same place.
    gray_scale = cv2.flip(gray_scale, 1)
    faces = face_cascade.detectMultiScale(gray_scale, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Mirrored', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

capture.release()
