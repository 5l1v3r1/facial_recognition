#!/usr/bin/env python3 

import cv2 

#load cascade_classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

#load image
image = cv2.imread('/root/anaconda3/envs/face-recog/testimages/0/geoffroy2.jpg') 

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

#run face detection
face_coords = face_cascade.detectMultiScale(image_gray, 1.1, 8) 

print(face_coords) 

e = 0

#draw rectangle for detect face
for face_coord in face_coords: 

    x, y, x1, y1 = face_coord 

    cv2.rectangle(image, (x, y), (x+x1, y+y1), (0,255,0), 2) 

    face = image[y:y+y1, x:x+x1] 

    cv2.imshow('face{}'.format(e), face) 

    e += 1

cv2.imshow('faces', image) 
cv2.waitKey(0) 











