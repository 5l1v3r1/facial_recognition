#!/usr/env/bin python3 

import cv2 

image = cv2.imread('/root/anaconda3/envs/face-recog/testimages/1/download3.jpeg') 

cv2.imshow('Ladies', image) 
cv2.waitKey(0) 

image.release() 
cv2.destroyAllWindows()

