# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 01:58:42 2018

@author: amine bahlouli
"""

import cv2
import numpy as np

image = cv2.imread("blobs.jpg")

cv2.imshow(",",image)
cv2.waitKey(0)

params = cv2.SimpleBlobDetector_Params()
#params.filterByArea = True
#params.filterByCircularity = True
#params.minCircularity = 0.00001
#params.minConvexity = 0.95
#params.filterByConvexity = True
detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(image)

blank=np.zeros((1,1))
blob = cv2.drawKeypoints(image,keypoints,blank,(0,255,255),cv2.DRAW_MATCHES_FLAGS_DEFAULT)
number_of_blobs = len(keypoints)
text= "total number of blobs "+str(len(keypoints))
cv2.putText(blob,text,(20,550),cv2.FONT_HERSHEY_SIMPLEX,1,(100,0,255),2)

cv2.imshow("blobs using default paramers",blob)
cv2.waitKey(0)


params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.filterByCircularity = True
params.minCircularity = 0.9

params.filterByConvexity = True
params.minConvexity = 0.2
params.filterByInertia=True
params.minInertiaRatio=0.01
detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(image)

blank=np.zeros((1,1))
blob = cv2.drawKeypoints(image,keypoints,blank,(0,255,255),cv2.DRAW_MATCHES_FLAGS_DEFAULT)
number_of_blobs = len(keypoints)
text= "total number of blobs "+str(len(keypoints))
cv2.putText(blob,text,(20,550),cv2.FONT_HERSHEY_SIMPLEX,1,(100,0,255),2)

cv2.imshow("blobs using default paramers",blob)
cv2.waitKey(0)
