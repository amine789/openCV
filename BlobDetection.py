# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 00:39:04 2018

@author: amine bahlouli
"""

import cv2
import numpy as np

img = cv2.imread("Sunflowers.jpg",0)
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.filterByCircularity = True
params.minCircularity = 0.00001
params.minConvexity = 0.95
params.filterByConvexity = True
detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(img)

blank=np.zeros((1,1))


blob = cv2.drawKeypoints(img,keypoints,blank,(0,255,255),cv2.DRAW_MATCHES_FLAGS_DEFAULT)


cv2.imshow("blob", blob)
cv2.waitKey(0)
cv2.destroyAllWindows()