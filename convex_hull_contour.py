# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 03:25:48 2018

@author: amine bahlouli
"""

import numpy as np
import cv2

image = cv2.imread("hand.jpg")
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("original image",image)
cv2.waitKey(0)

#threshhold image

ret, thresh = cv2.threshold(gray,176,255,0)
cv2.imshow("original image",thresh)
cv2.waitKey(0)

#find contours
_,contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)


#sort contours by area and remove the largest frame contour
n = len(contours)-1

contours = sorted(contours, key=cv2.contourArea, reverse=False)[:n]

for c in contours:
    hull = cv2.convexHull(c)
    cv2.drawContours(image,[hull],0,(0,255,0),2)
    cv2.imshow("convex hull ", image)

cv2.waitKey(0)