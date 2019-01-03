# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 02:50:11 2018

@author: amine bahlouli
"""

import cv2
import numpy as np

def x_cord_contour(contours):
    # returns the x cordinate  for the contour centroid
    if cv2.contourArea(contours)>10:
        M = cv2.moments(contours)
        return (int(M["m10"]/M["m00"]))
    
def label_contour_center(image,c):
    # places a red circle on the centers of contours
    M = cv2.moments(c)
    cx = (int(M["m10"]/M["m00"]))
    cy = (int(M["m01"]/M["m00"]))
    
    # draw circles in 
    cv2.circle(image,(cx,cy),10,(0,0,255),-1)
    return image

image = cv2.imread("shapes.jpg")
original_image = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#canny edge
edged = cv2.Canny(gray, 50,200)
cv2.imshow("canny edge",edged)
cv2.waitKey(0)
_,contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

for (i,c) in enumerate(contours):
    orig = label_contour_center(image,c)
    
cv2.imshow("4 contour = centers", image)
cv2.waitKey(0)

contours_left_to_right = sorted(contours, key=x_cord_contour,reverse=False)

for (i,c) in enumerate(contours_left_to_right):
    cv2.drawContours(original_image,[c], -1,(0,0,255),-3)
    M = cv2.moments(c)
    cx = (int(M["m10"]/M["m00"]))
    cy = (int(M["m01"]/M["m00"]))
    cv2.putText(original_image,str(i+1),(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow("left_to_rate", original_image)
    cv2.waitKey(0)
    
    (x,y,w,h) = cv2.boundingRect(c)
    
    cropped_countour = original_image[y:y+h, x:x+w]
    image_name = "output_number "+str(i+1)+".jpg"
    cv2.imwrite(image_name,cropped_countour)
cv2.destroyAllWindows()
    
    