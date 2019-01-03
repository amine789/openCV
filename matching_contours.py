# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 18:57:14 2018

@author: amine bahlouli
"""
import cv2
import numpy as np
import sys



# load target image with shapes we are trying to match

target = cv2.imread("shapestomatch.jpg",0)

def get_contour(img):
    ref_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(ref_gray,127,255,0)
    _,contours,hierarchy= cv2.findContours(thresh,1,2)
    
    for c in contours:
        area = cv2.contourArea(c)
        total_area = img.shape[0]*img.shape[1]
        
        if 0.05<float(area/total_area)<0.8:
            return c


def get_all(img):
    ref_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(ref_gray,127,255,0)
    _,contours,hierarchy= cv2.findContours(thresh,1,2)
    return contours

template = cv2.imread("4star.jpg")


# load target image with shapes we are trying to match

target = cv2.imread("shapestomatch.jpg")

# extracts the reference shape
ref_inp = get_contour(template)
all_inp = get_all(target)
min_dist = sys.maxsize
closest_contour = ref_inp[0]
for c in all_inp:
    ret = cv2.matchShapes(ref_inp,c,1,0)
    if ret<min_dist:
        min_dist=ret
        closest_contour=c

cv2.drawContours(target,[closest_contour],-1,(0,0,255),3)
cv2.imshow(",", target)
cv2.waitKey()
        
        