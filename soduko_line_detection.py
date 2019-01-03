# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 23:05:55 2018

@author: amine bahlouli
"""

import sys
import cv2
import numpy as np

image = cv2.imread("soduku.jpg",0)

#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(image,100,170,apertureSize=3)

lines = cv2.HoughLines(edges,1,np.pi/180,240)

for line in lines:
    rho,theta= line[0]
    a=np.cos(theta)
    b= np.sin(theta)
    x0 = a*rho
    y0 =b*rho
    x1 = int(x0+1000*(-b))
    y1 = int(y0+1000*(a))
    x2 = int(x0-1000*(-b))
    y2 = int(y0-1000*(a))
    cv2.line(image,(x1,y1),(x2,y2),(255,0,0),2)
    
cv2.imshow("hough: ",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
    
    
    