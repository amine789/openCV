# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 00:46:35 2018

@author: amine bahlouli
"""

import cv2

import numpy as np

image = cv2.imread("WaldoBeach.jpg",0)
cv2.imshow("where is waldo3",image)
cv2.waitKey(0)



template = cv2.imread("waldo.jpg",0)
cv2.imshow(",",template)
cv2.waitKey(0)
cv2.destroyAllWindows()

result = cv2.matchTemplate(image,template,cv2.TM_CCOEFF)
min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(result)

#creating the box

top_left = max_loc
bottom_right = (top_left[0] + 50, top_left[1]+50)
cv2.rectangle(image,top_left,bottom_right,(0,0,255),5)

cv2.imshow("where is waldo", image)
cv2.waitKey(0)
cv2.destroyAllWindows()