# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 21:45:52 2018

@author: amine bahlouli
"""
import time
import cv2
import numpy as np

def sketch(image):
    #convert to gray scale
    img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #Clean up image using gaussian filter
    img_gray_blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    #Extract edges
    canny_edges = cv2.Canny(img_gray_blur,10,70)
    # do an invert binarize image
    ret, mask = cv2.threshold(canny_edges, 70,255,cv2.THRESH_BINARY_INV)
    return mask


cap = cv2.VideoCapture(0)
time.sleep(60)
while True:
    ret, frame = cap.read()
    cv2.imshow("Our live sktcher is: ", sketch(frame))
    if cv2.waitKey(1)==13:
        break

cv2.release()
cv2.destroyAllWindows()