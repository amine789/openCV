# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 01:07:00 2018

@author: amine bahlouli
"""

import cv2
import numpy as np

image = cv2.imread("shapes.jpg")
cv2.imshow("0 - original image", image)
cv2.waitKey(0)

# create a black image

blank_image = np.zeros((image.shape[0], image.shape[1],3))

original_image = image

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#canny edge
edged = cv2.Canny(gray, 50,200)
cv2.imshow("canny edge",edged)
cv2.waitKey(0)

#find countours
_,contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
print("number of contours: ",len(contours))

cv2.drawContours(blank_image, contours,-1,(0,255,0),3)
cv2.imshow("all contours over blank image",blank_image)
cv2.waitKey(0)

cv2.drawContours(image,contours,-1,(0,255,0),3)
cv2.imshow("all contours over original image", image)
cv2.waitKey(0)

cv2.destroyAllWindows()


def get_contour_areas(contours):
    all_areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)
    return all_areas

print(get_contour_areas(contours))
sorted_contours = sorted(contours, key=cv2.contourArea,reverse=True)
print("contours area after sorting ", get_contour_areas(sorted_contours))

for c in sorted_contours:
    cv2.drawContours(original_image,[c],-1,(255,0,0),3)
    cv2.waitKey(0)
    cv2.imshow("contour by area", original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

