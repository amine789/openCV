# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 22:16:05 2018

@author: amine bahlouli
"""

import numpy as np
import cv2

image = cv2.imread("digits.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
small = cv2.pyrDown(image)
cv2.imshow("Digits image ", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
x = np.array(cells)
print("the shape of cells array: ",str(x.shape))

train = x[:,:70].reshape(-1,400).astype(np.float32)
test = x[:,70:100].reshape(-1,400).astype(np.float32)

k= [0,1,2,3,4,5,6,7,8,9]

train_labels = np.repeat(k,350)[:,np.newaxis]
test_labels = np.repeat(k,150)[:,np.newaxis]

knn = cv2.ml.KNearest_create()
knn.train(train,cv2.ml.ROW_SAMPLE,train_labels)
ret,results,neighbors,distance= knn.findNearest(test,k=3)

matches = results==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*(100/results.size)
print("accuracy is : ",accuracy)




image = cv2.imread("numbers.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("image ", image)
cv2.imshow("gray",gray)
cv2.waitKey(0)

blurred = cv2.GaussianBlur(gray,(5,5),0)

cv2.imshow("blurred",blurred)
cv2.waitKey(0)

edged = cv2.Canny(blurred,30,150)
cv2.imshow("edged", edged)
cv2.waitKey(0)

def x_cord_contour(contour):
    # This function take a contour from findContours
    # it then outputs the x centroid coordinates
    
    M = cv2.moments(contour)
    return (int(M['m10']/M['m00']))
    
def square(not_square):
    BLACK = [0,0,0]
    img_dim = not_square.shape
    height = img_dim.shape[0]
    width = img_dim.shape[1]
    if (height==width):
        square == not_square
        return square
    else:
        doublesize = cv2.resize(not_square,(2*height,2*width), INTERPOLATION=cv2.INTER_CUBIC)
        height = height*2
        width = width*2
        if (height>width):
            pad = (height-width)/2
            doublesize_square = cv2.MakeBody(doublesize,0,0,pad,pad,cv2.BORDER_CONSTANT,value=BLACK)
        else:
            pad = (width-height)/2
            doublesize_square = cv2.copyMakeBorder(doublesize,pad,pad,0,0,\
                                                   cv2.BORDER_CONSTANT,value=BLACK)
    doublesize_square_dim = doublesize_square.shape
    return doublesize_square
            
#find contours
_,contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
contours = sorted(contours,key= x_cord_contour,reverse=False)

full_numbers = []

for i in contours:
    (x,y,w,h) = cv2.boundingRect(i)
    if w >=5 and h>=25:
        roi = blurred[y:y+h, x:x+w]
        ret,roi = cv2.threshold(roi,127,255,cv2.THRESH_BINARY_INV)
        squared = makeSquare(roi)
        final = resize_to_pixel(20,squared)
        cv2.imshow("final",)
        final_array = final.reshape((1,400))
        final_array = final_array.astype(np.float32)
        ret,result,neighbours,dist = knn.find_dist(final_array, k=1)
        number = str(int(float(result[0])))
        full_numner.append(number)
        cv2.rectangle(image, (x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(image,number,(x,y+155))
        cv2.putText(image, number, (x , y + 155),
            cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
        cv2.imshow("image", image)
        cv2.waitKey(0) 
cv2.destroyAllWindows()
print ("The number is: " + ''.join(full_number))


        
        
        



