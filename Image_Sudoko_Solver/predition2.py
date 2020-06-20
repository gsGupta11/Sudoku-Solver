import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import imagesplitting
import pickle
import cv2

img = cv2.imread("./testimages/test2.jpeg", 1)
toidentifyarr, matrixdic = imagesplitting.splitFunc(img)

model_final = open("../knn.model", 'rb')
model_to_use = pickle.load(model_final)
model_final.close()

import matplotlib.pyplot as plt

img1 = cv2.resize(matrixdic["9"][11:,12:], (30, 30))
kernel = np.ones((1,1),np.uint8)
img1 = cv2.dilate(img1,kernel=kernel,iterations=3)
con,_ = cv2.findContours(img1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
max_area =0
for i in range(len(con)):
    cnt = con[i]
    area = cv2.contourArea(cnt)
    if area > max_area:
        max_area = area
        ci = i
cnt = con[ci]
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
print(box)
x = []
y = []
for i in box:
    if i[0] not in x:
        x.append(i[0])
    if i[1] not in y:
        y.append(i[1])
newimg = img1[min(y):max(y), min(x):max(x)]
# newimg = img1[0:27,4:22]
# cv2.drawContours(img1,[box],0,(0,0,255),2)

img1 = cv2.bitwise_not(img1)
cv2.imshow("ABCD",img1)
newimg = cv2.resize(newimg,(30,30))
# newimg = cv2.bitwise_not(newimg)
# plt.show()
res = str(model_to_use.predict([newimg.flatten()])[0])
print(res)
cv2.waitKey(0)


