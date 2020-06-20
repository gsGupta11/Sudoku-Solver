import imagesplitting
import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np

model_pickle = open("../knn.model", 'rb')
model = pickle.load(model_pickle)
model_pickle.close()

image = cv2.imread("./testimages/test5.png", 1)
toidentifyarr, matrixdic,colorcelldict = imagesplitting.splitFunc(image)

for i in toidentifyarr:
    img = colorcelldict[i][11:,12:]
    img = cv2.resize(img,(30,30))
    img = cv2.bitwise_not(img)
    img = cv2.dilate(img, np.ones((1,1), np.uint8),iterations=2)
    # img = cv2.erode(img, np.ones((1,1), np.uint8),iterations=1)
    img =cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lb=np.array([0,0,30])
    ub=np.array([80,80,255])
    img = cv2.inRange(img,lb,ub)
    cv2.imshow("ORIGINAL",img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # img = cv2.GaussianBlur(img,(5,5),0)
    # img = cv2.medianBlur(img,3,0)
    
    # _, img = cv2.threshold(img, 65, 255, cv2.THRESH_BINARY)
    print(str(model.predict([img.flatten()])[0]))
    cv2.imshow("i",img)

    cv2.waitKey(0)