import cv2
import numpy as np


def SplitImage(img):
    grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(grayimg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,8)
    canny = cv2.Canny(thresh,100,200)
    blurredimg = cv2.GaussianBlur(canny,(3,3),0)
    kernel = np.ones((2,2),np.uint8)
    dilatedimg = cv2.erode(blurredimg,kernel=kernel,iterations=1)
    contours,_ = cv2.findContours(dilatedimg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    coordinates = []
    max_area = 0
    for i in contours:
        approx = cv2.approxPolyDP(i,0.01*cv2.arcLength(i,True),True)
        area=cv2.contourArea(approx)
        if area>max_area:
            max_area = area
            coordinates = approx.copy()
    print(coordinates)
    cv2.drawContours(img, coordinates, 0, (0, 255, 0), 4)
    cv2.imshow("Contoured IMAGE",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# img = cv2.imread("./testimages/test3.jpeg",1)
img = cv2.imread("./testimages/test2.jpeg",1)
# img = cv2.imread("./testimages/test1.jpg",1)
SplitImage(img)