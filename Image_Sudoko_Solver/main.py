import sys
import cv2
import numpy as np

img = cv2.imread(sys.argv[-1],1)
def tryfunc():
    cv2.imshow("Original Image",img)

    cv2.waitKey(0)


try:
    tryfunc()
except:
    print("Image Not Loaded Properly.")