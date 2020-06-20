import cv2
import numpy as np

# Main Image
img = cv2.imread("./testimages/test7.png", 1)


def splitFunc(img):
    # Corner of Sudoku
    def getcornercoordinates(img):
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurredimg = cv2.GaussianBlur(grayimg, (3, 3), 0)
        threshimg = cv2.adaptiveThreshold(blurredimg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 12)
        cannyimg = cv2.bitwise_not(threshimg)
        contours, hierarchy = cv2.findContours(cannyimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        ci = 0
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                ci = i
        cnt = contours[ci]
        cnt = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(cnt) != 4:
            print("Image cannot be loaded Properly")
            print(cnt)
            exit(0)
        # cv2.drawContours(img,cnt,-1,(255,0,0),5)
        # cv2.imshow("Original Image", img)
        # cv2.imshow("Canny Image", cannyimg)
        # cv2.imshow("Black Image", blackimg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return cnt

    corners = getcornercoordinates(img)

    def produceImage(img, corners):
        w, h = (512, 512)
        cv2.circle(img, tuple(corners[2][0]), 3, (0, 0, 255), -1)
        skewedvertices = np.float32([corners[0][0], corners[3][0], corners[1][0], corners[2][0]])
        newvertices = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        M = cv2.getPerspectiveTransform(skewedvertices, newvertices)
        newimg = cv2.warpPerspective(img, M, (512, 512))
        # cv2.imshow("IMAGE", img)
        # cv2.imshow("New Image", newimg)
        # cv2.waitKey(0)
        return newimg

    squareimage = produceImage(img, corners)

    def getCellImages(img, squareimage):
        graysquareimage = cv2.cvtColor(squareimage, cv2.COLOR_BGR2GRAY)
        threshgrayimage = cv2.adaptiveThreshold(graysquareimage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV,
                                                9, 5)
        kernel = np.ones((2, 2), np.uint8)
        erodedimg = cv2.erode(threshgrayimage, kernel=kernel, iterations=1)
        images = {}
        coloredimages = {}
        countrowwise = 1
        w, h = graysquareimage.shape
        x, y = h // 9, w // 9

        for i in range(1, 10):
            for j in range(1, 10):
                images[str(countrowwise)] = (erodedimg[x * (i - 1):x * (i), y * (j - 1):y * (j)])
                coloredimages[str(countrowwise)] = (squareimage[x * (i - 1):x * (i), y * (j - 1):y * (j)])
                countrowwise += 1
        print(len(images))
        # cv2.imshow("GRAY SQUARED CROPPED IMAGE",erodedimg)
        # cv2.imshow("FIRST",coloredimages["4"])
        # cv2.waitKey(0)
        return images,coloredimages

    cellImages,coloredcellImages = getCellImages(img, squareimage)

    def getDigitImagesArray():
        digitImages = []
        count = 0
        for i in cellImages:
            if (list(((cellImages[i])[16:42, 22:48]).ravel())).count(255) > 0.025 * len(
                    ((cellImages[i])[10:49, 14:49]).ravel()):
                digitImages.append(i)
                count += 1
        print("THe Count is ", count)
        # print(digitImages)
        # cv2.imshow("TESTING",cellImages["78"][16:42,22:48])
        # cv2.imshow("TO CHECk",cellImages["78"])
        # cv2.imshow("original Image",img)
        # cv2.waitKey(0)
        return digitImages

    digitImages = getDigitImagesArray()
    return (digitImages,cellImages,coloredcellImages)
