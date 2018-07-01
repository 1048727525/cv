import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract as tess



def recognize_text(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV | cv.THRESH_TRIANGLE)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 3))
    bin1 = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (1, 5))
    bin2 = cv.morphologyEx(bin1, cv.MORPH_OPEN, kernel)


    cv.bitwise_not(bin2, bin2)
    cv.imshow("binary-image", bin2)
    textImage = Image.fromarray(bin2)
    text = tess.image_to_string(textImage)
    print("识别结果: %s"%text)



img1 = cv.imread("yzm.jpg")
cv.imshow("img1", img1)
recognize_text(img1)
c = cv.waitKey(0)
if c == 27:
    cv.destroyAllWindows()