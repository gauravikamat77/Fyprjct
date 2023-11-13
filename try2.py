import numpy as np
import cv2
from skimage.morphology import (dilation)
# Load the image, convert it to grayscale, and show it
image = cv2.imread("2.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", image)
#Compute the Laplacian of the image
lap = cv2.Laplacian(image, cv2.CV_64F)
lap = np.uint8(np.absolute(lap))
cv2.imshow("Laplacian", lap)
di = dilation(lap)
di = np.uint8(np.absolute(di))
cv2.imshow('dilation',di)
cv2.waitKey(0)
# Compute gradients along the X and Y axis, respectively
# sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0)
# sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1)

# sobelX = np.uint8(np.absolute(sobelX))
# sobelY = np.uint8(np.absolute(sobelY))
# cv2.imshow("Sobel X", sobelX)
# cv2.imshow("Sobel Y", sobelY)
# sobelCombined = cv2.bitwise_or(sobelX, sobelY)
# cv2.imshow("Sobel Combined", sobelCombined)
