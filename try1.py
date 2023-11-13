import numpy as np
import cv2 as cv
from skimage.morphology import (dilation)
from matplotlib import pyplot as plt
img = cv.imread('3.webp', cv.IMREAD_GRAYSCALE)


assert img is not None, "file could not be read, check with os.path.exists()"

kernel = np.ones((5,5),np.float32)/25
dst = cv.filter2D(img,-1,kernel)
laplacian = cv.Laplacian(dst,cv.CV_64F)
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Gradiant Operation'), plt.xticks([]), plt.yticks([])
di = dilation(laplacian)
plt.subplot(2,2,3), plt.imshow(di, cmap= 'gray')
plt.title('Morphological Expansion'), plt.xticks([]), plt.yticks([])


threshold_value = 8
max_val = 255

ret, image = cv.threshold(di, threshold_value, max_val, cv.THRESH_BINARY_INV)
cv.imshow('InverseBinaryThresholding', image)

plt.show()
