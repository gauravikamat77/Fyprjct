
import numpy as np
import cv2 as cv
from skimage.morphology import (dilation)
from matplotlib import pyplot as plt
img = cv.imread('3.webp', cv.IMREAD_GRAYSCALE)

assert img is not None, "file could not be read, check with os.path.exists()"
kernel = np.ones((5,5),np.float32)/25
dst = cv.filter2D(img,-1,kernel)
edges = cv.Canny(dst,100,200)
plt.subplot(2,2,1),plt.imshow(edges,cmap = 'gray')
plt.title('Canny edge'), plt.xticks([]), plt.yticks([])
di = dilation(edges)
plt.subplot(2,2,2), plt.imshow(di, cmap= 'gray')
plt.title('Morphological Expansion'), plt.xticks([]), plt.yticks([])
threshold_value = 120
max_val = 255

ret, image = cv.threshold(di, threshold_value, max_val, cv.THRESH_BINARY_INV)
cv.imshow('InverseBinaryThresholding', image)
plt.show()