import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import zoom


img = cv2.imread('test1.png',0)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
newimg = zoom(img, (150. / img.shape[0], 100. / img.shape[1]), 1)
edges = cv2.Canny(newimg,100,200)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
