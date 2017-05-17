import numpy as np
import cv2
import matplotlib.pyplot as plt


input_img = cv2.imread("test.jpg",cv2.IMREAD_GRAYSCALE)
img = cv2.resize(input_img, (50, 50))

plt.imshow(img)

plt.show()
