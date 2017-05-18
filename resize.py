import numpy as np
import cv2
import matplotlib.pyplot as plt


input_img = cv2.imread("test.jpg",cv2.IMREAD_GRAYSCALE)
img = cv2.resize(input_img, (50, 50))
cv2.putText(input_img, "Disgust",(50,100), cv2.FONT_HERSHEY_SIMPLEX, 2, 10, 10)

plt.imshow(input_img,cmap = 'gray')

plt.show()
