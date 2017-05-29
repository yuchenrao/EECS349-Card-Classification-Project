import cv2
import numpy as np
from os import listdir
from random import randint

def transMat():
    return np.float32([[1,0,randint(-10,10)],[0,1,randint(-5,5)]])

def rotMat():
    return cv2.getRotationMatrix2D((40,30),randint(0,360),1)

directory = "test_data"
result = "test_result"
one_image = "test_data/3_club_1.jpg"
# one_image = "data/6_heart_19.jpg"
imageNameList = listdir(directory)

for fn in imageNameList:
    print fn
    img = cv2.imread(directory + "/" + fn)
    res = cv2.resize(img, (80, 60), interpolation=cv2.INTER_AREA)  # original resized
    cv2.imwrite(result + "/" + fn + "_original", res)

    compress = cv2.resize(img, (50, 90), interpolation=cv2.INTER_AREA)  # compressed
    cv2.imwrite(result + "/" + fn + "_compress", compress)
    stretch = cv2.resize(img, (90, 50), interpolation=cv2.INTER_AREA)  # stretched
    cv2.imwrite(result + "/" + fn + "_stretch", stretch)

    crop = res[5:55, 10:70]  # cropped image for zooming in
    zoom = cv2.resize(crop, (80, 60),
                      interpolation=cv2.INTER_AREA)  # scaled cropped image to original size to simulate zooming
    cv2.imwrite(result + "/" + fn + "_zoom", zoom)

    trans1 = cv2.warpAffine(res, transMat(), (80, 60))
    cv2.imwrite(result + "/" + fn + "_trans1", trans1)
    trans2 = cv2.warpAffine(res, transMat(), (80, 60))
    cv2.imwrite(result + "/" + fn + "_trans2", trans2)
    trans3 = cv2.warpAffine(res, transMat(), (80, 60))
    cv2.imwrite(result + "/" + fn + "_trans3", trans3)
    trans4 = cv2.warpAffine(res, transMat(), (80, 60))
    cv2.imwrite(result + "/" + fn + "_trans4", trans4)

    rot1 = cv2.warpAffine(res, rotMat(), (80, 60))
    cv2.imwrite(result + "/" + fn + "_rot1", rot1)
    rot2 = cv2.warpAffine(res, rotMat(), (80, 60))
    cv2.imwrite(result + "/" + fn + "_rot2", rot2)


# img = cv2.imread(one_image)
# res = cv2.resize(img, (80,60), interpolation=cv2.INTER_AREA) # original resized
#
# compress = cv2.resize(img, (50,90), interpolation=cv2.INTER_AREA) # compressed
# stretch = cv2.resize(img, (90,50), interpolation=cv2.INTER_AREA) # stretched
#
# crop = res[5:55,10:70] # cropped image for zooming in
# zoom = cv2.resize(crop, (80,60), interpolation=cv2.INTER_AREA) # scaled cropped image to original size to simulate zooming
#
# trans1 = cv2.warpAffine(res,transMat(),(80,60))
# trans2 = cv2.warpAffine(res,transMat(),(80,60))
# trans3 = cv2.warpAffine(res,transMat(),(80,60))
# trans4 = cv2.warpAffine(res,transMat(),(80,60))
#
# rot1 = cv2.warpAffine(res,rotMat(),(80,60))
# rot2 = cv2.warpAffine(res,rotMat(),(80,60))


# cv2.imshow('original',img)
# cv2.imshow('res',res)
# cv2.imshow('zoom',zoom)
# cv2.imshow('compress',compress)
# cv2.imshow('stretch',stretch)
# cv2.imshow('trans1',trans1)
# cv2.imshow('trans2',trans2)
# cv2.imshow('trans3',trans3)
# cv2.imshow('trans4',trans4)
# cv2.imshow('rot1',rot1)
# cv2.imshow('rot2',rot2)
# cv2.waitKey(0)
# cv2.imwrite(result + "/" + fn,res)



