import numpy as np
import os
from shutil import move
import cv2

imgPath = '/data/wd_DBnet_data/origin_data/datasets/USTB1K/images/'
gtPath = '/data/wd_DBnet_data/origin_data/datasets/USTB1K/gt/'
trainImgPath = '/data/wd_DBnet_data/origin_data/datasets/USTB1K/train_images/'
trainGtPath = '/data/wd_DBnet_data/origin_data/datasets/USTB1K/train_gts/'
trainListPath = '/data/wd_DBnet_data/origin_data/datasets/USTB1K/train_list.txt'
testImgPath = '/data/wd_DBnet_data/origin_data/datasets/USTB1K/test_images/'
testGtPath = '/data/wd_DBnet_data/origin_data/datasets/USTB1K/test_gts/'
testListPath = '/data/wd_DBnet_data/origin_data/datasets/USTB1K/test_list.txt'

allImg = os.listdir(imgPath)
imgNum = len(allImg)

cutRatio = 0.9
trainSampleNum = int(np.round(imgNum*cutRatio))

print('\n\n**************************')
print("Preparing Train Data")
for i in range(trainSampleNum):
    curName = allImg[i]
    print("Processing ", curName)
    txtName = curName.split('.')[0] + '.txt'
    srcPath = imgPath + curName
    move(srcPath, trainImgPath)
    gtSrcPath = gtPath + txtName
    move(gtSrcPath, trainGtPath + curName + '.txt')
    with open(trainListPath, 'a') as trainList:
        trainList.write(curName + '\n')


print('\n\n**************************')
print("Preparing Test Data")
for i in range(trainSampleNum, imgNum, 1):
    curName = allImg[i]
    print("Processing ", curName)
    txtName = curName.split('.')[0] + '.txt'
    srcPath = imgPath + curName
    move(srcPath, testImgPath)
    gtSrcPath = gtPath + txtName
    move(gtSrcPath, testGtPath + curName + '.txt')
    with open(testListPath, 'a') as testList:
        testList.write(curName+'\n')
