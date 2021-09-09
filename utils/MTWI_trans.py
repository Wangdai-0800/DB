import numpy as np
import os
from shutil import move
import cv2

imgPath = '/data/wd_DBnet_data/origin_data/datasets/MTWI/images/'
gtPath = '/data/wd_DBnet_data/origin_data/datasets/MTWI/gt/'
trainImgPath = '/data/wd_DBnet_data/origin_data/datasets/MTWI/train_images/'
trainGtPath = '/data/wd_DBnet_data/origin_data/datasets/MTWI/train_gts/'
trainListPath = '/data/wd_DBnet_data/origin_data/datasets/MTWI/train_list.txt'
testImgPath = '/data/wd_DBnet_data/origin_data/datasets/MTWI/test_images/'
testGtPath = '/data/wd_DBnet_data/origin_data/datasets/MTWI/test_gts/'
testListPath = '/data/wd_DBnet_data/origin_data/datasets/MTWI/test_list.txt'

allImg = os.listdir(imgPath)
imgNum = len(allImg)

cutRatio = 0.95
trainSampleNum = int(np.round(imgNum*cutRatio))

print('\n\n**************************')
print("Preparing Train Data")
for i in range(trainSampleNum):
    curName = allImg[i]
    print("Processing ", curName)
    txtName = curName.split('.')[0] + '.txt'
    srcPath = imgPath + curName
    img = cv2.imread(srcPath)
    [maxHeight, maxWidth, _] = img.shape
    print('(%d, %d)' % (maxHeight, maxWidth))
    move(srcPath, trainImgPath)
    with open(gtPath + txtName, 'r') as fin:
        lines = fin.readlines()
        recNum = len(lines)
    with open(trainGtPath + curName + '.txt', 'w') as fout:
        for lineIndex in range(recNum):
            line = lines[lineIndex]
            args = line.split('\t')
            assert(len(args) == 9)
            for j in range(len(args)-1):
                num = int(args[j])
                if num < 0:
                    print("One pos < 0 detected: ", args[j])
                    args[j] = '0'
                elif j%2==0 and num > maxWidth:
                    print("One pos > maxWidth detected: ", args[j])
                    args[j] = str(maxWidth)
                elif j%2==1 and num > maxHeight:
                    print('One pos > maxHeight detected: ', args[j])
                    args[j] = str(maxHeight)
                else:
                    pass
            line = ','.join(args)
            fout.write(line)
            fout.flush()
    with open(trainListPath, 'a') as trainList:
        trainList.write(curName + '\n')

print('\n\n**************************')
print("Preparing Test Data")
for i in range(trainSampleNum, imgNum, 1):
    curName = allImg[i]
    print("Processing ", curName)
    txtName = curName.split('.')[0] + '.txt'
    srcPath = imgPath + curName
    img = cv2.imread(srcPath)
    [maxHeight, maxWidth, _] = img.shape
    print('(%d, %d)' % (maxHeight, maxWidth))
    move(srcPath, testImgPath)
    with open(gtPath + txtName, 'r') as fin:
        lines = fin.readlines()
        recNum = len(lines)
    with open(testGtPath + curName + '.txt', 'w') as fout:
        for lineIndex in range(recNum):
            line = lines[lineIndex]
            args = line.split('\t')
            assert (len(args) == 9)
            for j in range(len(args) - 1):
                num = int(args[j])
                if num < 0:
                    print("One pos < 0 detected: ", args[j])
                    args[j] = '0'
                elif j % 2 == 0 and num > maxWidth:
                    print("One pos > maxWidth detected: ", args[j])
                    args[j] = str(maxWidth)
                elif j % 2 == 1 and num > maxHeight:
                    print('One pos > maxHeight detected: ', args[j])
                    args[j] = str(maxHeight)
                else:
                    pass
            line = ','.join(args)
            fout.write(line)
            fout.flush()
    with open(testListPath, 'a') as testList:
        testList.write(curName+'\n')