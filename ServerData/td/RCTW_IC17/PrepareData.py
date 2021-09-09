#RCTW_IC17 anotation creater
#Make it identical with ICDAR15
import glob
import os
import random
from shutil import copy, move

trainImgPath = "./train_images/"
testImgPath = "./test_images/"
trainGtPath = "./train_gts/"
testGtPath = "./test_gts/"

#6个part全部读出来
allImgs = []
allGts = []
for i in range(6):
    curFolder = './part' + str(i+1) + '/'
    curImgList = glob.glob(curFolder + "*.jpg")
    curGtList = glob.glob(curFolder + "*.txt")
    assert(len(curImgList) == len(curGtList))

    allImgs.append(curImgList)
    allGts.append(curGtList)
allImgs = sum(allImgs, [])
allGts = sum(allGts, [])

#检校是否匹配，可考虑用排序替换
for i in range(0, len(allImgs), 10):
    assert(allImgs[i].split('.')[1] == allGts[i].split('.')[1])

#随机数据，是否需要？
'''random.seed(0)
tmpIndex = list(range(len(allImgs)))
random.shuffle(tmpIndex)
allImgs = [allImgs[_] for _ in tmpIndex]
allGts = [allGts[_] for _ in tmpIndex]'''

#7000个作为训练集，其余1034个作为测试集，当前数据不全
trainSamplesNum = 7000
for item in allImgs[0:trainSamplesNum]:
    move(item, trainImgPath)
for item in allImgs[trainSamplesNum: ]:
    move(item, testImgPath)

#将标记归为与ICDAR15一致
def generateGt(dstPath, gtfile):
    newGtPath = dstPath + gtfile.split('/')[-1].split('.')[0] + ".jpg.txt"
    with open(gtfile, 'r') as of:
        text = of.readlines()
    with open(newGtPath, 'w') as newf:
        for line in text:
            newline = line.replace('"', "")
            newf.write(newline)

for item in allGts[0:trainSamplesNum]:
    generateGt(trainGtPath, item)
for item in allGts[trainSamplesNum: ]:
    generateGt(testGtPath, item)

#生成train_list.txt和test_list.txt
with open('./train_list.txt', 'w') as train_list:
    for item in allImgs[0:trainSamplesNum]:
        train_list.write(item.split('/')[-1] + "\r\n")

with open('./test_list.txt', 'w') as test_list:
    for item in allImgs[trainSamplesNum: ]:
        test_list.write(item.split('/')[-1] + "\r\n")
