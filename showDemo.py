#Using cmd call "demo.py'
#Show the demo result of pic
# import pydevd_pycharm
# pydevd_pycharm.settrace('192.168.2.134', port=10010, stdoutToServer=True, stderrToServer=True)

import os
from glob import glob
import torch
from shutil import move
import numpy as np
import cv2

def DBoutput2box(labelPath):
    #labelPath = '/data/wd_DBnet_data/Results/DIY_myMV3/labels/'
    labelList = glob(labelPath + "*.txt")

    for file in labelList:
        with open(file, 'r') as fin:
            lines = fin.readlines()
        with open(file, 'w') as fout:
            for line in lines:
                line = line.replace('\n', '')
                line = line.replace('\r', '')
                params = line.split(',')
                if len(params) > 0:
                    x = [int(params[i]) for i in range(0, len(params)-1, 2)]
                    y = [int(params[i]) for i in range(1, len(params)-1, 2)]
                    x = np.array(x)
                    y = np.array(y)

                    xleft = np.min(x)
                    xright = np.max(x)
                    yup = np.min(y)
                    ybottom = np.max(y)

                    outputLine = "%d,%d,%d,%d,%d,%d,%d,%d\n" % (xleft, yup, xright, yup, xright, ybottom, xleft, ybottom)
                    fout.write(outputLine)

def icdar15_mv3large():
    imgList = ["img_100.jpg", "img_200.jpg", "img_300.jpg", "img_400.jpg", "img_500.jpg"]
    rootPath = "/data/wd_DBnet_data/origin_data/datasets/icdar2015/test_images/"
    exp = "experiments/seg_detector/ic15_mobilenetv3_large_thre.yaml"
    modelPath = "/data/wd_DBnet_data/trainedModels/mine_mv3large_ic15_final"

    for img in imgList:
        os.system("python3 demo.py --exp " + exp + " --image_path " + rootPath + img +
                  " --polygon --box_thresh 0.5 --visualize --resume " + modelPath)
        print("\t*** Figure " + img + ": Done ")

def icdar15_res18():
    imgList = ["img_100.jpg", "img_200.jpg", "img_300.jpg", "img_400.jpg", "img_500.jpg"]
    rootPath = "/data/wd_DBnet_data/origin_data/datasets/icdar2015/test_images/"
    exp = "experiments/seg_detector/ic15_resnet18_deform_thre.yaml"
    modelPath = "/data/wd_DBnet_data/trainedModels/mine_res18_ic15_final"

    for img in imgList:
        os.system("python3 demo.py --exp " + exp + " --image_path " + rootPath + img +
                  " --polygon --box_thresh 0.5 --visualize --resume " + modelPath)
        print("\t*** Figure " + img + ": Done ")

def td500_res18():
    print("TD500 Res18:")
    imgList = ["IMG_0172.JPG", "IMG_0485.JPG", "IMG_0711.JPG", "IMG_0833.JPG", "IMG_1940.JPG"]
    rootPath = "/data/wd_DBnet_data/origin_data/datasets/TD_TR/TD500/test_images/"
    exp = "experiments/seg_detector/td500_resnet18_deform_thre.yaml"
    modelPath = "/data/wd_DBnet_data/trainedModels/mine_res18_td500_final"

    for img in imgList:
        os.system("/root/anaconda2/envs/wd_DBnet_pytorch/bin/python3.7 demo.py --exp " + exp + " --image_path " + rootPath + img +
                  " --polygon --box_thresh 0.5 --visualize --resume " + modelPath)
        print("\t*** Figure " + img + ": Done ")

def diy_myMV3(modelPath, resultPath, box_thresh = 0.55, is_largerInput=False, ):
    print("DIY myMV3:")
    rootPath = "/data/wd_DBnet_data/PSEnet/DB_TestData2/"
    #modelPath = '/data/wd_DBnet_data/recorders/outputs/workspace/mv3_mixup_1500epochs_resume/model/model_epoch_59_minibatch_72000'
    exp = "/home/daiwang/code/DB/experiments/seg_detector/mixup_mv3net_large_thre.yaml"
    imgList = glob(rootPath + "*/*.jpg")
    for img in imgList:
        if is_largerInput:
            rawImg = cv2.imread(img)
            [h, w, _] = rawImg.shape
            shortSide = min(h, w)
            if shortSide > 1472:
                os.system(
                    "/root/anaconda2/envs/wd_DBnet_pytorch/bin/python3.7 demo.py --exp " + exp + " --image_path " + img +
                    " --polygon --visualize --resume " + modelPath +
                    " --result_dir " + resultPath +
                    " --box_thresh " + str(box_thresh) +
                    " --image_short_side 1472")
            else:
                os.system(
                    "/root/anaconda2/envs/wd_DBnet_pytorch/bin/python3.7 demo.py --exp " + exp + " --image_path " + img +
                    " --polygon --visualize --resume " + modelPath +
                    " --result_dir " + resultPath +
                    " --box_thresh " + str(box_thresh))
        else:
            os.system("/root/anaconda2/envs/wd_DBnet_pytorch/bin/python3.7 demo.py --exp " + exp + " --image_path " + img +
                      " --polygon --visualize --resume " + modelPath +
                      " --result_dir " + resultPath +
                      " --box_thresh " + str(box_thresh))
        print("\t*** Figure " + img.split('/')[-1] + ": Done ")

def diy_myMV3_Straight(modelPath, resultPath, box_thresh = 0.55, is_largerInput=False, ):
    print("DIY myMV3 with Straight Input:")
    rootPath = "/data/wd_DBnet_data/PSEnet/DB_TestData2_Straight/"
    #modelPath = '/data/wd_DBnet_data/recorders/outputs/workspace/mv3_mixup_1500epochs_resume/model/model_epoch_59_minibatch_72000'
    exp = "/home/daiwang/code/DB/experiments/seg_detector/mixup_mv3net_large_thre.yaml"
    imgList = glob(rootPath + "*/*.jpg")
    for img in imgList:
        if is_largerInput:
            rawImg = cv2.imread(img)
            [h, w, _] = rawImg.shape
            shortSide = min(h, w)
            if shortSide > 1472:
                os.system(
                    "/root/anaconda2/envs/wd_DBnet_pytorch/bin/python3.7 demo2.py --exp " + exp + " --image_path " + img +
                    " --polygon --visualize --resume " + modelPath +
                    " --result_dir " + resultPath +
                    " --box_thresh " + str(box_thresh) +
                    " --image_short_side 1472")
            else:
                os.system(
                    "/root/anaconda2/envs/wd_DBnet_pytorch/bin/python3.7 demo2.py --exp " + exp + " --image_path " + img +
                    " --polygon --visualize --resume " + modelPath +
                    " --result_dir " + resultPath +
                    " --box_thresh " + str(box_thresh))
        else:
            os.system("/root/anaconda2/envs/wd_DBnet_pytorch/bin/python3.7 demo2.py --exp " + exp + " --image_path " + img +
                      " --polygon --visualize --resume " + modelPath +
                      " --result_dir " + resultPath +
                      " --box_thresh " + str(box_thresh))
        print("\t*** Figure " + img.split('/')[-1] + ": Done ")


def diy_res18(modelPath, resultPath, box_thresh = 0.55):
    print("DIY res18:")
    rootPath = "/data/wd_DBnet_data/PSEnet/DB-TestData/"
    #modelPath = '/data/wd_DBnet_data/recorders/outputs/workspace/mv3_mixup_1500epochs_resume/model/model_epoch_59_minibatch_72000'
    exp = "/home/daiwang/code/DB/experiments/seg_detector/ic15_resnet18_deform_thre.yaml"
    imgList = glob(rootPath + "*/*.jpg")
    for img in imgList:
        os.system("/root/anaconda2/envs/wd_DBnet_pytorch/bin/python3.7 demo.py --exp " + exp + " --image_path " + img +
                  " --polygon --box_thresh 0.5 --visualize --resume " + modelPath +
                  " --result_dir " + resultPath + " --box_thresh " + str(box_thresh))
        print("\t*** Figure " + img.split('/')[-1] + ": Done ***")

def diy_res18_withSE(modelPath, resultPath, box_thresh = 0.55):
    print("DIY res18_withSE:")
    rootPath = "/data/wd_DBnet_data/PSEnet/DB-TestData/"
    #modelPath = '/data/wd_DBnet_data/recorders/outputs/workspace/mv3_mixup_1500epochs_resume/model/model_epoch_59_minibatch_72000'
    exp = "/home/daiwang/code/DB/experiments/seg_detector/ic15_resnet18_deform_thre.yaml"
    imgList = glob(rootPath + "*/*.jpg")
    for img in imgList:
        os.system("/root/anaconda2/envs/wd_DBnet_pytorch/bin/python3.7 demo.py --exp " + exp + " --image_path " + img +
                  " --polygon --box_thresh 0.5 --visualize --resume " + modelPath +
                  " --result_dir " + resultPath + " --box_thresh " + str(box_thresh))
        print("\t*** Figure " + img.split('/')[-1] + ": Done ***")

def diy_MV3_withSE(modelPath, resultPath, box_thresh = 0.55):
    print("DIY MV3_withSE:")
    rootPath = "/data/wd_DBnet_data/PSEnet/DB-TestData/"
    #modelPath = '/data/wd_DBnet_data/recorders/outputs/workspace/mv3_mixup_1500epochs_resume/model/model_epoch_59_minibatch_72000'
    exp = "/home/daiwang/code/DB/experiments/seg_detector/ic15_resnet18_deform_thre.yaml"
    imgList = glob(rootPath + "*/*.jpg")
    for img in imgList:
        os.system("/root/anaconda2/envs/wd_DBnet_pytorch/bin/python3.7 demo.py --exp " + exp + " --image_path " + img +
                  " --polygon --visualize --resume " + modelPath +
                  " --result_dir " + resultPath + " --box_thresh " + str(box_thresh))
        print("\t*** Figure " + img.split('/')[-1] + ": Done ***")

def diy_MV3(modelPath, resultPath, box_thresh = 0.55):
    #经ICDAR 1200epochs训得
    print("DIY MV3:")
    rootPath = "/data/wd_DBnet_data/PSEnet/DB-TestData/"
    #modelPath = '/data/wd_DBnet_data/recorders/outputs/workspace/mv3_mixup_1500epochs_resume/model/model_epoch_59_minibatch_72000'
    exp = "/home/daiwang/code/DB/experiments/seg_detector/ic15_mobilenetv3_large_thre.yaml"
    imgList = glob(rootPath + "*/*.jpg")
    for img in imgList:
        os.system("/root/anaconda2/envs/wd_DBnet_pytorch/bin/python3.7 demo.py --exp " + exp + " --image_path " + img +
                  " --polygon --visualize --resume " + modelPath +
                  " --result_dir " + resultPath + " --box_thresh " + str(box_thresh))
        print("\t*** Figure " + img.split('/')[-1] + ": Done ***")

def ic15_myMV3(modelPath, resultPath, box_thresh):
    print("IC15 myMV3:")
    rootPath = "/data/wd_DBnet_data/origin_data/datasets/icdar2015/test_images/"
    #modelPath = '/data/wd_DBnet_data/recorders/outputs/workspace/mv3_mixup_1500epochs_resume/model/model_epoch_59_minibatch_72000'
    exp = "/home/daiwang/code/DB/experiments/seg_detector/mixup_mv3net_large_thre.yaml"
    imgList = glob(rootPath + "*.jpg")
    for img in imgList:
        os.system("/root/anaconda2/envs/wd_DBnet_pytorch/bin/python3.7 demo.py --exp " + exp + " --image_path " + img +
                  " --polygon --visualize --resume " + modelPath +
                  " --result_dir " + resultPath +
                  " --box_thresh " + str(box_thresh))
        print("\t*** Figure " + img.split('/')[-1] + ": Done ***")

def reorganizeFile(resultPath):
    imgList = glob(resultPath + "*.jpg")
    labelList = glob(resultPath + "*.txt")
    assert (len(imgList) == len(labelList))

    imgDst = resultPath + "imgs/"
    labelDst = resultPath + "labels/"
    if not os.path.exists(imgDst):
        os.mkdir(imgDst)
    if not os.path.exists(labelDst):
        os.mkdir(labelDst)

    for img in imgList:
        move(img, imgDst + img.split('/')[-1])
    for label in labelList:
        move(label, labelDst + label.split("/")[-1])

if __name__ == "__main__":
    #icdar15_mv3large()
    #icdar15_res18()
    #td500_res18()
    # PROCESS_NAME = 'diy_myMV3_99000_withoutOC_largerInput'
    PROCESS_NAME = "diy_myMV3_99000_withoutOC_largerInput_Straight"
    import time
    tic = time.time()
    if PROCESS_NAME == "diy_myMV3_72000":
        modelPath = '/data/wd_DBnet_data/recorders/outputs/workspace/mv3_mixup_1500epochs_resume/model/model_epoch_59_minibatch_72000'
        resultPath = "/data/wd_DBnet_data/Results/DIY_myMV3_72000/"
        diy_myMV3(modelPath, resultPath)
        reorganizeFile(resultPath)
        DBoutput2box(resultPath + "labels/")

    elif PROCESS_NAME == "diy_myMV3_99000":
        # 带有开闭操作，会忽略细节
        modelPath = "/data/wd_DBnet_data/recorders/outputs/workspace/mv3_mixup_1500epochs_resume2/model/model_epoch_81_minibatch_99000"
        resultPath = "/data/wd_DBnet_data/Results/TestData2/" + PROCESS_NAME + "_withOC/"
        box_thresh = 0.4
        diy_myMV3(modelPath, resultPath, box_thresh)
        reorganizeFile(resultPath)
        DBoutput2box(resultPath + "labels/")

    elif PROCESS_NAME == "diy_myMV3_99000_withoutOC":
        # 不使用开闭操作，保留细节
        print("PROCESS: ", PROCESS_NAME)
        modelPath = "/data/wd_DBnet_data/recorders/outputs/workspace/mv3_mixup_1500epochs_resume2/model/model_epoch_81_minibatch_99000"
        resultPath = "/data/wd_DBnet_data/Results/TestData2/" + PROCESS_NAME + "/"
        box_thresh = 0.4
        diy_myMV3(modelPath, resultPath, box_thresh)
        reorganizeFile(resultPath)
        DBoutput2box(resultPath + "labels/")

    elif PROCESS_NAME == "diy_myMV3_99000_withoutOC_largerInput":
        # 对大尺寸图像减小缩放比
        print("PROCESS: ", PROCESS_NAME)
        modelPath = "/data/wd_DBnet_data/recorders/outputs/workspace/mv3_mixup_1500epochs_resume2/model/model_epoch_81_minibatch_99000"
        resultPath = "/data/wd_DBnet_data/Results/TestData2/" + PROCESS_NAME + "/"
        box_thresh = 0.4
        # 使用较大输入
        diy_myMV3(modelPath, resultPath, box_thresh, is_largerInput=True)
        reorganizeFile(resultPath)
        DBoutput2box(resultPath + "labels/")


    # diy_myMV3_99000_withoutOC_largerInput_Straight
    elif PROCESS_NAME == "diy_myMV3_99000_withoutOC_largerInput_Straight":
        # 对大尺寸图像减小缩放比
        print("PROCESS: ", PROCESS_NAME)
        modelPath = "/data/wd_DBnet_data/recorders/outputs/workspace/mv3_mixup_1500epochs_resume2/model/model_epoch_81_minibatch_99000"
        resultPath = "/data/wd_DBnet_data/Results/TestData2/" + PROCESS_NAME + "/"
        box_thresh = 0.4
        # 使用较大输入
        diy_myMV3_Straight(modelPath, resultPath, box_thresh, is_largerInput=True)
        reorganizeFile(resultPath)
        DBoutput2box(resultPath + "labels/")

    elif PROCESS_NAME == 'diy_res18':
        modelPath = '/data/wd_DBnet_data/trainedModels/mine_res18_ic15_final'
        resultPath = "/data/wd_DBnet_data/Results/DIY_res18/"
        box_thresh = 0.55
        diy_res18(modelPath, resultPath, box_thresh)
        reorganizeFile(resultPath)
        DBoutput2box(resultPath + "labels/")

    elif PROCESS_NAME == 'diy_MV3':
        modelPath = '/data/wd_DBnet_data/trainedModels/mine_mv3large_ic15_final'
        resultPath = "/data/wd_DBnet_data/Results/DIY_mv3/"
        box_thresh = 0.6
        diy_MV3(modelPath, resultPath, box_thresh)
        reorganizeFile(resultPath)
        DBoutput2box(resultPath + "labels/")

    elif PROCESS_NAME == "ic15_myMV3":
        modelPath = '/data/wd_DBnet_data/recorders/outputs/workspace/mv3_mixup_1500epochs_resume/model/model_epoch_59_minibatch_72000'
        resultPath = "/data/wd_DBnet_data/Results/ic15_myMV3/"
        box_thresh = 0.55
        ic15_myMV3(modelPath, resultPath, box_thresh)
        reorganizeFile(resultPath)
        DBoutput2box(resultPath + "labels/")

    toc = time.time()
    print("Total Time Cost: ", toc-tic)