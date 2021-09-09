from glob import glob
import os

rootPath = "/data/wd_DBnet_data/PSEnet/DB-TestData/"
imgList = glob(rootPath + "*/*.jpg")

for file in imgList:
    newname = file.replace(" (", '_')
    newname = newname.replace(")", '')
    os.rename(file, newname)