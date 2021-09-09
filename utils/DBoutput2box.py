import os
from glob import glob
import numpy as np

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

if __name__ == "__main__":
    labelPath = '/data/wd_DBnet_data/Results/DIY_res18/labels/'
    DBoutput2box(labelPath)