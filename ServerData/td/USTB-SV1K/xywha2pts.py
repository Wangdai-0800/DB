# -*- coding: utf8 -*-
import glob
import math

def xywha2pts(x, y, w, h, angle):
    # 矩形框中点(x0,y0)
    x0 = x + w / 2
    y0 = y + h / 2
    l = math.sqrt(pow(w / 2, 2) + pow(h / 2, 2))  # 即对角线的一半
    # angle小于0，逆时针转
    if angle < 0:
        a1 = -angle + math.atan(h / float(w))  # 旋转角度-对角线与底线所成的角度
        a2 = -angle - math.atan(h / float(w))  # 旋转角度+对角线与底线所成的角度
        pt1 = (x0 - l * math.cos(a2), y0 + l * math.sin(a2))
        pt2 = (x0 + l * math.cos(a1), y0 - l * math.sin(a1))
        pt3 = (x0 + l * math.cos(a2), y0 - l * math.sin(a2))  # x0+左下点旋转后在水平线上的投影, y0-左下点在垂直线上的投影，显然逆时针转时，左下点上一和左移了。
        pt4 = (x0 - l * math.cos(a1), y0 + l * math.sin(a1))
    else:
        a1 = angle + math.atan(h / float(w))
        a2 = angle - math.atan(h / float(w))
        pt1 = (x0 - l * math.cos(a1), y0 - l * math.sin(a1))
        pt2 = (x0 + l * math.cos(a2), y0 + l * math.sin(a2))
        pt3 = (x0 + l * math.cos(a1), y0 + l * math.sin(a1))
        pt4 = (x0 - l * math.cos(a2), y0 - l * math.sin(a2))
    return [pt1[0], pt1[1], pt2[0], pt2[1], pt3[0], pt3[1], pt4[0], pt4[1]]

gts = glob.glob('./gt/*.gt')
for gt in gts:
    tls = []
    with open(gt, 'r') as fp:
        tls = fp.readlines()
    with open(gt.replace('./gt/', './txt_gt/').replace('.gt', '.txt'), 'w') as pf:
        for tl in tls:
            nl = tl.replace(' ', '\t', 7)
            wha = nl.split('\t')
            pts = xywha2pts(int(wha[2]), int(wha[3]), int(wha[4]), int(wha[5]), float(wha[6]))
            for pt in pts:
                pf.write(str(int(round(pt)))+',')
            pf.write(wha[7][1:-4]+'\n')
