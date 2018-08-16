import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.close('all')
import os,glob
os.chdir('C:\Users\masha\Dropbox\python')
#os.chdir('D:\Sha Matlab OpenCV\python')
import FeatureExtraction as fe

for filename in glob.glob("data\*.png"):
    if '_ref' not in filename:
        print filename
        img = cv2.imread(filename,0)
        #fe.TestSURF(img)
        #fe.TestHoughTransform(img)
        #fe.TestBlobDetection(img)
        fe.TestContourFind(img)
        #fe.adjust_gamma(img, gamma=1.0)
