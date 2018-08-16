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




##thresholding
#ret,th1 = cv2.threshold(img,100,255,cv2.THRESH_BINARY)
#th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#            cv2.THRESH_BINARY,13,2)
#th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#            cv2.THRESH_BINARY,13,2)
#
#titles = ['Original Image', 'Global Thresholding (v = 127)',
#            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
#images = [img, th1, th2, th3]
#
#plt.figure()
#for i in xrange(4):
#    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
#    plt.title(titles[i])
#    plt.xticks([]),plt.yticks([])
#plt.show()
