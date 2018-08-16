#!/usr/bin/env python

import cv2
import numpy as np
from matplotlib import pyplot as plt
import common 

def TestSURF(img):    
    plt.figure()

    hist_ori = cv2.calcHist([img],[0],None,[256],[0,256])
    plt.subplot(2,2,1)    
    plt.imshow(img,'gray')
    plt.title('original image')

   
    #show histogram
    img=common.prepro(img)
    plt.subplot(2,2,2)    
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    plt.plot(hist_ori,label='original')
    plt.plot(hist,label='equalized')
    plt.legend(loc=0)
    plt.title('histogram equalizing')


    
    plt.subplot(2,2,3)    
    plt.imshow(img,'gray')
    plt.title('prepro image')


    
    #SURF, Speeded-Up Robust Features, Hessian Threshold to 400
    surf = cv2.xfeatures2d.SURF_create(5000) 
    # Find keypoints and descriptors directly
    kp, des = surf.detectAndCompute(img,None)

    
    img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
    plt.subplot(2,2,4) 
    plt.imshow(img2)
    plt.title('Speeded-Up Robust Features')

    plt.show()

def TestHoughTransform(img):
    img=common.prepro(img)
    edges = cv2.Canny(img,50,150,apertureSize = 3)#argument1 input image
                                                  #argument2 minVal 50
                                                  #argument3 maxVal 150
                                                  #a4 Sobel kernel size, default=3
    print 'len edges',len(edges)
    lines = cv2.HoughLines(edges,2,2*np.pi/180,200)#parameter1 input image
                                                   #parameter2 pixel accuracies
                                                   #parameter3 radians accutacies
                                                   #p4 minimum length of drawed lines
    
    print len(lines)
    for line in lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
    
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    plt.figure()
    plt.imshow(img)
    plt.show()
    
def TestBlobDetection(img):
    img=common.prepro(img)
    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector()
 
    # Detect blobs.
    keypoints = detector.detect(img)
 
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
    # Show keypoints
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey(0)
    
    
def TestContourFind(img):
    img=common.prepro(img)
    ret,thresh = cv2.threshold(img,127,255,0)
    #first source image
    #second contour retrieval mode
    #third contour approximation method
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #To draw all the contours in an image
    cv2.drawContours(img, contours, -1, (0,255,0), 3)
#    cnt = contours[4]
#    cv2.drawContours(img, [cnt], 0, (0,255,0), 3)
    cv2.imshow("Keypoints", img)
    cv2.waitKey(0)
    
    
def adjust_gamma(img, gamma=1.0):
#    img=common.prepro(img)
#    invGamma = 1.0 / gamma
#    table = np.array([((i / 255.0) ** invGamma) * 255
#      for i in np.arange(0, 256)]).astype("uint8")
#
#      return cv2.LUT(img, table)
#    gamma = 0.5                                   # change the value here to get different result
#    adjusted = adjust_gamma(original, gamma=gamma)
#    cv2.putText(adjusted, "g={}".format(gamma), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
#    cv2.imshow("gammam image 1", adjusted)
#    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#    
#    # Setup SimpleBlobDetector parameters.
#    params = cv2.SimpleBlobDetector_Params()
# 
#    # Change thresholds
#    params.minThreshold = 10;
#    params.maxThreshold = 200;
# 
#    # Filter by Area.
#    params.filterByArea = True
#    params.minArea = 1500
# 
#    # Filter by Circularity
#    params.filterByCircularity = True
#    params.minCircularity = 0.1
# 
#    # Filter by Convexity
#    params.filterByConvexity = True
#    params.minConvexity = 0.87
# 
#    # Filter by Inertia
#    params.filterByInertia = True
#    params.minInertiaRatio = 0.01
# 
#    # Create a detector with the parameters
#    ver = (cv2.__version__).split('.')
#    if int(ver[0]) < 3 :
#        detector = cv2.SimpleBlobDetector(params)
#    else : 
#        detector = cv2.SimpleBlobDetector_create(params)
#        
#    #Detect blobs.
#    keypoints = detector.detect(img)
#    
#    
#    # Draw detected blobs as red circles.
#    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
#    # the size of the circle corresponds to the size of blob
#    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
#    # Show blobs
#    cv2.imshow("Keypoints", im_with_keypoints)
#    cv2.waitKey(0)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
