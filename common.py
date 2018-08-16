#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 05 22:42:53 2018

@author: Sha
"""


import cv2
import numpy as np

def prepro(img):
    #OpenCV-Python Tutorials, Image Processing in OpenCV, Histograms in OpenCV 
    #adaptive histogram equalization
    clahe = cv2.createCLAHE()
    img = clahe.apply(img)
    
    #Bilateral Filtering, high performance but slow
    #img = cv2.bilateralFilter(img, 9,74,74)
    img = cv2.GaussianBlur(img,(5,5),0)
    return img


