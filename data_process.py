# -*- coding: utf-8 -*-

import cv2 as cv
import os
import numpy as np
from PIL import Image
from skimage import data,filters,color
import matplotlib.pyplot as plt


from_path="/home/workplace/yz/works/EDSR-Keras-master/imgs/lr/"
data_path="/home/workplace/yz/works/MergeNet/imgs/sr"
label_path="/home/workplace/yz/works/MergeNet/imgs/mask"
hr_path="/home/workplace/yz/works/EDSR-Keras-master/imgs/hr/"
dest_path="/home/workplace/yz/works/MergeNet/imgs/hr"

def edge(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_RGB2GRAY)
    # xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0) #x
    # ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1) #y
    # edge_output = cv.Canny(xgrad, ygrad, 50, 150)
    edge_output = cv.Canny(gray, 50, 150)
    return edge_output

files=os.listdir(from_path)
index=1
for file in files:
    print('正在处理图像： %s' % index)
    img_path=from_path+file
    img_hrpath=hr_path+file
    print(img_path)
    img = cv.imread(img_path)
    img_hr = cv.imread(img_hrpath)
    hr_size = (480, 320)
    img_sr = cv.resize(img, hr_size, interpolation = cv.INTER_CUBIC)
    img_mask = edge(img_sr)
    cv.imwrite(data_path+'/'+file,img_sr)
    cv.imwrite(label_path+'/'+file,img_mask)
    cv.imwrite(dest_path+'/'+file,img_hr)
    print('处理成功： %s' % index)
    
    index += 1
