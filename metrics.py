import numpy
import math
import cv2 as cv
import os
from skimage import data,filters,color


def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


hr_path="results/ref_0.jpg"
sr_path="results/pred_0.jpg"
bi_path="results/bicubic_0.jpg"

hr = cv.imread(hr_path)
sr = cv.imread(sr_path)
bicubic = cv.imread(bi_path)

print("EDSR:"+str(psnr(hr,sr)))
print("Bicubic:"+str(psnr(hr,bicubic)))


