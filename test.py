import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
from PIL import Image 

mask = cv2.imread("masked_img.png")
print(mask.shape)
th , im_th = cv2.threshold(mask , 0 , 255 , cv2.THRESH_BINARY)
print(im_th.shape)
cv2.imwrite("test.png" , im_th)