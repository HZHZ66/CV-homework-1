import cv2 as cv
from load_data import load_data
from comp_KR import comp_KR
from mosaic_global import mosaic_global
from mosaic_local_ori import mosaic_local_ori
import matplotlib.pyplot as plt

data_path = 'images/C/'
imfile1 = data_path + '1.jpg'
imfile2 = data_path + '2.jpg'
im1 = cv.imread(imfile1)
im2 = cv.imread(imfile2)
X1,X2 = load_data(im1,im2)#生成两行n列的数组，第一行为im中关键点坐标
print("Number of matches: ", X1.shape[1])

H, X1_ok, X2_ok,img3 = comp_KR(im1, im2, X1, X2)
cv.imwrite(data_path+'Ransac result.jpg', img3)
print("H =\n", H)
print("H normalize =\n", H/H[2,2])

mosaic = mosaic_global(im1, im2, H)
cv.imwrite(data_path+'mosaic_global.jpg', mosaic)
cv.imshow("mosaic_global", mosaic)
cv.waitKey(1)

mosaic = mosaic_local_ori(im1, im2, H, X1_ok, X2_ok)    
# cv.imwrite(data_path+'Bayesian Refinement of Feature Matches.jpg', img4)
cv.imwrite(data_path+'mosaic_REW.jpg', mosaic)
cv.imshow("mosaic_REW", mosaic)
cv.waitKey(0)
