import glob
import os
import cv2
print os.getcwd()
for file in glob.glob(os.getcwd()+'/data/*.jpg'):
	cv2.imwrite(file.replace('data','data_resize'),cv2.resize(cv2.imread(file,1),(800,600)))