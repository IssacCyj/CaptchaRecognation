import cv2
import numpy as np

def image_preprocess(test_image0):
	#read image
	# test_image0 = cv2.imread(image_path)
	
	#resize to 28*28
	test_image1 = cv2.resize(test_image0,(28,28))
	#convert RGB to GRAY
	test_image2 = cv2.cvtColor(1-test_image1,cv2.COLOR_BGR2GRAY)
	#binaryzation
	(_,thresh) = cv2.threshold(test_image2,30,255,cv2.THRESH_BINARY)
	#erode and dilate to erase spots on the background
	erode = cv2.erode(thresh,None,iterations=1)
	dilate = cv2.dilate(erode,None,iterations=1)
	cv2.imshow('1',dilate)
	cv2.waitKey()
	#flatten
	test_image = dilate.flatten()
	#adjust the shape
	test_image = np.reshape(test_image, (1,28*28))
	return test_image