#-*-coding:utf8-*-
# from PIL import Image
import cv2
import numpy as np
# from time import sleep

def separate_captcha(image_path):
	captcha0 = cv2.imread(image_path)
	captcha = cv2.resize(captcha0,(400,150))
	image = cv2.cvtColor(1-captcha,cv2.COLOR_BGR2GRAY)

	(_,thresh) = cv2.threshold(image,70,255,cv2.THRESH_BINARY)

	erode = cv2.erode(thresh,None)
	dilate = cv2.dilate(erode,None,iterations=4)

	#画出四个外接矩形,成功!
	(contour, _) = cv2.findContours(dilate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	#r = cv2.boundingRect(contour[0])
	x0,y0,w0,h0 = cv2.boundingRect(contour[0])
	x1,y1,w1,h1 = cv2.boundingRect(contour[1])
	x2,y2,w2,h2 = cv2.boundingRect(contour[2])
	x3,y3,w3,h3 = cv2.boundingRect(contour[3])
	cv2.rectangle(captcha,(x0,y0),(x0+w0,y0+h0),(255,0,0),thickness=2)
	cv2.rectangle(captcha,(x1,y1),(x1+w1,y1+h1),(255,0,0),thickness=2)
	cv2.rectangle(captcha,(x2,y2),(x2+w2,y2+h2),(255,0,0),thickness=2)
	cv2.rectangle(captcha,(x3,y3),(x3+w3,y3+h3),(255,0,0),thickness=2)

	# cv2.imwrite('roi.jpeg', captcha[y2:y2+h2,x2:x2+w2])

	return captcha[y0:y0+h0,x0:x0+w0],captcha[y1:y1+h1,x1:x1+w1],\
	captcha[y2:y2+h2,x2:x2+w2],captcha[y3:y3+h3,x3:x3+w3]
	# #最小矩形
	# (contour, _) = cv2.findContours(dilate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	# c = sorted(contour,key=cv2.contourArea, reverse=True)[2]
	# rect = cv2.minAreaRect(c)
	# box = np.int0(cv2.cv.BoxPoints(rect))
	# cv2.drawContours(captcha,[box],-1,(255,0,0),2)


	# # 全部曲线轮廓
	# (contour, _) = cv2.findContours(dilate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	# cv2.drawContours(captcha,contour,-1,(255,0,0),2)

	# # #轮廓近似(不实用)(注意dreawContour中轮廓参数加[]!!!!!!!!!!!!!!!!)
	# (contour, _) = cv2.findContours(dilate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	# epsilon = 0.06*cv2.arcLength(contour[0],True)
	# approx = cv2.approxPolyDP(contour[0],epsilon,True)
	# cv2.drawContours(captcha,[approx],-1,(255,0,0),2)


	# # 凸包(貌似可用但不是均匀形状)
	# (contour, _) = cv2.findContours(dilate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	# hull = cv2.convexHull(contour[0])
	# cv2.drawContours(captcha,[hull],-1,(255,0,0),2)


	#外接矩形
	# (contour, _) = cv2.findContours(dilate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	# x,y,w,h = cv2.boundingRect(contour[3])
	# cv2.rectangle(captcha,(x,y),(x+w,y+h),(255,0,0),thickness=2)



	#print r[0],r[1],r[2],r[3]
	# print x0,y0,w0,h0
	#print np.shape(image)
	#cv2.imwrite('roi.jpeg', captcha[r[0]:r[0]+r[2], r[1]:r[1]+r[3]])


	#box = (x0,y0,x0+w0,y0+h0)
	# region = captcha0.crop(box)
	# region = region.transpose(Image.ROTATE_180)
	# captcha.paste(region,box)
	#cap = cv2.cv.LoadImage('yzm.jpeg',cv2.cv.CV_LOAD_IMAGE_COLOR)
	# cv2.cv.SetImageROI(cap,(x0,y0,x0+w0,y0+h0))
	#cv2.cv.SetImageROI(cap,(0,0,15,15))



	# cv2.imshow('image',image)
	# cv2.imshow('captcha',dilate)
	# cv2.imshow('captcha',captcha)
	# cv2.waitKey()
