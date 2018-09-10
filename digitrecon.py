# USAGE
# python recognize_digits.py
#https://www.pyimagesearch.com/done/
# python digitrecon.py --load_model 1 --save_weights cnn_weights.hdf5
# import the necessary packages
import argparse
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
from cnn.neural_network import CNN
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.preprocessing import image
import numpy as np
from PIL import Image


#-------------------------------------------------------------------------------
def load_cnn_data():
	# Parse the Arguments
	# ap = argparse.ArgumentParser()
	# ap.add_argument("-s", "--save_model", type=int, default=-1)
	# ap.add_argument("-l", "--load_model", type=int, default=-1)
	# ap.add_argument("-w", "--save_weights", type=str)
	#args = vars(ap.parse_args())

	# Defing and compile the SGD optimizer and CNN model
	#print('\n Compiling model...'+args["save_weights"])
	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	#clf = CNN.build(width=28, height=28, depth=1, total_classes=10, Saved_Weights_Path=args["save_weights"] if args["load_model"] > 0 else None)
	clf = CNN.build(width=28, height=28, depth=1, total_classes=10,
			Saved_Weights_Path="cnn_weights.hdf5" if 1 > 0 else None)
	clf.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])


	# If weights saved and argument load_model; Load the pre-trained model.
	if 1 < 0:
		print('\nTraining the Model...')
		clf.fit(train_img, train_labels, batch_size=b_size, epochs=num_epoch,verbose=verb)

		# Evaluate accuracy and loss function of test data
		print('Evaluating Accuracy and Loss Function...')
		loss, accuracy = clf.evaluate(test_img, test_labels, batch_size=128, verbose=1)
		print('Accuracy of Model: {:.2f}%'.format(accuracy * 100))
	return clf
#-------------------------------------------------------------------------------

def draw_contour(image, c, i):
	# compute the center of the contour area and draw a circle
	# representing the center
	M = cv2.moments(c)
	if M["m00"]==0:
		return image
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])

	# draw the countour number on the image
	cv2.putText(image, "#{}".format(i), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
		0.5, (0, 0, 255), 2)

	# return the image with the contour number drawn on it
	return image


def filter_contours(wraped):
	# threshold the warped image, then apply a series of morphological
	# operations to cleanup the thresholded image
    cv2.imshow("wraped 2 ", wraped)
    cv2.waitKey(0)
	thresh = cv2.threshold(wraped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 3))
	#kernel = np.ones((5,5),np.uint8)
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

	# find contours in the thresholded image, then initialize the
	# digit contours lists
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)

	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	digitCnts = []
	print(len(cnts))
	# loop over the digit area candidates
	for c in cnts:
		# compute the bounding box of the contour
		(x, y, w, h) = cv2.boundingRect(c)
		#roi = thresh[y:y + h, x:x + w]
		# if the contour is sufficiently large, it must be a digit
		#if (w >= 1 and w <= 45) and (h >= 5 and h <= 50):
		if (w >= 1 and w <= 30) and (h >= 14 and h <= 45):
			digitCnts.append(c)
	        #print(c)
	#cv2.imshow("output-Gowri rects ", output)
	#cv2.waitKey(0)
	# sort the contours from left-to-right, then initialize the
	# actual digits themselves
	digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
	digits = []
	return digitCnts, thresh
#--------
def predict_digit_countours(digitCnts,output,clf, thresh):
	# loop over each of the digits
	for c in digitCnts:
		# extract the digit ROI
		(x, y, w, h) = cv2.boundingRect(c)
		roi = thresh[y:y + h, x:x + w]
		# cv2.imshow("GK roi", roi)
		# cv2.waitKey(0)

		#compute the width and height of each of the 7 segments
		#we are going to examine
		#(roiH, roiW) = roi.shape
		# print("????????????????")
		# print(roiH)
		# print(roiW)
		constant = cv2.copyMakeBorder(roi,15,15,15,15,cv2.BORDER_CONSTANT,value=[0,0,0])
		cv2.imshow("GK new_image", constant)
		cv2.waitKey(0)
		# print("????????????????")
		# (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
		# dHC = int(roiH * 0.05)
		# define the set of 7 segments
		# segments = [
		# 	((0, 0), (w, dH)),	# top
		# 	((0, 0), (dW, h // 2)),	# top-left
		# 	((w - dW, 0), (w, h // 2)),	# top-right
		# 	((0, (h // 2) - dHC) , (w, (h // 2) + dHC)), # center
		# 	((0, h // 2), (dW, h)),	# bottom-left
		# 	((w - dW, h // 2), (w, h)),	# bottom-right
		# 	((0, h - dH), (w, h))	# bottom
		# ]
		# on = [0] * len(segments)

		#loop over the segments
		# for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
		# 	# extract the segment ROI, count the total number of
		# 	# thresholded pixels in the segment, and then compute
		# 	# the area of the segment
		# 	segROI = roi[yA:yB, xA:xB]
		# 	total = cv2.countNonZero(segROI)
		# 	area = (xB - xA) * (yB - yA)
		#
		# 	# if the total number of non-zero pixels is greater than
		# 	# 50% of the area, mark the segment as "on"
		# 	if total / float(area) > 0.5:
		# 		on[i]= 1

		# lookup the digit and draw it on the image
		digit = 1;#DIGITS_LOOKUP[tuple(on)]
		#print("DIGITS_LOOKUP[tuple(on)]")
		# print(".............................")
		# print(roi.dtype)
		# print(type(roi))
		# print(roi.shape)
		#mnist_data1 = roi.reshape(1,1, 28, 28)

		test_image=cv2.resize(constant, (28,28),interpolation=cv2.INTER_CUBIC)
		# print(test_image.dtype)
		# print(type(test_image))
		# print(test_image.shape)
		# cv2.imshow("sample",test_image)
		# cv2.waitKey(0)
		# print("//.............................//")
		# Predict the label of digit using CNN.

		probs = clf.predict(test_image.reshape(1,1,28,28))
		prediction = probs.argmax(axis=1)
		digit = prediction[0]
		#digits.append(digit)
		# cv2.imshow("Output is ", output)
		# cv2.waitKey(0)
		cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
		cv2.putText(output, str(digit), (x , y + 5 ),
		cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 0), 2)
		cv2.imshow("output-Gowri 1 ", output)
		cv2.waitKey(0)
	# display the digits
	#print(digits)
	cv2.imshow("Input", image1)
	cv2.imshow("Output", output)
	cv2.waitKey(0)


def filter_Image(image1):
	# pre-process the image by resizing it, converting it to
	# graycale, blurring it, and computing an edge map
	#image1 = imutils.resize(image1)
	image1 = imutils.resize(image1, height=490)
	#image1 = imutils.resize(image1, height=465)
	gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(blurred, 40, 200, 255)

	# cv2.imshow("gray", gray)
	# cv2.imshow("edged", edged)
	# cv2.waitKey(0)

	# find contours in the edge map, then sort them by their
	# size in descending order
	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
	displayCnt = None
	rect_count=[]
	# loop over the contours
	for  i,c in enumerate(cnts):
		# draw_contour(image1,c,i )
		# cv2.imshow("display",image1)
		# cv2.waitKey()
		# approximate the contour
		peri = cv2.arcLength(c, True)
		#print("peri")
		#print(peri)
		approx = cv2.approxPolyDP(c, 0.05  * peri, True)
		# draw_contour(image1,c,i, peri)
		# cv2.imshow("display",image1)
		# cv2.waitKey()
		# if the contour has four vertices, then we have found
		# the thermostat display
		#if len(approx) == 4:
		displayCnt = approx
		#break
		x,y,w,h = cv2.boundingRect(c)
		area = w * h
		#if len(approx) == 4:
		# print(peri, x,y,w,h,area,len(approx), i)
		# print(i,len(approx))
		# print(approx)

		#displayCnt = approx
		#draw_contour(image1,c,(i, x,y,w,h,area) )
		# cv2.imshow("display",image1)

		#rect_count.append(displayCnt)
		#break;
		# if len(approx) == 4:
		# 	displayCnt = approx
		#cv2.drawContours(image1, c, -1, (0,255,0), 3)
		#cv2.imshow("display 2 ",image1)
		#cv2.waitKey()
		break
	cv2.waitKey()
	wraped = four_point_transform(gray, displayCnt.reshape(4, 2))
	output = four_point_transform(image1, displayCnt.reshape(4, 2))
	cv2.imshow("output ", output)
	cv2.waitKey(0)
	digitCnts, thresh = filter_contours(wraped)
	return digitCnts, thresh, output


# ----------------------Def OVer--------------------

#-----------------------Main Program----------------
# load the example image
#image1 = cv2.imread("example.jpg")
#image1 = cv2.imread("photo_3.1.jpg")
# image1 = cv2.imread("img/photo_2.jpg")
# image1 = cv2.imread("img/mom_chq_amt.png")
# image1 = cv2.imread("img/sbh_old_m_chq1.png")
image1 = cv2.imread("chinchq.jpg")
# extract the thermostat display, apply a perspective transform
# to it
clf=load_cnn_data()
digitCnts, thresh, output = filter_Image(image1)
predict_digit_countours(digitCnts, output, clf, thresh)
displayCnt=filter_Image(image1, gray)
