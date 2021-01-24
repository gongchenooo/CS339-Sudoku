# import the necessary packages
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2

def find_puzzle(image, debug=True):
	# convert the image to grayscale and blur it slightly
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 颜色空间转换函数（需要转换的图片，格式）
	# 高斯滤波用于图像模糊处理：使图像像素值与邻域内其它像素值的变化程度减小，直观来看使图像上物体的边缘变模糊
	blurred = cv2.GaussianBlur(gray, (7, 7), 3)

	# apply adaptive thresholding and then invert the threshold map
	# 图像二值化
	thresh = cv2.adaptiveThreshold(blurred, 255,
		cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
	thresh = cv2.bitwise_not(thresh)

	# check to see if we are visualizing each step of the image
	# processing pipeline (in this case, thresholding)
	if debug:
		cv2.imshow("Puzzle Thresh", thresh)
		cv2.waitKey(0)

	# find contours in the thresholded image and sort them by size in
	# descending order
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE) # 只检测外轮廓；接受二值图而非灰度图；压缩水平垂直方向对角方向的元素，只保留终点坐标
	# cnts是一个list，每个元素都是图像的一个轮廓，即一个ndarray（轮廓上点的集合）
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True) # 排序，这样先检测大的轮廓

	# initialize a contour that corresponds to the puzzle outline
	puzzleCnt = None

	# loop over the contours
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True) # 计算轮廓周长
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # 将轮廓进行多边形近似

		# if our approximated contour has four points, then we can
		# assume we have found the outline of the puzzle
		if len(approx) == 4: # 找到四个轮廓点
			puzzleCnt = approx
			break

	# if the puzzle contour is empty then our script could not find
	# the outline of the sudoku puzzle so raise an error
	if puzzleCnt is None:
		raise Exception(("Could not find sudoku puzzle outline. "
			"Try debugging your thresholding and contour steps."))

	# check to see if we are visualizing the outline of the detected
	# sudoku puzzle
	if debug:
		# draw the contour of the puzzle on the image and then display
		# it to our screen for visualization/debugging purposes
		output = image.copy()
		cv2.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)
		cv2.imshow("Puzzle Outline", output)
		cv2.waitKey(0)

	# apply a four point perspective transform to both the original
	# image and grayscale image to obtain a top-down birds eye view
	# of the puzzle
	# 四点透视变换变为鸟瞰图
	puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
	warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))

	# check to see if we are visualizing the perspective transform
	if debug:
		# show the output warped image (again, for debugging purposes)
		cv2.imshow("Puzzle Transform", puzzle)
		cv2.imshow('Puzzle gray', warped)
		cv2.waitKey(0)

	# return a 2-tuple of puzzle in both RGB and grayscale
	return (puzzle, warped)

def extract_digit(cell, debug=False):
	# apply automatic thresholding to the cell and then clear any
	# connected borders that touch the border of the cell
	# thresh1 = cv2.GaussianBlur(cell, (3, 3), 3)
	thresh1 = cell
	thresh = cv2.threshold(thresh1, 140, 255,
		cv2.THRESH_BINARY_INV)[1] # 变为二值图，黑白二值反转
	# thresh_blur = cv2.GaussianBlur(thresh, (7, 7), 3)
	# thresh = clear_border(thresh1)  # 清除边界为1的值，边界都为0

	# check to see if we are visualizing the cell thresholding step
	if debug:
		cv2.imshow("cell", thresh1)
		cv2.imshow("Cell Thresh", thresh)
		cv2.waitKey(0)
	# thresh = modify_img2(thresh, thresh)
	# thresh = modify_img(thresh)
	# find contours in the thresholded cell
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	# if no contours were found than this is an empty cell
	if len(cnts) == 0:
		return None

	# otherwise, find the largest contour in the cell and create a
	# mask for the contour
	# c = max(cnts, key=cv2.contourArea)

	mask = np.zeros(thresh.shape, dtype="uint8")
	# cv2.drawContours(mask, [c], -1, 255, -1)
	cv2.drawContours(mask, cnts, -1, 255, -1)
	# compute the percentage of masked pixels relative to the total
	# area of the image
	(h, w) = thresh.shape
	percentFilled = cv2.countNonZero(mask) / float(w * h)

	# if less than 3% of the mask is filled then we are looking at
	# noise and can safely ignore the contour
	print(percentFilled)
	if percentFilled < 0.005:
		# print('<0.003')
		return None

	# apply the mask to the thresholded cell
	digit = cv2.bitwise_and(thresh, thresh, mask=mask)
	# digit_blur = cv2.GaussianBlur(thresh, (7, 7), 3)
	# digit = modify_img2(digit_blur, digit)
	digit = modify_img(digit)
	# check to see if we should visualize the masking step
	if debug:
		cv2.imshow("Digit", digit)
		cv2.waitKey(0)

	# return the digit to the calling function
	return digit


def modify_img(thresh):
	(h, w) = thresh.shape
	cnt_h = [0 for i in range(h)]
	cnt_w = [0 for i in range(w)]
	for i in range(0, h):
		for j in range(0, w):
			if thresh[i][j] != 0:
				cnt_h[i] += 1
				cnt_w[j] += 1

	h_min = h-1
	h_max = 0
	w_min = w-1
	w_max = 0
	for i in range(len(cnt_h)):
		if cnt_h[i] >= 1:
			h_min = min(h_min, i)
			h_max = max(h_max, i)
	for i in range(len(cnt_w)):
		if cnt_w[i] >= 1:
			w_min = min(w_min, i)
			w_max = max(w_max, i)
	# print(h_min, h_max, w_min, w_max)
	# cv2.imshow("old", thresh[h_min:h_max+1, w_min:w_max+1])
	# cv2.waitKey(0)
	length = max(h_max-h_min, w_max-w_min)

	h_max_new = (h_max + h_min + length + 1) // 2
	h_min_new = (h_max + h_min - length + 1) // 2
	w_max_new = (w_max + w_min + length + 1) // 2
	w_min_new = (w_max + w_min - length + 1) // 2

	# cv2.imshow("Thresh", new_thresh)
	# cv2.waitKey(0)
	# print(h_min_new, h_max_new, w_min_new, w_max_new)
	new_thresh = thresh[h_min_new:h_max_new + 1, w_min_new:w_max_new + 1]
	if (h_min_new < 0 or h_max_new >= h):
		if (h_min_new < 0):
			new_thresh = thresh[0:h_max_new + 1, w_min_new:w_max_new + 1]
			# print(h_min_new, h_max_new, w_min_new, w_max_new)
			# cv2.imshow("old", new_thresh)
			# cv2.waitKey(0)

			# print('old:', new_thresh.shape)
			new_thresh = np.pad(new_thresh, ((-h_min_new, 0), (0, 0)), 'constant', constant_values=0)
			# print('new:', new_thresh.shape)
		else:
			new_thresh = thresh[h_min_new:h, w_min_new:w_max_new + 1]
			# print(h_min_new, h_max_new, w_min_new, w_max_new)
			# cv2.imshow("old", new_thresh)
			# cv2.waitKey(0)
			# print('old:', new_thresh.shape)
			new_thresh = np.pad(new_thresh, ((0, h_max_new-h+1), (0, 0)), 'constant', constant_values=0)
			# print('new:', new_thresh.shape)
	if (w_min_new < 0 or w_max_new >= w):
		if (w_min_new < 0):
			new_thresh = thresh[h_min_new:h_max_new + 1, 0:w_max_new + 1]
			#print(h_min_new, h_max_new, w_min_new, w_max_new)
			# cv2.imshow("old", new_thresh)
			# cv2.waitKey(0)
			#print('old:', new_thresh.shape)
			new_thresh = np.pad(new_thresh, ((0, 0), (-w_min_new, 0)), 'constant', constant_values=0)
			#print('new:', new_thresh.shape)
		else:
			new_thresh = thresh[h_min_new:h_max_new + 1, w_min_new:w]
			#print(h_min_new, h_max_new, w_min_new, w_max_new)
			# cv2.imshow("old", new_thresh)
			# cv2.waitKey(0)
			#print('old:', new_thresh.shape)
			new_thresh = np.pad(new_thresh, ((0, 0), (0, w_max_new-w+1)), 'constant', constant_values=0)
			#print('new:', new_thresh.shape)
	# cv2.imshow("new", new_thresh)
	# cv2.waitKey(0)
	if (new_thresh.shape == (0,0)):
		# print(h_min_new, h_max_new, w_min_new, w_max_new)
		return thresh
		# cv2.imshow("", thresh)
		# cv2.waitKey(0)
	new_thresh = cv2.resize(new_thresh, (28, 28))
	# print(new_thresh.shape)
	return new_thresh


def modify_img2(thresh, img_original):
	(h, w) = thresh.shape
	cnt_h = [0 for i in range(h)]
	cnt_w = [0 for i in range(w)]
	for i in range(0, h):
		for j in range(0, w):
			if thresh[i][j] != 0:
				cnt_h[i] += 1
				cnt_w[j] += 1

	h_min = h-1
	h_max = 0
	w_min = w-1
	w_max = 0
	for i in range(len(cnt_h)):
		if cnt_h[i] >= 1:
			h_min = min(h_min, i)
			h_max = max(h_max, i)
	for i in range(len(cnt_w)):
		if cnt_w[i] >= 1:
			w_min = min(w_min, i)
			w_max = max(w_max, i)
	# print(h_min, h_max, w_min, w_max)
	# cv2.imshow("old", thresh[h_min:h_max+1, w_min:w_max+1])
	# cv2.waitKey(0)
	length = max(h_max-h_min, w_max-w_min)

	h_max_new = (h_max + h_min + length + 1) // 2
	h_min_new = (h_max + h_min - length + 1) // 2
	w_max_new = (w_max + w_min + length + 1) // 2
	w_min_new = (w_max + w_min - length + 1) // 2

	# cv2.imshow("Thresh", new_thresh)
	# cv2.waitKey(0)
	# print(h_min_new, h_max_new, w_min_new, w_max_new)
	new_thresh = img_original[h_min_new:h_max_new + 1, w_min_new:w_max_new + 1]
	if (h_min_new < 0 or h_max_new >= h):
		if (h_min_new < 0):
			new_thresh = img_original[0:h_max_new + 1, w_min_new:w_max_new + 1]
			# print(h_min_new, h_max_new, w_min_new, w_max_new)
			# cv2.imshow("old", new_thresh)
			# cv2.waitKey(0)

			# print('old:', new_thresh.shape)
			new_thresh = np.pad(new_thresh, ((-h_min_new, 0), (0, 0)), 'constant', constant_values=0)
			# print('new:', new_thresh.shape)
		else:
			new_thresh = img_original[h_min_new:h, w_min_new:w_max_new + 1]
			# print(h_min_new, h_max_new, w_min_new, w_max_new)
			# cv2.imshow("old", new_thresh)
			# cv2.waitKey(0)
			# print('old:', new_thresh.shape)
			new_thresh = np.pad(new_thresh, ((0, h_max_new-h+1), (0, 0)), 'constant', constant_values=0)
			# print('new:', new_thresh.shape)
	if (w_min_new < 0 or w_max_new >= w):
		if (w_min_new < 0):
			new_thresh = img_original[h_min_new:h_max_new + 1, 0:w_max_new + 1]
			#print(h_min_new, h_max_new, w_min_new, w_max_new)
			# cv2.imshow("old", new_thresh)
			# cv2.waitKey(0)
			#print('old:', new_thresh.shape)
			new_thresh = np.pad(new_thresh, ((0, 0), (-w_min_new, 0)), 'constant', constant_values=0)
			#print('new:', new_thresh.shape)
		else:
			new_thresh = img_original[h_min_new:h_max_new + 1, w_min_new:w]
			#print(h_min_new, h_max_new, w_min_new, w_max_new)
			# cv2.imshow("old", new_thresh)
			# cv2.waitKey(0)
			#print('old:', new_thresh.shape)
			new_thresh = np.pad(new_thresh, ((0, 0), (0, w_max_new-w+1)), 'constant', constant_values=0)
			#print('new:', new_thresh.shape)
	# cv2.imshow("new", new_thresh)
	# cv2.waitKey(0)
	if (new_thresh.shape == (0,0)):
		return img_original
		# cv2.imshow("", thresh)
		# cv2.waitKey(0)
	# print(new_thresh.shape)
	new_thresh = cv2.resize(new_thresh, (28, 28))
	return new_thresh