from Utilities import extract_digit
from Utilities import find_puzzle
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from Sudoku import Sudoku
import numpy as np
import imutils
import cv2
import tensorflow as tf
import copy
# construct the argument parser and parse the arguments

model_path = './model_parameters/Alex_28_new.h5'
image_path = 'test/2-5.jpg'
whether_debug = False

print("[INFO] loading digit classifier...")
model  = load_model(model_path, custom_objects={"lrn": tf.nn.local_response_normalization})

print("[INFO] processing image...")
image = cv2.imread(image_path)
image = imutils.resize(image, width=600)

# find the puzzle in the image and then
(puzzleImage, warped) = find_puzzle(image, whether_debug) # 找到数独的大区域

# initialize our 9x9 sudoku board
board = np.zeros((9, 9), dtype="int")

# a sudoku puzzle is a 9x9 grid (81 individual cells), so we can
# infer the location of each cell by dividing the warped image
# into a 9x9 grid
# shape[0]：图片垂直尺寸 shape[1]：图片水平尺寸
stepX = warped.shape[1] // 9
stepY = warped.shape[0] // 9

# initialize a list to store the (x, y)-coordinates of each cell
# location
cellLocs = []
add = []
# loop over the grid locations
for y in range(0, 9):
	# initialize the current list of cell locations
	row = []

	for x in range(0, 9):
		# compute the starting and ending (x, y)-coordinates of the
		# current cell
		startX = x * stepX
		startY = y * stepY
		endX = (x + 1) * stepX
		endY = (y + 1) * stepY

		# add the (x, y)-coordinates to our cell locations list
		# row.append((startX, startY, endX, endY))

		# crop the cell from the warped transform image and then
		# extract the digit from the cell
		cell = warped[startY+8:endY-4, startX+8:endX-4]
		digit = extract_digit(cell, whether_debug)

		# verify that the digit is not empty
		if digit is not None:
			# foo = np.hstack([cell, digit])
			# cv2.imshow("Cell/Digit", foo)

			# resize the cell to 28x28 pixels and then prepare the
			# cell for classification

			roi = cv2.resize(digit, (20, 20))
			paddings = tf.constant([[4, 4], [4, 4]])
			roi = np.pad(roi, paddings, "constant")
			tmp = copy.deepcopy(roi)
			roi = tf.cast(roi, dtype=tf.float32)
			roi = roi / 255.0
			roi = img_to_array(roi)
			roi = np.expand_dims(roi, axis=0)

			# classify the digit and update the sudoku board with the
			# prediction
			pred = model.predict(roi).argmax(axis=1)[0]


			if pred >= 10:
				pred = pred - 9
			'''
			print(pred)
			cv2.imshow('img', tmp)
			cv2.waitKey(0)
			a = int(input('whether to save:'))
			if a==1:
				b = int(input('flag:'))
				add.append([tmp, b])
			'''
			board[y, x] = pred
		if digit is not None:
			row.append((startX, startY, endX, endY, True))
		else:
			row.append((startX, startY, endX, endY, False))
	# add the row to our cell locations
	cellLocs.append(row)
# construct a sudoku puzzle from the board
np.save('1.npy', add)
print("[INFO] OCR'd sudoku board:")
puzzle = Sudoku(board=board.tolist())
print("[INFO] Sudoku difficulty", puzzle.difficulty)
puzzle.show(which_board='original')

# solve the sudoku puzzle
print("[INFO] solving sudoku puzzle...")
print('whether solve', puzzle.solve())
solution = puzzle.res_board
puzzle.show(which_board='result')
puzzle.correct()

# loop over the cell locations and board
count = 0
for (cellRow, boardRow) in zip(cellLocs, puzzle.res_board):
	# loop over individual cell in the row
	for (box, digit) in zip(cellRow, boardRow):
		# unpack the cell coordinates
		startX, startY, endX, endY, flag = box

		# compute the coordinates of where the digit will be drawn
		# on the output puzzle image
		textX = int((endX - startX) * 0.33)
		textY = int((endY - startY) * -0.2)
		textX += startX
		textY += endY

		# draw the result digit on the sudoku puzzle image
		if flag:
			if puzzle.pos_correct is None:
				cv2.putText(puzzleImage, str(digit), (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

			elif count == (puzzle.pos_correct[0]*9 + puzzle.pos_correct[1]):
				cv2.putText(puzzleImage, str(digit), (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
			else:
				cv2.putText(puzzleImage, str(digit), (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
		else:
			# count+=1
			# continue
			cv2.putText(puzzleImage, str(digit), (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
		count += 1
# show the output image
puzzle.show('result')
cv2.imshow("Sudoku Result", puzzleImage)
cv2.waitKey(0)