import cv2
import imutils
import numpy as np
from skimage.segmentation import clear_border
import matplotlib.pyplot as plt
import os
import skimage.io
import skimage.color
import random
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer

data_dir = 'EI339-CN dataset sjtu/'  # 文件地址/名称
classes = os.listdir(data_dir)
WriteData = []
WriteDataLabels = []
TestData = []
TestDataLabels = []
for cls in classes:
    if (cls=='10'):
        print('10 not include')
        continue
    if len(cls) <= 2:
        folds = os.listdir(data_dir + cls)
        for t in folds:
            if t == "training":
                files = os.listdir(data_dir + cls + '/' + t)
                for f in files:
                    if ("jpg" or "png") in f:
                        img = cv2.imread(data_dir + cls + '/' + t + '/' + f)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图片转为灰度图
                        img = cv2.bitwise_not(img)
                        img = cv2.resize(img, (28, 28))
                        WriteData.append(img)
                        WriteDataLabels.append(int(cls))
            if t == "testing":
                files = os.listdir(data_dir + cls + '/' + t)
                for f in files:
                    if ("jpg" or "png") in f:
                        img = cv2.imread(data_dir + cls + '/' + t + '/' + f)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图片转为灰度图
                        img = cv2.bitwise_not(img)
                        img = cv2.resize(img, (28, 28))
                        TestData.append(img)
                        TestDataLabels.append(int(cls))

c = list(zip(WriteData, WriteDataLabels))
random.shuffle(c)
WriteData[:], WriteDataLabels[:] = zip(*c)
c = list(zip(TestData, TestDataLabels))
random.shuffle(c)
TestData[:], TestDataLabels[:] = zip(*c)

WriteData = np.array(WriteData)
WriteDataLabels = np.array(WriteDataLabels)
WriteDataLabels += 9
TestData = np.array(TestData)
TestDataLabels = np.array(TestDataLabels)
TestDataLabels += 9
print("[INFO] accessing MNIST...")
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
trainData = np.concatenate([WriteData, trainData])
testData = np.concatenate([TestData, testData])
trainLabels = np.concatenate([WriteDataLabels, trainLabels])
testLabels = np.concatenate([TestDataLabels, testLabels])

# add a channel (i.e., grayscale) dimension to the digits
trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
testData = testData.reshape((testData.shape[0], 28, 28, 1))

# scale data to the range of [0, 1]
trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0

np.save("train_data_1.npy", trainData)
np.save("train_labels_1.npy", trainLabels)
np.save("test_data_1.npy", testData)
np.save("test_labels_1.npy", testLabels)