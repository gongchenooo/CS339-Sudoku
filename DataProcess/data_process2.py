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
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
def modify_img(thresh, img_original):
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
        print(h_min_new, h_max_new, w_min_new, w_max_new)
        # cv2.imshow("", thresh)
        # cv2.waitKey(0)
    print(new_thresh.shape)
    return new_thresh

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

                        img_original = img.copy()
                        img_blur = cv2.GaussianBlur(img, (7, 7), 3)

                        img_blur = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                                       11, 2)
                        img_original = cv2.adaptiveThreshold(img_original, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                                       11, 2)

                        img_blur = cv2.bitwise_not(img_blur)
                        img_original = cv2.bitwise_not(img_original)

                        img = modify_img(img_blur, img_original)
                        # cv2.imshow(f, img)
                        # cv2.waitKey(0)
                        if (img.shape == (0, 0)):
                            continue
                        img = cv2.resize(img, (20, 20))
                        img = np.pad(img, ((4, 4), (4, 4)), 'constant')
                        img = cv2.resize(img, (28, 28))
                        '''
                        if ('518030910301-1' in f):

                            cv2.imshow("blur", img_blur)
                            cv2.waitKey(0)
                            cv2.imshow("original", img_original)
                            cv2.waitKey(0)
                            cv2.imshow("img", img)
                            cv2.waitKey(0)
                        '''
                        WriteData.append(img)
                        WriteDataLabels.append(int(cls))
            if t == "testing":
                files = os.listdir(data_dir + cls + '/' + t)
                for f in files:
                    if ("jpg" or "png") in f:
                        img = cv2.imread(data_dir + cls + '/' + t + '/' + f)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图片转为灰度图
                        # img = cv2.GaussianBlur(img, (5, 5), 3)

                        img_original = img.copy()
                        img_blur = cv2.GaussianBlur(img, (7, 7), 3)

                        img_blur = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                         cv2.THRESH_BINARY,
                                                         11, 2)
                        img_original = cv2.adaptiveThreshold(img_original, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                             cv2.THRESH_BINARY,
                                                             11, 2)

                        img_blur = cv2.bitwise_not(img_blur)
                        img_original = cv2.bitwise_not(img_original)

                        img = modify_img(img_blur, img_original)
                        # cv2.imshow(f, img)
                        # cv2.waitKey(0)
                        if (img.shape == (0, 0)):
                            continue
                        img = cv2.resize(img, (20, 20))
                        img = np.pad(img, ((4, 4), (4, 4)), 'constant')
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
WriteDataLabels -= 1
# WriteDataLabels += 9
TestData = np.array(TestData)
TestDataLabels = np.array(TestDataLabels)
TestDataLabels -= 1
# TestDataLabels += 9

WriteData = WriteData.reshape((WriteData.shape[0], 28, 28, 1))
TestData = TestData.reshape((TestData.shape[0], 28, 28, 1))

'''
paddings = tf.constant([[0, 0], [2, 2], [2, 2]])
WriteData = tf.pad(WriteData, paddings, "CONSTANT")
TestData = tf.pad(TestData, paddings, "CONSTANT")

WriteData = np.array(WriteData).reshape((WriteData.shape[0], 32, 32, 1))
TestData = np.array(TestData).reshape((TestData.shape[0], 32, 32, 1))
'''
WriteData = WriteData.astype("float32") / 255.0
TestData = TestData.astype("float32") / 255.0
WriteData = (WriteData - 0.5) / 0.5
TestData = (TestData - 0.5) / 0.5
# np.save('Data/Chinese_train_data_4.npy', WriteData)
# np.save('Data/Chinese_train_labels_4.npy', WriteDataLabels)
# np.save('Data/Chinese_test_data_4.npy', TestData)
# np.save('Data/Chinese_test_labels_4.npy', TestDataLabels)

print("[INFO] accessing MNIST...")
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
testData = testData.reshape((testData.shape[0], 28, 28, 1))

'''
paddings = tf.constant([[0, 0], [2, 2], [2, 2]])
trainData = tf.pad(trainData, paddings, "CONSTANT")
testData = tf.pad(testData, paddings, "CONSTANT")
trainData = np.array(trainData).reshape((trainData.shape[0], 32, 32, 1))
testData = np.array(testData).reshape((testData.shape[0], 32, 32, 1))
'''
trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0
trainData = (trainData - 0.5) / 0.5
testData = (testData - 0.5) / 0.5
# np.save('Data/English_train_data_4.npy', trainData)
# np.save('Data/English_train_labels_4.npy', trainLabels)
# np.save('Data/English_test_data_4.npy', testData)
# np.save('Data/English_test_labels_4.npy', testLabels)


WriteDataLabels += 10
TestDataLabels += 10
trainData = np.concatenate([WriteData, trainData])
testData = np.concatenate([TestData, testData])
trainLabels = np.concatenate([WriteDataLabels, trainLabels])
testLabels = np.concatenate([TestDataLabels, testLabels])

shuffle_ix = np.random.permutation(np.arange(len(trainData)))
trainData = trainData[shuffle_ix]
trainLabels = trainLabels[shuffle_ix]

shuffle_ix = np.random.permutation(np.arange(len(testData)))
testData = testData[shuffle_ix]
testLabels = testLabels[shuffle_ix]
# add a channel (i.e., grayscale) dimension to the digits
'''
trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
testData = testData.reshape((testData.shape[0], 28, 28, 1))
'''
# scale data to the range of [0, 1]

np.save("Data/mix_train_data_28_decentralize.npy", trainData)
np.save("Data/mix_train_labels_28_decentralize.npy", trainLabels)
np.save("Data/mix_test_data_28_decentralize.npy", testData)
np.save("Data/mix_test_labels_28_decentralize.npy", testLabels)
