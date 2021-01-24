# coding=utf-8
# import the necessary packages
# Holds the SudokuNet CNN architecture implemented with TensorFlow and Keras.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import tensorflow as tf
import argparse
import numpy as np
import os


# 选择编号为2的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config=tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.compat.v1.Session(config=config)


class SudokuNet:
	@staticmethod
	def build(width, height, depth, classes):
		'''
		width: width of an MNIST digit(28 pixels)
		height: height of an MNIST digit(28 pixels)
		depth: channels of MNIST digit images(1 grayscale channel)
		classes: the number of digits 0-9(10 digits)
		'''
		# initialize the model
		model = Sequential()
		inputShape = (height, width, depth)

		# first set of CONV => RELU => POOL layers
		'''
		keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', ...)
		'''
		model.add(Conv2D(32, (5, 5), padding="same", input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		# second set of CONV => RELU => POOL layers
		model.add(Conv2D(32, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		# first set of FC => RELU layers
		'''
		Dense(units, activation=None, kernel_initializer='glorot_uniform')
		units即神经元
		在全连接层添加一个层使输入的维数变为units维: output = activation(dot(input, kernel) + bias)
		'''
		model.add(Flatten()) # 从多维变为一维，常用在从卷积层到全连接层过渡
		model.add(Dense(64)) #
		model.add(Activation("relu"))
		model.add(Dropout(0.5))

		# second set of FC => RELU layers
		model.add(Dense(64))
		model.add(Activation("relu"))
		model.add(Dropout(0.5)) # Fully-connected layer set with 50% dropout

		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model


if __name__ == '__main__':
    INIT_LR = 1e-4
    EPOCHS = 50
    BS = 32

    trainData = np.load("mix_train_data_28.npy")
    testData = np.load("mix_test_data_28.npy")
    trainLabels = np.load("mix_train_labels_28.npy")
    testLabels = np.load("mix_test_labels_28.npy")

    # convert the labels from integers to vectors
    le = LabelBinarizer()
    trainLabels = le.fit_transform(trainLabels)
    testLabels = le.transform(testLabels)
    # construct the argument parser and parse the arguments

    # initialize the optimizer and model
    print("[INFO] compiling model...")
    opt = Adam(lr=INIT_LR)
    model = SudokuNet.build(width=28, height=28, depth=1, classes=19)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # train the network
    print("[INFO] training network...")
    H = model.fit(
        trainData, trainLabels,
        validation_data=(testData, testLabels),
        batch_size=BS,
        epochs=EPOCHS,
        verbose=1)

    acc = H.history['accuracy']
    val_ac = H.history['val_accuracy']
    loss = H.history['loss']
    val_loss = H.history['val_loss']

    epochs = range(len(acc))  # Get number of epochs
    # 画accuracy曲线
    plt.plot(epochs, acc, 'r')
    plt.plot(epochs, val_ac, 'b')
    plt.title('Training and Testing Accuracy of Original Net')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["Training Accuracy", "Validation Accuracy"])
    fig1 = plt.gcf()
    fig1.savefig('Original_accuracy.png', dpi=300)
    plt.figure()

    # 画loss曲线
    plt.plot(epochs, loss, 'r')
    plt.plot(epochs, val_loss, 'b')
    plt.title('Training and Testing Loss of Original Net')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Training Loss", "Testing Loss"])
    fig2 = plt.gcf()
    fig2.savefig('Original_loss.png', dpi=300)
    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(testData)
    print(classification_report(
        testLabels.argmax(axis=1),
        predictions.argmax(axis=1),
        target_names=[str(x) for x in le.classes_]))

    # serialize the model to disk
    print("[INFO] serializing digit model...")
    model.save("Original.h5", save_format="h5")
