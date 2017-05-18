import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from random import shuffle
import os

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

LAYER_SIZE = 2  # change number to get more layers
LR = 1e-5       # learning rate
MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv-basic')

# Get labels
def get_label(name):

    if name == 'data/Ace':
        im_label = [1,0,0,0,0,0,0,0,0,0,0,0,0]
    elif name == 'data/2':
        im_label = [0,1,0,0,0,0,0,0,0,0,0,0,0]
    elif name == 'data/3':
        im_label = [0,0,1,0,0,0,0,0,0,0,0,0,0]
    elif name == 'data/4':
        im_label = [0,0,0,1,0,0,0,0,0,0,0,0,0]
    elif name == 'data/5':
        im_label = [0,0,0,0,1,0,0,0,0,0,0,0,0]
    elif name == 'data/6':
        im_label = [0,0,0,0,0,1,0,0,0,0,0,0,0]
    elif name == 'data/7':
        im_label = [0,0,0,0,0,0,1,0,0,0,0,0,0]
    elif name == 'data/8':
        im_label = [0,0,0,0,0,0,0,1,0,0,0,0,0]
    elif name == 'data/9':
        im_label = [0,0,0,0,0,0,0,0,1,0,0,0,0]
    elif name == 'data/10':
        im_label = [0,0,0,0,0,0,0,0,0,1,0,0,0]
    elif name == 'data/Jack':
        im_label = [0,0,0,0,0,0,0,0,0,0,1,0,0]
    elif name == 'data/Queen':
        im_label = [0,0,0,0,0,0,0,0,0,0,0,1,0]
    elif name == 'data/King':
        im_label = [0,0,0,0,0,0,0,0,0,0,0,0,1]

    return im_label

def create_train_data():
    training_data = []
    for filename in glob.glob('data/*.jpg'):
        name = filename.split("_")
        label = get_label(name[0])
        input_img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(input_img, (50, 50))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)

    return training_data

def create_model():
    convnet = input_data(shape=[None, 50, 50, 1], name='input')

    for i in range(LAYER_SIZE):
        convnet = conv_2d(convnet, 128, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, 13, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    return convnet

def predict_card(image, model):

    # img = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (50, 50))
    img = img.reshape(50,50,1)
    label = model.predict([img])[0]
    index = label.index(max(label)) + 1

    return index

def get_value(label):

    value = "No detection!"
    if label == 1:
        value = "Ace"
    elif label == 2:
        value = "2"
    elif label == 3:
        value = "3"
    elif label == 4:
        value = "4"
    elif label == 5:
        value = "5"
    elif label == 6:
        value = "6"
    elif label == 7:
        value = "7"
    elif label == 8:
        value = "8"
    elif label == 9:
        value = "9"
    elif label == 10:
        value = "10"
    elif label == 11:
        value = "Jack"
    elif label == 12:
        value = "Queen"
    elif label == 13:
        value = "King"

    return value


def main():
    train_data = create_train_data()

    convnet = create_model()
    model = tflearn.DNN(convnet, tensorboard_dir='log')

    # training model
    #  prepare data
    # train = train_data[:-300]
    # test = train_data[-300:]

    # X = np.array([i[0] for i in train]).reshape(-1,50,50,1)
    # Y = [i[1] for i in train]

    # test_x = np.array([i[0] for i in test]).reshape(-1,50,50,1)
    # test_y = [i[1] for i in test]

    # # # fit model
    # model.fit({'input': X}, {'targets': Y}, n_epoch=100, validation_set=({'input': test_x}, {'targets': test_y}),
    #     snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

    # model.save(MODEL_NAME)

    # loading model when we have model
    model.load(MODEL_NAME)

    # detect lively
    video_capture = cv2.VideoCapture(0)
    while 1:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        label = predict_card(frame, model)
        value = get_value(label)
        # print the result
        cv2.putText(frame, value, (50,100), cv2.FONT_HERSHEY_SIMPLEX, 2, 10, 10)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

