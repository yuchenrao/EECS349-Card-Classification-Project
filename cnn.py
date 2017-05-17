import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from random import shuffle

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

    model = tflearn.DNN(convnet, tensorboard_dir='log')
    return model

def main():
    train_data = create_train_data()

    model = create_model()

    # prepare data
    train = train_data[:-300]
    test = train_data[-300:]

    X = np.array([i[0] for i in train]).reshape(-1,50,50,1)
    Y = [i[1] for i in train]

    test_x = np.array([i[0] for i in test]).reshape(-1,50,50,1)
    test_y = [i[1] for i in test]

    # fit model
    model.fit({'input': X}, {'targets': Y}, n_epoch=100, validation_set=({'input': test_x}, {'targets': test_y}),
        snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

    model.save(MODEL_NAME)

if __name__ == "__main__":
    main()

