import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import cross_val_score, KFold
from scipy.stats import sem
from sklearn import metrics
from scipy.ndimage import zoom
from matplotlib.patches import Rectangle
from sklearn.preprocessing import StandardScaler

# different machine learning algrithom
svc_1 = SVC(kernel='linear', decision_function_shape='ovo')
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
# cnn = MLPClassifier(hidden_layer_sizes=(2,), random_state=1, max_iter=1)
cnn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)

# Get labels
def getlabel(name):

    if name == 'data/Ace':
        im_label = '1'
    elif name == 'data/2':
        im_label = '2'
    elif name == 'data/3':
        im_label = '3'
    elif name == 'data/4':
        im_label = '4'
    elif name == 'data/5':
        im_label = '5'
    elif name == 'data/6':
        im_label = '6'
    elif name == 'data/7':
        im_label = '7'
    elif name == 'data/8':
        im_label = '8'
    elif name == 'data/9':
        im_label = '9'
    elif name == 'data/10':
        im_label = '10'
    elif name == 'data/Jack':
        im_label = '11'
    elif name == 'data/Queen':
        im_label = '12'
    elif name == 'data/King':
        im_label = '13'

    return im_label

# get features based on SIFT features
def getSIFTfeatures(input_img):

    # gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT()
    step_size = 10
    newimg = input_img
    # newimg = zoom(gray, (200. / gray.shape[0], 200. / gray.shape[1]))
    kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, newimg.shape[0], step_size)
                                        for x in range(0, newimg.shape[1], step_size)]
    kp, dense = sift.compute(newimg, kp)
    im_mat = dense

    return im_mat

# get features based on edge
def getEdgefeatures(input_img):

    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    input_img = zoom(gray, (300. / gray.shape[0], 400. / gray.shape[1]))
    edges = cv2.Canny(input_img,100,200)

    return edges


# ecaluate by 10-fold cross validation
def evaluate_cross_validation(clf, X, y, K):
    # create a k-fold cross validation iterator
    cv = KFold(len(y), K, shuffle=True, random_state=0)
    # by default the score used is the one returned by score method of the estimator (accuracy)
    scores = cross_val_score(clf, X, y, cv=cv)
    print clf
    print (scores)
    print ("Mean score: {0:.3f} (+/-{1:.3f})".format(
        np.mean(scores), sem(scores)))

# training and evaluting
def train_and_evaluate(clf, X_train, X_test, y_train, y_test):

    clf.fit(X_train, y_train)

    print ("Accuracy on training set:")
    print (clf.score(X_train, y_train))
    print ("Accuracy on testing set:")
    print (clf.score(X_test, y_test))

    y_pred = clf.predict(X_test)

    print ("Classification Report:")
    print (metrics.classification_report(y_test, y_pred))
    print ("Confusion Matrix:")
    print (metrics.confusion_matrix(y_test, y_pred))


def main():

    # image preprocessing
    image_list = []
    image_data = []
    image_label = []
    length = []
    scaler = StandardScaler()

    for filename in glob.glob('data/*.jpg'):
        name = filename.split("_")
        label = getlabel(name[0])
        input_img = cv2.imread(filename)
        im_feature = getSIFTfeatures(input_img)
        # im_feature = getEdgefeatures(input_img)
        im_data = im_feature.flatten()
        image_list.append(input_img)
        image_data.append(im_data)
        image_label.append(label)

    # machine learning
    # np.random.shuffle(image_data)
    train_data = image_data[:700]
    train_labels = image_label[:700]
    test_data = image_data[700:]
    test_labels = image_label[700:]
    # for cnn, scale the feature values
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    evaluate_cross_validation(cnn, train_data, train_labels, 10)
    train_and_evaluate(cnn, train_data, test_data, train_labels, test_labels)

if __name__ == "__main__":
    main()
