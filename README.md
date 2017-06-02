# Black Jack Buddy
## EECS 349 Machine Learning at Northwestern University

**Group Members and Contact:** 

Elton Cheng <eltoncheng2017@u.northwestern.edu>

Yuchen Rao <yuchenrao2017@u.northwestern.edu>

Weiyuan Deng <weiyuandeng2017@u.northwestern.edu>


### Objective and Introduction

The objective of the card classification project is to interpret and classify cards for a game of Blackjack. The camera detects the cards  and the computer has an algorithm to classify them in real time while playing. The computer will be able to decide if the player should add more cards or stop.

Blackjack, also known as twenty-one, is the most widely played casino banking game in the world. It is a comparing card game between a player and dealer, and it is played with one or more decks of 52 cards. It is interesting that we apply computer vision with machine learning on the popular gambling game and be able to do the object detection and recognition.

![image of Blackjack here](https://github.com/yuchenrao/EECS349-Card-Classification-Project/blob/master/images/image1.JPG)

### Dataset and Approach

The dataset contains 10400 images of one decks of 52 cards (Ace = 1, 2 ~ 10, Jack = 11, Queen = 12, and King = 13). We obtained the images by taking pictures of each card with an outer camera fixed at a constant height and labeled them by changing file name. Each card has 20 pictures with the same background in different orientation. Then with data augmentation, we were able to add small translation, rotation and scaling to expand the dataset to 200 pictures for each card.

![image of getting dataset](https://github.com/yuchenrao/EECS349-Card-Classification-Project/blob/master/images/image2.JPG)

There are two ways to extract features in the project. The first one is to use a computer vision (CV) algorithm called Dense SIFT (DSIFT) (http://docs.opencv.org/trunk/da/df5/tutorial_py_sift_intro.html) to extract the features that the model will train on for sckit-learn algorithms. The other is to preprocess the images with edge detection to use edge information of the cards as features.
After test of each feature, DSIFT is chosen.

The learners used in the project are 5-Nearest Neighbor, Support Vector Machine, Neural Net algorithm with 15 layers from Scikit-Learn
and Convolutional Neural Net algorithm from tflearn packages. 10-fold cross validation is used to verify the accuracy of the model.

### Results

![image of card recognition](https://github.com/yuchenrao/EECS349-Card-Classification-Project/blob/master/images/image2.JPG)

iii. Describe the key results (how well your solution performed in no more than
a paragraph, along with your key findings, e.g. which learners performed
best, which features were most important)


f. At least one picture or graph that illustrates your work, with a caption explaining
what the figure shows and its significance.
