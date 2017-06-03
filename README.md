# Card Classification
## EECS 349 Machine Learning at Northwestern University

**Group Members' Github and Contact:** 

[**Elton Cheng**](https://github.com/echeng22) eltoncheng2017@u.northwestern.edu

[**Yuchen Rao**](https://github.com/yuchenrao) yuchenrao2017@u.northwestern.edu

[**Weiyuan Deng**](https://github.com/WeiyuanDeng) weiyuandeng2017@u.northwestern.edu


### Objective and Introduction

The objective of the card classification project is to interpret and classify cards for a game of Blackjack. The camera detects the cards  and the computer has an algorithm to classify them in real time while playing.

![image of Blackjack here](/images/image1.JPG)
*Figure 1. Blackjack*

### Dataset and Approach

At the begining, the dataset contains 1040 images of one decks of 52 cards (Ace = 1, 2 ~ 10, Jack = 11, Queen = 12, and King = 13). 

![image of getting dataset](/images/image2.JPG)
*Figure 2. The camera was fixed at a constant heigh and took pictures of every card.*

Each card has 20 pictures with the same background in different orientation. Then with data augmentation, we were able to add small translation, rotation and scaling to expand the dataset to 10400 images, 200 for each card.

![image of after data augmentation](/images/image5.JPG)
*Figure 2. After data augmentation, the dataset has been expanded to 10400 images.*

The learners in the project are 5-Nearest Neighbor, Support Vector Machine, Neural Net algorithm with 15 layers from Scikit-Learn
and Convolutional Neural Net algorithm(CNN) from tflearn packages. The feature used for CNN is raw data, wihle feature for other algorithms is [Dense SIFT (DSIFT)](http://docs.opencv.org/trunk/da/df5/tutorial_py_sift_intro.html)

10-fold cross validation is used to verify the accuracy of the model.

### Results

With data augmentation, the accuracy has an significant increase. The final accuracy of the model is around 97%, which is giving by CNN. The correspongding important feature for is only pixel information. 

![image of card recognition](/images/Image3.JPG)
*Figure 3. The algorithm is able to reconize every card it had learned. 13 Cards are shown to the camera with random order, from the screen we can tell the algothm works well.*

### Demo Video

Here is a demo video of our final model.

<div align="center">
    <video align="center" src="demo/mltest.mp4" poster="images/demo.JPG" width="600" height="400" controls preload></video>
</div>
