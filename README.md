# Card Classification
## EECS 349 Machine Learning at Northwestern University

### [**Final Report**](/final_report/EECS349final_Group18.pdf)

**Group Members' Github and Contact:** 

[**Elton Cheng**](https://github.com/echeng22) eltoncheng2017@u.northwestern.edu

[**Yuchen Rao**](https://github.com/yuchenrao) yuchenrao2017@u.northwestern.edu

[**Weiyuan Deng**](https://github.com/WeiyuanDeng) weiyuandeng2017@u.northwestern.edu

### Objective and Introduction

The goal of our project is to classify cards for a game of Blackjack. Given an image of a card, can the computer correctly identify it? (Ace, 2-10, Jack, Queen, King). 

This task explores the idea of object recognition, a tool that is being used more and more in fields, such as self-driving cars and pick/place and sorting robots. Object recognition can provide more information for computers to make decisions on by being able to tell apart a red balloon to a stop sign, or a bottle of gatorade from dish soap.

We used playing cards specifically because for people, it is easy for us to classify each card, as we can tell them apart from the letters, numbers, artwork, etc. of the cards. The challenge for a computer to classify these cards comes in trying to recognize these features of the cards, and being able to interpret these features correctly. Each of the cards are unique enough that it can be a challenge for a computer to try and classify them all. 


![image of Blackjack here](/images/image1.JPG)
*Figure 1. Blackjack*

### Dataset and Approach

At the beginning, the dataset contained 1040 images of one decks of 52 cards (Ace = 1, 2 ~ 10, Jack = 11, Queen = 12, and King = 13).

![image of getting dataset](/images/image2.JPG)
*Figure 2. The camera was fixed at a constant height and took pictures of every card.*

Each card has 20 pictures with the same background in different orientation. Then with data augmentation, we were able to add small random translations, rotations, zoom and scaling to expand the dataset to 10400 image (10 augments per image).

![image of after data augmentation](/images/image5.JPG)
*Figure 3. After data augmentation, the dataset has been expanded to 10400 images.*

The learners in the project are 5-Nearest Neighbor, Support Vector Machine, 15-Layer Neural Net algorithm from Scikit-Learn and Convolutional Neural Net algorithm(CNN) from tflearn packages. The feature used for CNN is convolutional features, while features for other algorithms is Dense SIFT (DSIFT), a computer vision tool used to detect objects in an image. 

10-fold cross validation is used to verify the accuracy of the model.

### Results

With the augmented data set, the accuracy of SVM and CNN algorithms increased significantly. The best model we obtained was around 97% accuracy from the CNN algorithm. The features used for the best model used a convolutional filter size of 5.

![image of card recognition](/images/Image3.JPG)
*Figure 4. The algorithm is able to recognize every card it had learned. 13 Cards are shown to the camera with random order, from the screen we can tell the algorithm works well.*

### Demo Video

Here is a demo video of our final model.

<div align="center">
    <video align="center" src="demo/mltest.mp4" poster="images/demo.JPG" width="600" height="400" controls preload></video>
</div>
