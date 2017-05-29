# Black Jack Buddy
## EECS 349 Machine Learning at Northwestern University

**Group Members and Contact:** 

Elton Cheng <eltoncheng2017@u.northwestern.edu>

Yuchen Rao <yuchenrao2017@u.northwestern.edu>

Weiyuan Deng <weiyuandeng2017@u.northwestern.edu>


### Objective and Introduction

The objective of the card classification project is to interpret and classify cards for a game of Blackjack. The camera detects the cards  and the computer has an algorithm to classify them in real time while playing. The computer will be able to decide if the player should add more cards or stop.

Blackjack, also known as twenty-one, is the most widely played casino banking game in the world. It is a comparing card game between a player and dealer, and it is played with one or more decks of 52 cards. It is interesting that we apply computer vision with machine learning on the popular gambling game and be able to do the object detection and recognition.

[image of Blackjack here]

### Dataset and Approach

The dataset contains 10400 images of one decks of 52 cards (Ace = 1, 2 ~ 10, Jack = 11, Queen = 12, and King = 13). We obtained the images by taking pictures of each card with an outer camera fixed at a constant height and labeled them by changing file name. Each card has 20 pictures with the same background in different orientation. Then with data augmentation, we were able to add small translation, rotation and scaling to expand the dataset to 200 pictures for each card.



### Results



e. A ~2 paragraph synopsis of what this work is about
ii. Describe your approach in high-level terms: what kind of learner(s) did you
use, what types of features did you use
iii. Describe the key results (how well your solution performed in no more than
a paragraph, along with your key findings, e.g. which learners performed
best, which features were most important)


f. At least one picture or graph that illustrates your work, with a caption explaining
what the figure shows and its significance.
