# Predicting-Poker-Hand-Strength

## Introduction

### Background

Texas Hold’em is a popular variant of Poker. Between 2-8 players sit at a table and each player is dealt two face-down cards (called their “pocket”). Across 4 stages of betting, 5 face-up cards (called “community cards”) are played on the table. After the last stage of betting, remaining players reveal their pocket cards and make the best 5 card hand possible, using a combination of their own cards and the face-up community cards. The player with the best 5 card hand collects all the money bet by each of the players (called the “pot”) for that round. 

### Objective

The objective of this project was to predict the strength of a player’s two pocket cards, as determined by a hand strength formula known as the “Chen Formula”. Knowing the strength of an opponent’s pocket cards allows a player to make optimal betting decisions in a round. 

### Methods

Data was obtained from the UofAlberta IRC Poker Database, which contained game logs for thousands of online players. First, the dataset was segmented by dividing players into different groups based on playing style using K-Means clustering. Then, Logistic Regression, Multinomial Naive Bayes and AdaBoost (Boosted) Decision Tree models were fit to each player group using player betting actions and betting amounts within a round as predictors for pocket hand strength. 

### Results

For all player groups, the Logistic Regression model proved the most accurate in predicting pocket hand strength, with accuracies in the range of 30-36% depending on the player group. 

## Data Processing and Visualization

### Understanding the Raw Data

This is what our original dataframe looks like:

Each row represents a hand played by a given player in a given game. The important columns are interpreted as follows:

##### 'preflop', 'flop_x', 'turn_x' and 'river_x'

The string of characters underneath these columns represent betting actions by the player in the different stages of betting. The betting action is encoded with a single character for each action:

|  Character   |    Action    |
| ------------ | ------------ |
| - | no action; player is no longer contesting pot |
| B | blind bet |
| f | fold |
| k | check |
| b | bet |
| c | call |
| r | raise |
| A | all-in |
| Q | quits game |
| K | kicked from game |

##### 'bankroll'

Total amount available for a player to bet at the beginning of a round. 

##### 'card_1' and 'card_2'

If the player played until the showodown without folding, then their two pocket cards are shown under these two columns. 

##### 'flop_y', 'turn_y', 'river_y' and 'showdown'

The first number is the number of players remaining in the game at the start of the stage. The second number is the total pot size at the start of the stage (the showdown is when players reveal their cards; no betting happens at this stage). 

### Generating Features and the Target Variable

In order to generate the target variable, we take the 'pocket' column and apply the Chen Formula, which maps each hand to a hand strength value between -2 and 20. The histogram shows the frequencies of each hand strength value.

The first set of features we generate are counts of the number of times each betting action (call, check, raise, bet and all-in) was performed in each of the stages of betting. Our assumption is that aggresive moves, like raising, betting and going all-in should be an indication that a player has a strong pocket and passive moves like checking should be an indication that a player has a weak poker. Calling is somewhere in the middle on the passive-aggressive spectrum of betting actions and thus we expect that it should be an indication of neither a particularily strong nor a particularily weak pocket. 

We visualized the spread of each of these features to see if there was enough variance for the features to have predictive power, then we plotted the features against hand strengths to test our intuitions about the underlying relationships. 


We assume that amateur players typically do not play hands close in hand strength value differently enough such that we could expect to accurately discriminate between say a strength 5 hand and a strength 6 hand on the basis of betting actions. To test this assumption, we look at how some of our features vary between adjacent hand strength values in the lower, middle and upper range of hand strength. 





