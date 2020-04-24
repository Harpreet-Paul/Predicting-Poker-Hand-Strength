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

In order to generate the target variable, we take the 'pocket' column and apply the Chen Formula, which maps each hand to a hand strength value between -2 and 20. The bar plot shows the frequencies of each hand strength value.

From the plot we can see that a hand strength of 5 is most common and that most hand strengths fall in range between 2 and 10. The mean, median and standard deviation for the hand strengths are: 6.11, 6.0 and 3.62 respectively. 

All of the total betting action count features show a fair bit of spread. Each has roughly the same spread, with the total call count feature showing the most spread and the total bet count feature showing the least. Calling appears to be the most common betting action. There appear to be a number of very high leverage points for the total raise count feature. We will address these later when diagnosing our logistc regression model fit. 

There is a negative linear relationship between a player's total check count and hand strength. There is no clear relationship between total call count and hand strength, but a call count of 0 seems to imply a little bit above average of a handstrength and calling more than 10 times seems to imply moderately below average hand strength. There is a positive linear relationship between total bet count and hand strength. 

In restricting our plot to only show total raise counts in the range of 0 and 12 raises, we see a very non-linear relationship between hand strength and the total raise count. Up until about 3 total raises, there is a positive and roughly linear relationship between hand strength and total raise count. After 3 total raises and until 8 raises, there is a negative and roughly linear relationship between hand strength and total raise count. Ultimately though, a total raise count of 1 or higher seems to imply an above average hand strength. 



The first set of features we generate are counts of the number of times each betting action (call, check, raise, bet and all-in) was performed in each of the stages of betting. Our assumption is that aggresive moves, like raising, betting and going all-in should be an indication that a player has a strong pocket and passive moves like checking should be an indication that a player has a weak pocket. Calling is somewhere in the middle on the passive-aggressive spectrum of betting actions and thus we expect that it should be an indication of neither a particularily strong nor a particularily weak pocket. 

We visualized the spread of each of these features to see if there was enough variance for the features to have predictive power, then we plotted the features against hand strengths to test our intuitions about the underlying relationships between them.

We see that the preflop bet count feature has 0 variance and the preflop, flop, turn and river all-in count features have very little variance. The preflop, flop, turn and river raise count and call count features have the most variance. 

From the spread of the preflop, flop, turn and river raise count features, we can see that there are many high leverage and potential outlier points that could impact the fit of the logistic regression model. We note this for now and will return to this issue when we diagnose our logistic regression fit by using influence plots. 

In plotting the action count features against hand strength, we see the following relationships:

There is a strong negative relationship between checking in the preflop and pocket hand strength. There is a moderate negative relationship between checking in the flop, turn and river stages and pocket hand strength.

There is a moderately negative relationship between calling in the preflop and pocket hand strength. There is no relationship between calling in the flop and hand strength. There is a moderately positive relationship between calling in the turn and river stages and hand strength. It seems that a moderate relationship between calling and hand strength does exist, contrary to our assumption.

There is a strongly negative relationship between going all-in in the preflop and hand strength, and a moderately negative relationship between going all-in in any other stage and hand strength. This relationship runs completely counter to our assumptions. Because there is a very, very small sample size of hands where a player went all-in, and intuitvely players would only go all-in when they possess stronger hands, it is likely that this relationship does not reflect the true relationhsip that exists between the variables. The all-in count features will be removed from the analysis as it seems like they are simply introducing noise into the model. 

There is a moderately positive relationship between betting in the flop and turn and hand strength. There is a weakly positive relationship between betting in the river and hand strength. 

For the sake of visualization, we plot raise counts in the range of 0-5 raises against hand strength. In the preflop stage, there is a strong positive relationship between raising or not raising and hand strength, but no clear relationship between hand strength and additional raises. In the flop, there is a moderately positive relationship between raising or not raising and hand strength, but no clear relationship between hand strength and additional raises. In the turn, there is a weakly positive relationship between raising or not raising and hand strength, but no clear relationship between hand strength and additional raises. In the river stage, there is no relationship between number of raises and hand strength. 

The next set of features we generated were the amounts bet by a player in each stage of betting. Our assumption was that higher bet amounts should be indicative of a stronger pocket and lower bet amounts should be indicative of a weaker pocket. 

As with the betting actions, we visualized the spread of the betting amounts and then plotted betting amounts against hand strengths to test our assumptions.

Similar to the case with raise count features, in plotting the spread of the bet amount features we see that there are many high leverage (and potential outlier) points in the data. We will manage these points later, and for now visualize the spread of the bet amount features in the ranges of 0-70 bet for the preflop and flop bet amounts, and 0-100 bet for the turn and river bet amounts. 

All 4 features show adequate spread. The preflop and flop bet amount distributions show similar spread, but the average bet amount in the preflop (20) is double the average in the flop (10). The turn and river bet amount distributions are nearly identical.

In the preflop, hand strength mostly seems to increase proportionately with the bet amount range. In the flop, not betting seems to imply a decently below average handstrength whereas betting implies an above average hand strength to some extent. There does not seem to be an obvious trend to changes in hand strength in response to incremental increases in the bet amount range above bets of 0. Bet amounts in the range of 80-100 do seem to imply a significantly above average hand strength, though. In the turn, betting an amount in the range of 0-10 seems to imply a below average handstrength while betting above that range implies an above average hand strength. There does not seem to be an obvious trend to changes in hand strength in response to incremental increases in the bet amount range above bets of 10. In the river, betting 0 seems to imply a slightly below average hand strength whereas betting above 0 seems to imply an average or slightly above average hand strength. 


The last set of features we generated were ratios of the amount a player bet in a stage of betting to the pot size at the beginning of that stage of betting. Our intuiton was that a given bet amount relative to a smaller pot is more indicative of a strong pocket than the same bet amount relative to a larger pot. 


We assume that amateur players typically do not play hands close in hand strength value differently enough such that we could expect to accurately discriminate between say a strength 5 hand and a strength 6 hand on the basis of betting actions. To test this assumption, we look at how some of our features vary between adjacent hand strength values in the lower, middle and upper range of hand strength. 





