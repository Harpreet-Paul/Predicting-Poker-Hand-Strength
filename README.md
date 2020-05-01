# Predicting-Poker-Hand-Strength

## Introduction

### Background

Texas Hold’em is a popular variant of Poker. Between 2-8 players sit at a table and each player is dealt two face-down cards (called their “pocket”). Across 4 stages of betting, 5 face-up cards (called “community cards”) are played on the table. After the last stage of betting, remaining players reveal their pocket cards and make the best 5 card hand possible, using a combination of their own cards and the face-up community cards. The player with the best 5 card hand collects all the money bet by each of the players (called the “pot”) for that round. 

### Objective

The objective of this project was to predict the strength of a player’s two pocket cards, as determined by a hand strength formula known as the “Chen Formula”. 

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

#### Target Variable

In order to generate the target variable, we took the 'pocket' column and applied the Chen Formula, which maps each hand to a hand strength value between -2 and 20. The bar plot shows the frequencies of each hand strength value.

From the plot we can see that a hand strength of 5 is most common and that most hand strengths fall in the range between 2 and 10. The mean, median and standard deviation for the hand strengths are 6.11, 6.0 and 3.62 respectively.

#### Features

For each feature, we visualized it's spread and then plotted hand strength against that feature to see if there was any relationship between the two. All of our features were discrete variables, so we plotted average hand strength of all hands corresponding to a particular value of the feature against possible values of the feature. 

##### Preflop, Flop, Turn and River -- Check/Bet/Call/Raise Counts

These features were counts of the number of times each betting action (call, check, raise and bet) was performed in each of the stages of betting (preflop, flop, turn and river).

<ins>Spread</ins>

The bet and check count features in all stages of betting follow a bernoulli distribution. A check count of 0 in the preflop is far more common than a check count of 1, but the proportion of hands with a check count of 1 steadily increases until the river, where it is roughly half. In all stages of betting, the number of hands with bet counts of 0 are about double the number of hands with bet counts of 1. 

The raise and call count features in all stages of betting roughly follow a poisson distribution. From the preflop to the river, the spreads of the call count features decrease and the means decreases towards 0. From the spread of the preflop, flop, turn and river raise count features, we see that there are many high leverage points that could impact the fit of the logistic regression model. We will address these high leverage points later when diagnosing the logistic regression model fit. To better visualize the raise count features, we limit our plots to show the frequency of raise counts between 0 and 5 raises per betting stage. From the preflop to the river, the spreads of the raise count features decrease and the means decreases towards zero.

<ins>Relationship to Hand Strength</ins>

There is a strong negative relationship between checking in the preflop and average pocket hand strength. There is a moderately negative relationship between checking in the flop, turn and river stages and average hand strength.

There is a moderately positive relationship between betting in the flop and turn stages and average hand strength. There is a weakly positive relationship between betting in the river and average hand strength. 

There is a non-linear relationship between calling in the preflop and average hand strength. Average hand strength decreases from 0 to 1 call in the preflop, but thereafter remains stable with additional calls in the preflop. There is no relationship between calling in the flop and hand strength. There is a weakly positive linear relationship between call count in the turn and average hand strength. There is a moderately positive linear relationship between river call count and average hand strength. 

There are no clear relationships between the raise count features and average hand strength when we plot the total range of raise counts for each stage of betting. When we plot raise counts in the range of 0-5 raises against average hand strength, some relationships emerge. In the preflop stage, average hand strength increases significantly between 0 and 1 raise, but there is no clear relationship between average hand strength and additional raises thereafter. In the flop, average hand strength increases by a moderate amount between 0 and 1 raise, but there is no clear relationship between average hand strength and additional raises thereafter. In the turn, average hand strength goes up slightly between 0 and 1 raise, but there is no clear relationship between average hand strength and additional raises thereafter. In the river stage, there is no relationship between number of raises and average hand strength. 

##### Total Check/Bet/Call/Raise Counts

These features were counts of the total number of times a player checked, called, raised or bet in a given hand. Each feature was generated by summing the counts for that particulation betting action across all stages of betting. E.g. Total Call Count = preflop call count + flop call count + turn call count + river call count. 

<ins>Spread</ins>

All of the total betting action count features roughly follow a poisson distribution.  Each feature has roughly the same spread, with the total call count feature showing the most spread and the total bet count feature showing the least. Calling appears to be the most common betting action. There are a number of very high leverage points for the total raise count feature. We will address these later when diagnosing our logistc regression model fit. 

<ins>Relationship to Hand Strength</ins>

There is a strong negative linear relationship between a player's total check count and average hand strength. There is no clear relationship between total call count and average hand strength, but a call count of 0 seems to imply a slightly above average hand strength and calling more than 10 times seems to imply a below average hand strength. There is a moderate positive linear relationship between total bet count and average hand strength. 

There is no clear relationship between the total raise count feature and average hand strength when we plot the entire range of total raise count values. In restricting our plot to only show total raise counts in the range of 0 and 12 raises, we see a very non-linear relationship between average hand strength and the total raise count. Up until about 3 total raises, there is a positive and roughly linear relationship between hand strength and total raise count. After 3 total raises and until 8 raises, there is a negative and roughly linear relationship between average hand strength and total raise count. Ultimately though, a total raise count of 1 or higher seems to imply an above average hand strength. 

##### Total Bet

This feature is the total amount that a player bet across all 4 stages of betting. 

<ins>Spread</ins>

We see a number of very high leverage points in plotting the spread of the total bet amounts. These will be addressed later when diagnosing our logistic regression model fit. Nearly all the total bets amounts are below 250 and an overwhelming majority of them are lower than 120. To get a view of the spread without the high leverage points, we plot the distribution of total bet amounts for bet amounts lower than 250. We see that total bets in the rage of 60 to 75 are most common, but overall there is a lot of spread in the distribution of the total bet amounts -- the standard deviation of total bet amounts is more than 40. 

<ins>Relationship to Hand Strength</ins>

In plotting the total bet amount against average hand strength, we see a non-linear relationship emerge. Up until a total bet amount of approximately 120 - - the range within which nearly all the total bet amounts lie -- we see a linear and positive relationship between total bet amount and average hand strength. After a total bet of 120, average hand strength seems to level off and doesn't change with the total bet amount.

##### Preflop, Flop, Turn and River Bet Amounts

These features were the amounts bet by a player in each particular stage of betting. 

<ins>Spread</ins>

In plotting the spread of the bet amount features we see that there are many high leverage points in the data. These will be addressed when diagnosing our logistic regression model fit. Nearly all the preflop and flop bets are below a bet amount of 60 and nearly all the turn and river bets are below a bet amount of 100. To get a view of the spread of the bet amounts by betting stage without the high leverage points, we restrict our distribution plots to  bets in the range of 0-70 for the preflop and flop bet amounts, and 0-100 for the turn and river bet amounts. 

All 4 features show modest spread, with the preflop bet amount showing the most. Clearly, the spreads of the bet amounts in the individual stages of betting are lower than the spread we saw with the total bet amount. The preflop and flop bet amount distributions show similar spread, but the average bet amount in the preflop is double the average in the flop. The turn and river bet amount distributions are nearly identical.

<ins>Relationship to Hand Strength</ins>

In the preflop, we can see a strong positive linear relationship between bet amount and average hand strength up until a bet amount of about 60. There is no clear relationship between preflop bet amount and average hand strength for bets larger than 60. There is a mild relationship between betting in the flop and average hand strength. Average hand strength increases modestly with flop bet amounts up until a size of 30, then average hand strength levels off with bet size until it starts decreasing for flop bets larger than about 40. In the turn and the river, there does not seem to be any relationship between bet amount and average hand strength. 






Given that prediction (of pocket hand strength) and not inference is the primary goal of our model, we will not concern ourselves with the presence of collinear predictor variables. We merely keep in mind that the predictor coefficient estimates generated by our model may have a degree of uncertainty depending on if some of the predictor variables are correlated. 

We assume that amateur players typically do not play hands close in hand strength value differently enough such that we could expect to accurately discriminate between say a strength 5 hand and a strength 6 hand on the basis of betting actions. To test this assumption, we look at how some of our features vary between adjacent hand strength values in the lower, middle and upper range of hand strength. 



