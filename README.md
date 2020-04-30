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

###### Spread

The bet and check count features in all stages of betting follow a bernoulli distribution. A check count of 0 in the preflop is far more common than a check count of 1, but the proportion of hands with a check count of 1 steadily increases until the river, where it is roughly half. In all stages of betting, the number of hands with bet counts of 0 are about double the number of hands with bet counts of 1. 

The raise and call count features in all stages of betting roughly follow a poisson distribution. From the preflop to the river, the spreads of the call count features decrease and the means decreases towards 0. From the spread of the preflop, flop, turn and river raise count features, we see that there are many high leverage points that could impact the fit of the logistic regression model. We will address these high leverage points later when diagnosing the logistic regression model fit. To better visualize the raise count features, we limit our plots to show the frequency of raise counts between 0 and 5 raises per betting stage. From the preflop to the river, the spreads of the raise count features decrease and the means decreases towards zero.

###### Relationship to Hand Strength

There is a strong negative relationship between checking in the preflop and average pocket hand strength. There is a moderately negative relationship between checking in the flop, turn and river stages and average hand strength.

There is a moderately positive relationship between betting in the flop and turn stages and average hand strength. There is a weakly positive relationship between betting in the river and average hand strength. 

There is a non-linear relationship between calling in the preflop and average hand strength. Average hand strength decreases from 0 to 1 call in the preflop, but thereafter remains stable with additional calls in the preflop. There is no relationship between calling in the flop and hand strength. There is a weakly positive linear relationship between call count in the turn and average hand strength. There is a moderately positive linear relationship between river call count and average hand strength. 

There are no clear relationships between the raise count features and average hand strength when we plot the total range of raise counts for each stage of betting. When we plot raise counts in the range of 0-5 raises against average hand strength, some relationships emerge. In the preflop stage, average hand strength increases significantly between 0 and 1 raise, but there is no clear relationship between average hand strength and additional raises thereafter. In the flop, average hand strength increases by a moderate amount between 0 and 1 raise, but there is no clear relationship between average hand strength and additional raises thereafter. In the turn, average hand strength goes up slightly between 0 and 1 raise, but there is no clear relationship between average hand strength and additional raises thereafter. In the river stage, there is no relationship between number of raises and average hand strength. 

##### Total Check/Bet/Call/Raise Counts

These features were counts of the total number of times a player checked, called, raised or bet in a given hand. Each feature was generated by summing the counts for that particulation betting action across all stages of betting. E.g. Total Call Count = preflop call count + flop call count + turn call count + river call count. 

###### Spread

All of the total betting action count features roughly follow a poisson distribution.  Each feature has roughly the same spread, with the total call count feature showing the most spread and the total bet count feature showing the least. Calling appears to be the most common betting action. There are a number of very high leverage points for the total raise count feature. We will address these later when diagnosing our logistc regression model fit. 

###### Relationship to Hand Strength

There is a strong negative linear relationship between a player's total check count and average hand strength. There is no clear relationship between total call count and average hand strength, but a call count of 0 seems to imply a slightly above average hand strength and calling more than 10 times seems to imply a below average hand strength. There is a moderate positive linear relationship between total bet count and average hand strength. 

There is no clear relationship between the total raise count feature and average hand strength when we plot the entire range of total raise count values. In restricting our plot to only show total raise counts in the range of 0 and 12 raises, we see a very non-linear relationship between average hand strength and the total raise count. Up until about 3 total raises, there is a positive and roughly linear relationship between hand strength and total raise count. After 3 total raises and until 8 raises, there is a negative and roughly linear relationship between average hand strength and total raise count. Ultimately though, a total raise count of 1 or higher seems to imply an above average hand strength. 

In plotting the action count features against hand strength, we see the following relationships:

There is a strong negative relationship between checking in the preflop and pocket hand strength. There is a moderate negative relationship between checking in the flop, turn and river stages and pocket hand strength.

There is a moderately negative relationship between calling in the preflop and pocket hand strength. There is no relationship between calling in the flop and hand strength. There is a moderately positive relationship between calling in the turn and river stages and hand strength. It seems that a moderate relationship between calling and hand strength does exist, contrary to our assumption.

There is a strongly negative relationship between going all-in in the preflop and hand strength, and a moderately negative relationship between going all-in in any other stage and hand strength. This relationship runs completely counter to our assumptions. Because there is a very, very small sample size of hands where a player went all-in, and intuitvely players would only go all-in when they possess stronger hands, it is likely that this relationship does not reflect the true relationhsip that exists between the variables. The all-in count features will be removed from the analysis as it seems like they are simply introducing noise into the model. 

There is a moderately positive relationship between betting in the flop and turn and hand strength. There is a weakly positive relationship between betting in the river and hand strength. 

For the sake of visualization, we plot raise counts in the range of 0-5 raises against hand strength. In the preflop stage, there is a strong positive relationship between raising or not raising and hand strength, but no clear relationship between hand strength and additional raises. In the flop, there is a moderately positive relationship between raising or not raising and hand strength, but no clear relationship between hand strength and additional raises. In the turn, there is a weakly positive relationship between raising or not raising and hand strength, but no clear relationship between hand strength and additional raises. In the river stage, there is no relationship between number of raises and hand strength. 

The next set of features we generated were the amounts bet by a player in each stage of betting. Our assumption was that higher bet amounts should be indicative of a stronger pocket and lower bet amounts should be indicative of a weaker pocket. 

Again, we see a number of very high leverage points in plotting the spread of the total bet amounts. Nearly all the total bets amounts are below 250 and an overwhelming majority of them are lower than 120. To get a view of the spread without the high leverage points, we plot the distribution of total bet amounts for bet amounts lower than 250. We see that total bets in the rage of 60 to 75 are most common, but overall there is a lot of spread in the distribution of the total bet amounts -- the standard deviation of total bet amounts is more than 40. 

In plotting the total bet amount against hand strength, we see a non-linear relationship emerge. Up until a total bet amount of approximately 120 - - the range within which nearly all the total bet amounts lie -- we see a linear and positive relationship between total bet amount and hand strength. After a total bet of 120, hand strength seems to level off and doesn't change with the total bet amount. 

As with the betting actions, we visualized the spread of the betting amounts and then plotted betting amounts against hand strengths to test our assumptions.

Similar to the case with raise count features, in plotting the spread of the bet amount features we see that there are many high leverage (and potential outlier) points in the data. Nearly all the preflop and flop bets are below a bet amount of 60 and nearly all the turn and river bets are below a bet amount of 100-110. For now we visualize the spread of the bet amount features in the ranges of 0-70 bet for the preflop and flop bet amounts, and 0-100 bet for the turn and river bet amounts. 

All 4 features show modest spread, with the preflop bet amount showing the most. Clearly, the spreads of the bet amounts in the individual stages of betting are lower than the spread we saw with the total bet amount. The preflop and flop bet amount distributions show similar spread, but the average bet amount in the preflop (20) is double the average in the flop (10). The turn and river bet amount distributions are nearly identical.

In the preflop, hand strength mostly seems to increase proportionately with the bet amount range. In the flop, not betting seems to imply a decently below average handstrength whereas betting implies an above average hand strength to some extent. There does not seem to be an obvious trend to changes in hand strength in response to incremental increases in the bet amount range above bets of 0. Bet amounts in the range of 80-100 do seem to imply a significantly above average hand strength, though. In the turn, betting an amount in the range of 0-10 seems to imply a below average handstrength while betting above that range implies an above average hand strength. There does not seem to be an obvious trend to changes in hand strength in response to incremental increases in the bet amount range above bets of 10. In the river, betting 0 seems to imply a slightly below average hand strength whereas betting above 0 seems to imply an average or slightly above average hand strength. 


Given that prediction (of pocket hand strength) and not inference is the primary goal of our model, we will not concern ourselves with the presence of collinear predictor variables. We merely keep in mind that the predictor coefficient estimates generated by our model may have a degree of uncertainty depending on if some of the predictor variables are correlated. 

The last set of features we generated were ratios of the amount a player bet in a stage of betting to the pot size at the beginning of that stage of betting. Our intuiton was that a given bet amount relative to a smaller pot is more indicative of a strong pocket than the same bet amount relative to a larger pot. 


We assume that amateur players typically do not play hands close in hand strength value differently enough such that we could expect to accurately discriminate between say a strength 5 hand and a strength 6 hand on the basis of betting actions. To test this assumption, we look at how some of our features vary between adjacent hand strength values in the lower, middle and upper range of hand strength. 





