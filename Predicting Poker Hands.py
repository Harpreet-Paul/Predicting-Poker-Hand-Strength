#!/usr/bin/env python
# coding: utf-8

# # Predicting-Poker-Hand-Strength

# ## Introduction
# 
# ### Background
# 
# Texas Hold’em is a popular variant of Poker. Between 2-8 players sit at a table and each player is dealt two face-down cards (called their “pocket”). Across 4 stages of betting, 5 face-up cards (called “community cards”) are played on the table. After the last stage of betting, remaining players reveal their pocket cards and make the best 5 card hand possible, using a combination of their own cards and the face-up community cards. The player with the best 5 card hand collects all the money bet by each of the players (called the “pot”) for that round. 
# 
# ### Objective
# 
# The objective of this project was to predict the strength of a player’s two pocket cards, as determined by a hand strength formula known as the “Chen Formula”. 
# 
# ### Methods
# 
# Data was obtained from the UofAlberta IRC Poker Database, which contained game logs for thousands of online poker players. First, Logistic Regression, Multinomial Naive Bayes and AdaBoost (Boosted) Decision Tree models were fit to the full set of player data and compared on their prediction accuracy. Then, the dataset was segmented by dividing players into different groups based on playing style using K-Means clustering. The best performing model from the first step was fit to each segment of the dataset corresponding to a particular player type. 
# 
# ### Results
# 
# The AdaBoost (Boosted) Decision Tree model best fit the full set of player data, yielding a prediction accuracy of 41.5%. Fitting a separate AdaBoost (Boosted) Decision Tree model to each player type did not improve the prediction accuracy. 

# ## Data Processing and Visualization

# In[1]:


## Importing all the necessary libraries
import pandas as pd
import numpy as np
import glob as glob
import os as os
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt
import re
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
import math
from yellowbrick.cluster import KElbowVisualizer


# In[2]:


## This function uses the glob module to retrieve the filenames of all files from a particular path which match a provided regex expression. 
## Each retrieved file is converted into a dataframe and each of the resulting dataframes are stored in a list. All the dataframes in the list are then concatenated together.
def concatenate_dataframes(path, regex, *names):
    files  = glob.glob(path + regex)
    file_list = []
    for filename in files:
        df = pd.read_csv(filename, delim_whitespace=True, header= None, names = list(names))
        file_list.append(df)
    dataframe = pd.concat(file_list, axis=0, ignore_index=True)
    return dataframe


# In[3]:


## We run the 'concatenate_dataframes' function on all the player database files and hands database files.
## The resulting dataframes are merged together on the 'game_id' column. 
players = concatenate_dataframes(r'C:\Users\PAUL-2\Desktop\199504', "/pdb*", 'player_name', 'game_id', '#_of_players', 'position', 'preflop', 'flop', 'turn', 'river', 'bankroll', 'total_bet', 'winnings', 'card_1', 'card_2')
hands = concatenate_dataframes(r'C:\Users\PAUL-2\Desktop\199504', "/hdb*", 'game_id', 'dealer', 'hand#', '#players', 'flop', 'turn', 'river', 'showdown', 'board_card_1', 'board_card_2', 'board_card_3', 'board_card_4', 'board_card_5')        

all_hands = pd.merge(players, hands, on = 'game_id')


# ### Understanding the Raw Data

# This is what our original dataframe looks like:

# In[4]:


all_hands


# Each row represents a hand played by a given player in a given game. The important columns are interpreted as follows:
# 
# ##### 'preflop', 'flop_x', 'turn_x' and 'river_x'
# 
# The string of characters underneath these columns represent betting actions by the player in the different stages of betting. The betting action is encoded with a single character for each action:
# 
# |  Character   |    Action    |
# | ------------ | ------------ |
# | - | no action; player is no longer contesting pot |
# | B | blind bet |
# | f | fold |
# | k | check |
# | b | bet |
# | c | call |
# | r | raise |
# | A | all-in |
# | Q | quits game |
# | K | kicked from game |
# 
# ##### 'bankroll'
# 
# Total amount available for a player to bet at the beginning of a round. 
# 
# ##### 'card_1' and 'card_2'
# 
# If the player played until the showodown without folding, then their two pocket cards are shown under these two columns. 
# 
# ##### 'flop_y', 'turn_y', 'river_y' and 'showdown'
# 
# The first number is the number of players remaining in the game at the start of the stage. The second number is the total pot size at the start of the stage (the showdown is when players reveal their cards; no betting happens at this stage). 

# ### Data Cleaning and Generating Features/Target Variable

# In[5]:


## Dropping irrelevant columns from the dataframe
all_hands.drop(all_hands.columns[[2,3,10,13,14,15,20,21,22,23,24]], axis = 1, inplace=True)


# #### Target Variable

# In order to generate the target variable, we took the 'pocket' column and applied the Chen Formula, which maps each hand to a hand strength value between -2 and 20. Rows with either a NaN value (player folded before the showdown) or the correct string format (e.g. '8s 9c') for a pocket hand were kept and any rows with an abhorrent pocket hand input were removed. 

# In[6]:


## A new column is created which shows the two pocket cards of each hand, if available. 
all_hands["pocket"] = all_hands["card_1"] +" "+ all_hands["card_2"] 


# In[7]:


all_hands['pocket']


# In[8]:


## This function is used to ensure all the pocket hands are in a correct format.
## The function returns False (hand is not corrupt) if the hand is either an NaN value, or if there is a full match between the hand and the regex expression that represents the following pattern: '8d Qs'. 
## The function returns True otherwise (hand is corrupt).
def check_corrupt_hand(hand):
    if (pd.isnull(hand) == True) or (re.fullmatch(r'\w[csdh]\s\w[csdh]', hand) != None):
        return False
    else:
        return True 


# In[9]:


## We apply the 'check_corrupt_hand' function to the 'pocket' column and store the indices of any hands which are corrupt.
## The rows corresponding to the stored indices are dropped from the dataframe.
indices = all_hands[all_hands['pocket'].apply(check_corrupt_hand) == True].index
all_hands.drop(indices, inplace = True)


# In[10]:


## This function will take the player's two pocket cards as input and return the corresponding hand strength. 
## If the two pocket cards are unavailable, the function returns NaN. 
def chen_formula(hand):
    if pd.isnull(hand) == True:
        return np.nan
    else:
        card_1 = hand[0]
        suite_1 = hand[1]
        card_2 = hand[3]
        suite_2 = hand[4]
        values = {'A' : 10, 'K' : 8, 'Q' : 7, 'J' : 6, 'T' : 5 , '9' : 4.5 , '8' : 4, '7' : 3.5, '6' : 3 , '5' : 2.5 , '4' : 2 , '3' : 1.5 , '2' : 1}
        values_list = list(values)
        differences = {0:0, 1:0, 2:1, 3:2, 4:4, 5:5, 6:5, 7:5, 8:5, 9:5, 10:5, 11:5, 12:5}
        hand_strength = 0
        hand_strength += max(values[card_1], values[card_2])
        if card_1 == card_2:
            hand_strength = (hand_strength)*2
            if card_1 in ('2','3','4'):
                hand_strength = 5
        if suite_1 == suite_2:
            hand_strength += 2
        hand_strength  = hand_strength - (differences[abs(values_list.index(card_1) - values_list.index(card_2))])
        if (values_list.index(card_1) and values_list.index(card_2)) > 2:
            if abs(values_list.index(card_1) - values_list.index(card_2)) in [1,2]:
                hand_strength += 1
        hand_strength = round(hand_strength)
        return hand_strength


# In[11]:


## A new column will be created which shows the hand strength of the two pocket cards, if available. 
all_hands['hand_strength'] = all_hands['pocket'].apply(chen_formula)


# In[12]:


all_hands['hand_strength']


# #### Features

# ##### Preflop, Flop, Turn and River -- Check/Bet/Call/Raise Counts

# These features were counts of the number of times each betting action (call, check, raise and bet) was performed in each of the stages of betting (preflop, flop, turn and river). Any rows with null values for the betting actions were removed.

# In[13]:


## This function checks if there is a null value where a string encoding a player's actions should be. 
def check_corrupt_actions(actions):
    if pd.isnull(actions) == True:
        return True
    else:
        return False


# In[14]:


## We apply the check_corrupt_actions function to each betting actions column, and store the indices of any rows where there is a null value where the betting actions string should be. 
## The corresponding indices are dropped from the dataframe using the .drop function. 
for column in ['preflop', 'flop_x', 'turn_x', 'river_x']:
    indices = all_hands[all_hands[column].apply(check_corrupt_actions) == True].index
    all_hands.drop(indices, inplace = True)


# In[15]:


## This function takes as input a particular betting stage and betting action and will return a count of how many times that action was performed in that stage.
## Actions in a stage are encoded in a string. The string is converted to a list and we count the number of occurences of a character (like 'k', which maps onto the action 'check') in the list.
## For example, if during the preflop the player's actions were 'bc', the function would return a bet count of 1 and call count of 1.
def count_moves(stage, action):
    actions = {'check' : 'k', 'bet' : 'b', 'call': 'c', 'raise': 'r'}
    stage = list(stage)
    return stage.count(actions[action])


# In[16]:


## New columns will be made which show the counts of each action in each stage of betting. We do this by applying the 'count_moves' function to the string of actions in a stage of bettng.
## For example, 'preflop_bet_count' and 'flop_raise_count' and 'turn_check_count'.
for column in ['preflop', 'flop_x', 'turn_x', 'river_x']:
    for action in ['check', 'bet', 'call', 'raise']:
        all_hands[column + '_' + action + '_' + 'count'] = all_hands[column].apply(count_moves, args = (action,))


# In[17]:


all_hands.iloc[:,15:31]


# ##### Total Check/Bet/Call/Raise Counts

# These features were counts of the total number of times a player checked, called, raised or bet in a given hand. Each feature was generated by summing the counts for that particulation betting action across all stages of betting. E.g. Total Call Count = preflop call count + flop call count + turn call count + river call count. 

# In[18]:


## We create 4 new columns for the total check, bet, call and raise count of a hand. 
## Each column is created by summing the corresponding entries from the preflop, flop, turn and river betting action count columns. 
for action in ['check', 'bet', 'call', 'raise']:
    all_hands['total ' + action + ' ' + 'count'] = all_hands['preflop_' + action + '_' + 'count'] + all_hands['flop_x_' + action + '_' + 'count'] + all_hands['turn_x_' + action + '_' + 'count'] + all_hands['river_x_' + action + '_' + 'count']
    


# In[19]:


all_hands.iloc[:,31:35]


# ##### Total Bet

# This feature is the total amount that a player bet across all 4 stages of betting. No data processing was needed for this feature as it came pre-recorded in the dataset. 

# In[20]:


all_hands['total_bet']


# ##### Preflop, Flop, Turn and River Bet Amounts

# These features were the amounts bet by a player in each particular stage of betting (preflop, flop, turn and river). We first created a new dataframe, hands_without_fold, from the previous dataframe, all_hands, that only contains rows for which the two pocket cards were available as this is the data that our model will be trained on. We did not perform this step earlier because we wanted to generate the betting action count features on all hands, folded or not, for use in the player clustering that will be performed later.

# In[21]:


## Create a new dataframe that only features rows(hands) where the pocket cards available, ie. hands that a player played until the showdown without folding.
hands_without_fold = all_hands.dropna(subset = ['card_1', 'card_2'])


# In[22]:


hands_without_fold['hand_strength']


# In[23]:


## This function takes as input a string of the form 'players/pot' and will return the bet amount for each player by divding the pot by the number of players.
## This function is specifically for calculating the players bets in the preflop stage of betting. 
def preflop_bet(flop_y):
    preflop = flop_y.split('/', 1)
    preflop_pot = int(preflop[1])
    players = int(preflop[0])
    try:
        bet = (preflop_pot/players)
    except ZeroDivisionError:
        return np.nan
    return bet


# In[24]:


## A new column is created that shows the preflop bet amount for each hand. 
hands_without_fold['preflop_bet'] = hands_without_fold['flop_y'].apply(preflop_bet)


# In[25]:


## This function is similar to the 'preflop_bet' function. It takes as input strings of the form 'players/pot' from a prior stage of betting and a subsequent stage of betting.
## The bet amount for each player in the subsequent stage is calculated by subtracting the total pot at the end of the prior stage from the total pot at the end of the subsequent stage and dividing by the number of players in the subsequent stage.
## This function will be used to calculate player bet amounts for the flop, turn and river stages of betting. 
def bet_amount(stage1, stage2):
    stage2 = stage2.split('/', 1)
    stage2_bet_total = int(stage2[1])
    stage2_players = int(stage2[0])
    stage1 = stage1.split('/', 1)
    stage1_bet_total = int(stage1[1])
    stage2_pot = stage2_bet_total - stage1_bet_total

    try:
        player_bet = (stage2_pot/stage2_players)
    except ZeroDivisionError:
        return np.nan
    return player_bet


# In[26]:


## New columns will be created that show the player bet amounts for the flop, turn and river stages of betting for each hand.
for group in [['flop_y', 'turn_y'], ['turn_y', 'river_y'],['river_y', 'showdown']]:
    stage1 = group[0] 
    stage = group[0][0:-2]
    stage2 = group[1]
    hands_without_fold[stage + '_' + 'bet'] = hands_without_fold.apply(lambda x: bet_amount(x[stage1], x[stage2]), axis=1)


# In[27]:


hands_without_fold.loc[:,['preflop_bet', 'flop_bet', 'turn_bet', 'river_bet']]


# After generating all the new columns, we looked for rows with NaN values for any of the bet amounts. We see that there are rows which have NaN values for the preflop, flop or river bet. Rows with NaN values for the preflop or flop bet occur because one of the players went all-in and the game went straight to the showdown, causing the number of players and pot size at each stage of betting to be registered as "0/0" for each player's hand. Rows with NaN values for the river bet occur either because of the aforementioned reason or because the player quit the game during the river. The two pocket cards are still shown for a hand if the player quit at the river, which explains why these rows weren't dropped when we filtered out the folded hands. Hands where a player quit at the river or hands in a game where a player went all-in cannot be used in the analysis because they have an incomplete set of betting actions or betting amounts or both. Fortunately, such hands are only a small percentage of our dataset, so they can be dropped from the dataframe with little consequence to the model fit.

# In[28]:


hands_without_fold[pd.isnull(hands_without_fold['total_bet'])]


# In[29]:


hands_without_fold[pd.isnull(hands_without_fold['preflop_bet'])].head(100)


# In[30]:


hands_without_fold[pd.isnull(hands_without_fold['flop_bet'])].head(100)


# In[31]:


hands_without_fold[pd.isnull(hands_without_fold['turn_bet'])]


# In[32]:


hands_without_fold[pd.isnull(hands_without_fold['river_bet'])].head(100)


# In[33]:


## Using the df.dropna function to remove any rows for which any of the bet amount values are NaN. 
hands_without_fold.dropna(subset = ['total_bet', 'preflop_bet', 'flop_bet', 'turn_bet', 'river_bet'], inplace=True)


# ### Visualizing Features and Target Variable

# In[34]:


## A function which returns a bar plot of the distribution of the values in a particular column, i.e. essentially a histogram of the spread of a particular variable.
def plot_distribution_with_stats(df, column, xlabel, ylabel, title):
    array = df[column].to_numpy()
    array = array[~np.isnan(array)]
    labels, counts = np.unique(array, return_counts=True)
    plt.bar(labels, counts, align='center')
    plt.gca().set_xticks(labels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    print(column + ' ' + 'mean is ' + str(df[column].mean()))
    print(column + ' ' + 'median is ' + str(df[column].median()))
    print(column + ' ' + 'standard deviation is ' + str(df[column].std()))
    print('\n')
    


# #### Target Variable

# From the plot we can see that a hand strength of 5 is most common and that most hand strengths fall in the range between 2 and 10. The mean, median and standard deviation for the hand strengths are 6.13, 6.0 and 3.57 respectively.

# In[35]:


plot_distribution_with_stats(all_hands, 'hand_strength', 'hand strength', '# of hands', 'distribution of hand strengths')


# #### Features

# For each feature, we visualized it's spread and then plotted hand strength against that feature to see if there was any relationship between the two. All of our features were discrete variables, so we plotted average hand strength of all hands corresponding to a particular value of the feature against possible values of the feature. 

# ##### Preflop, Flop, Turn and River -- Check/Bet/Call/Raise Counts

# ###### Spread

# The check count features in all stages of betting follow a bernoulli distribution. A check count of 0 in the preflop is far more common than a check count of 1, but the proportion of hands with a check count of 1 steadily increases until the river, where it is roughly half.

# In[36]:


for column in ['preflop', 'flop_x', 'turn_x', 'river_x']:
    for action in ['check']:
        stage = re.match('[a-z]{2,7}', column).group(0)
        plot_distribution_with_stats(hands_without_fold, column + '_' + action + '_' + 'count', '# of' + ' ' + action + 's in' + ' ' + stage, '# of hands', 'distribution of' + ' ' + '# of' + ' ' + action + 's in' + ' ' + stage)
        


# Aside from the preflop, the bet count features in all stages of betting follow a bernoulli distribution and the number of hands with bet counts of 0 are about double the number of hands with bet counts of 1. In hindsight, it makes sense that all the preflop bet counts are 0 as there are small blind and big blind bets at the start of the preflop.

# In[37]:


for column in ['preflop', 'flop_x', 'turn_x', 'river_x']:
    for action in ['bet']:
        stage = re.match('[a-z]{2,7}', column).group(0)
        plot_distribution_with_stats(hands_without_fold, column + '_' + action + '_' + 'count', '# of' + ' ' + action + 's in' + ' ' + stage, '# of hands', 'distribution of' + ' ' + '# of' + ' ' + action + 's in' + ' ' + stage)


# Since the preflop bet count variable has 0 variance, it will be excluded from the model. 

# In[38]:


hands_without_fold.drop(columns = ['preflop_bet_count'], inplace = True)


# The call count features in all stages of betting roughly follow a poisson distribution. From the preflop to the river, the spreads of the call count features decrease and the means decrease towards 0.

# In[39]:


for column in ['preflop', 'flop_x', 'turn_x', 'river_x']:
    for action in ['call']:
        stage = re.match('[a-z]{2,7}', column).group(0)
        plot_distribution_with_stats(hands_without_fold, column + '_' + action + '_' + 'count', '# of' + ' ' + action + 's in' + ' ' + stage, '# of hands', 'distribution of' + ' ' + '# of' + ' ' + action + 's in' + ' ' + stage)


# From the spread of the preflop, flop, turn and river raise count features, we see that there are potential high leverage points that could impact the fit of the logistic regression model. Unfortunately, there aren't any python tools with which we can systematically detect and remove influential points from a multinomial logistic regression fit. We will leave these points in  the dataset for now and work around this problem later by fitting other models that are more robust to outliers and high leverage points. 
# 
# The raise count features in all stages of betting roughly follow a poisson distribution. From the preflop to the river, the spreads of the raise count features remain roughly the but and the means decrease towards zero.
# 
# 

# In[40]:


for column in ['preflop', 'flop_x', 'turn_x', 'river_x']:
    for action in ['raise']:
        stage = re.match('[a-z]{2,7}', column).group(0)
        plot_distribution_with_stats(hands_without_fold, column + '_' + action + '_' + 'count', '# of' + ' ' + action + 's in' + ' ' + stage, '# of hands', 'distribution of' + ' ' + '# of' + ' ' + action + 's in' + ' ' + stage)


# ###### Relationship to Hand Strength

# In[42]:


## A function which returns a plot (bar or line, in our case) of a particular discrete variable feature against the average value of a response variable at each value of that discrete variable feature.
def plot_feature_with_response_var(df, column, response_var, agg_func, plot_type, rot, ymax, y_increments, xlabel, ylabel, title):
    grouped = df.groupby([column]).agg({response_var: agg_func})
    grouped.plot(kind = plot_type, legend = False, grid = True, rot = rot)
    plt.gca().set_yticks(np.linspace(start = 0.0, stop = ymax, num = y_increments))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title (title)
    mean = df[response_var].mean()
    plt.axhline(mean, color = 'r', linestyle = '--')
    plt.show()
 
    


# There is a strong negative relationship between checking in the preflop and average pocket hand strength. There is a moderately negative relationship between checking in the flop, turn and river stages and average hand strength.

# In[43]:


for column in ['preflop', 'flop_x', 'turn_x', 'river_x']:
    for action in ['check']:
        stage = re.match('[a-z]{2,7}', column).group(0)
        plot_feature_with_response_var(hands_without_fold, column + '_' + action + '_' + 'count', 'hand_strength', 'mean', 'bar', 0, 8, 17, '# of' + ' ' + action + 's in' + ' ' + stage, 'hand strength', stage + ' ' + action + ' ' + 'count' + ' ' + 'vs. hand strength')


# There is a moderately positive relationship between betting in the flop and turn stages and average hand strength. There is a weakly positive relationship between betting in the river and average hand strength. 

# In[44]:


for column in ['flop_x', 'turn_x', 'river_x']:
    for action in ['bet']:
        stage = re.match('[a-z]{2,7}', column).group(0)
        plot_feature_with_response_var(hands_without_fold, column + '_' + action + '_' + 'count', 'hand_strength', 'mean', 'bar', 0, 8, 17, '# of' + ' ' + action + 's in' + ' ' + stage, 'hand strength', stage + ' ' + action + ' ' + 'count' + ' ' + 'vs. hand strength')


# For the most part, there is a linear and negative relationship between calling in the preflop and average hand strength. Average hand strength decreases from 0 to 3 calls in the preflop, but shoots back up between 3 and 4 calls. However, we have low confidence that average hand strength really does shoot back up after 3 calls as the sample size of hands with 3 or more calls in the preflop is very small.  There is no relationship between calling in the flop and hand strength. There is a weakly positive linear relationship between call count in the turn and average hand strength. There is a moderately positive linear relationship between river call count and average hand strength up until 2 calls. Between 2 and 3 calls, average hand strength goes back down, but again, we have low confidence in this result since the sample size of hands with 2 or more calls in the river is small. 

# In[45]:


for column in ['preflop', 'flop_x', 'turn_x', 'river_x']:
    for action in ['call']:
        stage = re.match('[a-z]{2,7}', column).group(0)
        plot_feature_with_response_var(hands_without_fold, column + '_' + action + '_' + 'count', 'hand_strength', 'mean', 'line', 0, 8, 17, '# of' + ' ' + action + 's in' + ' ' + stage, 'hand strength', stage + ' ' + action + ' ' + 'count' + ' ' + 'vs. hand strength')


# There are no clear relationships between the raise count features and average hand strength when we plot the total range of raise counts for each stage of betting. 

# In[46]:


for column in ['preflop', 'flop_x', 'turn_x', 'river_x']:
    for action in ['raise']:
        stage = re.match('[a-z]{2,7}', column).group(0)
        plot_feature_with_response_var(hands_without_fold, column + '_' + action + '_' + 'count', 'hand_strength', 'mean', 'line', 0, 8, 17, '# of' + ' ' + action + 's in' + ' ' + stage, 'hand strength', stage + ' ' + action + ' ' + 'count' + ' ' + 'vs. hand strength')


# When we plot raise counts in the range of 0-3 raises against average hand strength, some relationships emerge. In the preflop stage, there is a strong positive linear relationship between number of raises and average hand strength. In the flop, average hand strength increases linearly with the raise count up until 2 raises. It seems to decrease after 2 raises, but we have low confidence in this result because the sample size of hands with flop raise counts above 2 is small. In the turn, average hand strength goes up slightly between 0 and 1 raise, but there is no clear relationship between average hand strength and additional raises thereafter. We have low confidence in the relationship between average hand strength and raise count in the turn after 1 raise as the sample size of hands with more than 1 raise in the turn is small. In the river stage, there is no relationship between number of raises and average hand strength. 

# In[47]:


for column in ['preflop', 'flop_x', 'turn_x', 'river_x']:
    stage = re.match('[a-z]{2,7}', column).group(0)
    plot_feature_with_response_var(hands_without_fold[hands_without_fold[column + '_' + 'raise' + '_' + 'count'] <= 3], column + '_' + 'raise' + '_' + 'count', 'hand_strength', 'mean', 'line', 0, 9.5, 20, '# of' + ' ' + 'raises in' + ' ' + stage, 'hand strength', stage + ' ' + 'raise' + ' ' + 'count' + ' ' + 'vs. hand strength')
    
    


# ##### Total Check/Bet/Call/Raise Counts

# ###### Spread

# All of the total betting action count features roughly follow a poisson distribution. Each feature has roughly the same spread, with the total call count feature showing the most spread and the total bet count feature showing the least. Calling appears to be the most common betting action. There are also a number of potential high leverage points for the total raise count feature. 

# In[48]:


for column in ['total check count', 'total call count', 'total raise count', 'total bet count']:
    plot_distribution_with_stats(hands_without_fold, column, 'total # of ' + column[6:-6] + 's', '# of hands', 'distribution of total # of ' + column[6:-6] + 's in a hand')


# ###### Relationship to Hand Strength

# There is a strong negative linear relationship between a player's total check count and average hand strength. 
# 
# There is no clear relationship between total call count and average hand strength.
# 
# There is no clear relationship between the total raise count feature and average hand strength when we plot the entire range of total raise count values.
# 
# There is a moderate positive linear relationship between total bet count and average hand strength.
# 

# In[49]:


for column in ['total check count', 'total call count', 'total raise count', 'total bet count']:
    plot_feature_with_response_var(hands_without_fold, column, 'hand_strength', 'mean', 'line', 0, 8.0, 17, column, 'hand strength', column + ' vs. ' + 'hand strength')


# In restricting our plot to only show total raise counts in the range of 0 and 4 raises, we see a fairly strong positive linear relationship between average hand strength and the total raise count. 

# In[50]:


plot_feature_with_response_var(hands_without_fold[hands_without_fold['total raise count'] <= 4], 'total raise count', 'hand_strength', 'mean', 'line', 0, 9.5, 20, 'total raise count', 'hand strength', 'total raise count vs. hand strength')


# ##### Total Bet

# In[51]:


## A function which returns a bar plot of the distribution of the values in a particular column, i.e. essentially a histogram of the spread of a particular variable.
## This function is specifically for discrete variables with high cardinality that need to be broken down into bins.
## The np.linspace function is returning an array of evenly spaced numbers over the interval (min value of column of interest, max value of column of interest).
## The pd.cut function sorts the column's values into the bins defined by the 'bins' variable, which is the output of the np.linspace function. 
def plot_distribution_with_bins_with_stats(df, column, bin_increments, rot, figsize, round_to, y_increments, xlabel, ylabel, title):
    bins = np.linspace(start = df[column].astype(dtype=int).min(), stop = df[column].astype(dtype=int).max(), num = bin_increments)
    grouped = df.groupby(pd.cut(df[column].astype(dtype=int), bins=bins)).size()
    grouped.plot.bar(legend=False, rot = rot, figsize = figsize)
    plt.gca().set_yticks(np.linspace(start = 0.0, stop = (math.ceil((grouped.max())/round_to))*round_to, num = y_increments))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    print(column + ' ' + 'mean is ' + str(df[column].astype(dtype=int).mean()))
    print(column + ' ' + 'median is ' + str(df[column].astype(dtype=int).median()))
    print(column + ' ' + 'standard deviation is ' + str(df[column].astype(dtype=int).std()))
    print('\n')


# In[52]:


## We convert the 'total bet' column into an int datatype so we can perform arithmetic operations on it. 
hands_without_fold['total_bet'] = hands_without_fold['total_bet'].astype(dtype=int)


# ###### Spread

# We see a number of very high leverage points in plotting the spread of the total bet amounts. Nearly all the total bets amounts are below 180 and an overwhelming majority of them are lower than 120. 

# In[55]:


plot_distribution_with_bins_with_stats(hands_without_fold, 'total_bet', 20, 30, (10,5), 10000, 9, 'total bet', '# of hands', 'distribution of total bet amounts')


# To get a view of the spread without the high leverage points, we plot the distribution of total bet amounts for bet amounts lower than 180. We see that total bets in the rage of 65 to 75 are most common, but overall there is a lot of spread in the distribution of the total bet amounts -- the standard deviation of total bet amounts is around half the mean/median of the distribution.  

# In[59]:


plot_distribution_with_bins_with_stats(hands_without_fold[hands_without_fold['total_bet'].astype(dtype=int) <= 180], 'total_bet', 18, 30, (15,5), 1000, 21, 'total bet', '# of hands', 'distribution of total bet amounts')


# ###### Relationship to Hand Strength

# In[60]:


## A function which returns a plot (bar or line, in our case) of a particular discrete variable feature against the average value of a response variable at each range of values (bin) of that discrete variable feature.
## This function is specifically for discrete variables with high cardinality that need to be broken down into bins.
## The np.linspace function is returning an array of evenly spaced numbers over the interval (min value of column of interest, max value of column of interest).
## The pd.cut function sorts the column's values into the bins defined by the 'bins' variable, which is the output of the np.linspace function. 
def plot_feature_with_bins_with_response_var(df, column, bin_increments, response_var, agg_func, plot_type, rot, figsize, round_to, y_increments, xlabel, ylabel, title):
    bins = np.linspace(start = df[column].astype(dtype=int).min(), stop = df[column].astype(dtype=int).max(), num = bin_increments)
    grouped = df.groupby(pd.cut(df[column].astype(dtype=int), bins=bins)).agg({response_var: agg_func})
    grouped.plot(kind = plot_type, legend=False, grid= True, rot = rot, figsize = figsize)
    plt.gca().set_yticks(np.linspace(start = 0.0, stop = (math.ceil((grouped.max())/round_to))*round_to, num = y_increments))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    mean = df[response_var].mean()
    plt.axhline(mean, color = 'r', linestyle = '--')
    plt.show()


# There is no clear relationship between total bet amount and average hand strength when plotting the entire range of total bet amount values. 

# In[61]:


plot_feature_with_bins_with_response_var(hands_without_fold, 'total_bet', 20, 'hand_strength', 'mean', 'bar', 30, (15,5), 10, 21, 'total bet', 'hand strength', 'total bet amount vs. hand strength')


# In plotting total bet amounts in the range of 0-250 against average hand strength, we see a non-linear relationship emerge. Up until a total bet amount of approximately 120 - - the range within which nearly all the total bet amounts lie -- we see a linear and positive relationship between total bet amount and average hand strength. After a total bet of 120, average hand strength seems to mostly level off and doesn't change with the total bet amount.

# In[63]:


plot_feature_with_bins_with_response_var(hands_without_fold[hands_without_fold['total_bet'].astype(dtype=int) <= 180], 'total_bet', 18, 'hand_strength', 'mean', 'bar', 30, (20,5), 10, 21, 'total bet', 'hand strength', 'total bet amount vs. hand strength')


# In[64]:


plot_feature_with_bins_with_response_var(hands_without_fold[hands_without_fold['total_bet'].astype(dtype=int) <= 180], 'total_bet', 18, 'hand_strength', 'mean', 'line', 30, (20,5), 10, 21, 'total bet', 'hand strength', 'total bet amount vs. hand strength')


# ##### Preflop, Flop, Turn and River Bet Amounts

# ###### Spread

# In plotting the spread of the bet amount features we see that there are many high leverage points in the data. 
# 
# Nearly all the preflop and flop bets are below a bet amount of 50 and nearly all the turn and river bets are below a bet amount of 115. 

# In[73]:


for column in ['preflop_bet', 'flop_bet', 'turn_bet', 'river_bet']:
    plot_distribution_with_bins_with_stats(hands_without_fold, column, 20, 30, (10,5), 1000, 9, column[0:-4] + ' bet', '# of hands', 'distribution of ' + column[0:-4] + ' bet amounts')


# To get a view of the spread of the bet amounts by betting stage without the high leverage points, we restrict our distribution plots to show bets in the range of 0-115.
# 
# All 4 features show modest spread, with the turn bet amount showing the most. Clearly, the spreads of the bet amounts in the individual stages of betting are lower than the spread we saw with the total bet amount. The preflop and flop bet amount distributions show similar spread, but the average bet amount in the preflop is double the average in the flop. The turn and river bet amount distributions are nearly identical.

# In[76]:


for column in ['preflop_bet', 'flop_bet', 'turn_bet', 'river_bet']:
    plot_distribution_with_bins_with_stats(hands_without_fold[hands_without_fold[column]<= 115], column, 20, 30, (10,5), 1000, 11, column[0:-4] + ' bet', '# of hands', 'distribution of ' + column[0:-4] + ' bet amounts')


# ###### Relationship to Hand Strength

# Other than the preflop bet amount feature, there is no clear relationship between the bet amount features and average hand strength when looking at the entire range of bet amounts. 

# In[77]:


for column in ['preflop_bet', 'flop_bet', 'turn_bet', 'river_bet']:
    plot_feature_with_bins_with_response_var(hands_without_fold, column, 15, 'hand_strength', 'mean', 'bar', 30, (10,5), 10, 11, column[0:-4] + ' bet', 'hand strength', column[0:-4] + ' bet amount vs. hand strength')


# We restrict the bet amount vs average hand strength plots to show bets in the range of 0-50 for the preflop and flop. In the preflop, we can see a strong positive linear relationship between bet amount and average hand strength. There is no obvious relationship between average hand strength and flop bet amount.

# In[79]:


for column in ['preflop_bet', 'flop_bet']:
    plot_feature_with_bins_with_response_var(hands_without_fold[hands_without_fold[column]<= 50], column, 15, 'hand_strength', 'mean', 'bar', 30, (10,5), 10, 11, column[0:-4] + ' bet', 'hand strength', column[0:-4] + ' bet amount vs. hand strength')


# In[81]:


for column in ['preflop_bet', 'flop_bet']:
    plot_feature_with_bins_with_response_var(hands_without_fold[hands_without_fold[column]<= 50], column, 15, 'hand_strength', 'mean', 'line', 30, (10,5), 10, 11, column[0:-4] + ' bet', 'hand strength', column[0:-4] + ' bet amount vs. hand strength')


# We restrict the bet amount vs average hand strength plots to show bets in the range of 0-80 for the preflop and flop.There is no obvious relationship between turn or river bet amount and average hand strength. 

# In[80]:


for column in ['turn_bet', 'river_bet']:
    plot_feature_with_bins_with_response_var(hands_without_fold[hands_without_fold[column]<= 80], column, 15, 'hand_strength', 'mean', 'bar', 30, (10,5), 10, 11, column[0:-4] + ' bet', 'hand strength', column[0:-4] + ' bet amount vs. hand strength')


# In[82]:


for column in ['turn_bet', 'river_bet']:
    plot_feature_with_bins_with_response_var(hands_without_fold[hands_without_fold[column]<= 80], column, 15, 'hand_strength', 'mean', 'line', 30, (10,5), 10, 11, column[0:-4] + ' bet', 'hand strength', column[0:-4] + ' bet amount vs. hand strength')


# ## Model Fitting

# ### Regularized Logistic Regression

# The regularized logistic regression model can handle count data -- like our betting action counts and bet amounts -- as input, thus no feature conversion is needed. We will, however, perform a z-score standardization of our features. Our betting action count features take on much smaller values compared to our betting amount features. Therefore, we would expect our model to assign larger regression coefficients to these features. Without standardization, our model will be biased towards shrinking these coefficients in order to minimize the L-2 regularization cost function.
# 
# Aside from the standardization, we will fit the model to our data as is and use the 5-fold cross validation result as our baseline. 
# 
# 

# In[84]:


## We create an instance of the 'Pipeline' class,  passing inside instances of a Transformer class (e.g. 'StandardScaler') and an Estimator class (e.g. 'LogisticRegression') as arguments.
## Transformer class instances are optional depending on if the data requires some kind of transformation, but the last parameter must always be an Estimator class instance.
## The 'Pipeline' class sequentially applies the 'fit' and 'transform' methods of the Transformer class instance to a dataframe of the features and then applies the 'fit' method of the Estimator class instance to a dataframe of the features (x) and the response variable (y).
LR_hand_clf = Pipeline([('scale', StandardScaler()),('LR', LogisticRegression())])

## We create a dictionary mapping parameters of the Transformer/Estimator instances to a list of values that we would like a particular paramter to take.
parameters_LR = {'LR__C' : [1]}

## We create an instance of the 'GridSearchCV' class, passing inside the 'Pipeline' instance and the Transformer/Estimator parameters dictionary as arguments.
## The 'GridSearchCV' class instance performs an exhaustive search over the paramters values specified in the Transformer/Estimator parameters dictionary.
gs_LR_hand_clf = GridSearchCV(LR_hand_clf, parameters_LR, cv=5, n_jobs=-1, return_train_score = True)

## We make separate dataframes for our features (x) and our response variable (y).
x = hands_without_fold.iloc[:, list(range(16,39))]
y = hands_without_fold['hand_strength']

## We call the 'fit' method on our 'GridSearchCV' class instance, passing inside the x (features) and y (response variable) dataframes as arguments to fit our model to the data.
gs_LR_hand_clf = gs_LR_hand_clf.fit(x,y)


# In[85]:


## 'cv_results_' is an attribute of the 'GridSearchCV' class instance that displays the results of the cross-validation.
gs_LR_hand_clf.cv_results_


# Our baseline model correctly classified 19.3% of test set cases and 19.6% of training set cases. It performed only marginally better than the null model of predicting the median hand strength for each test case would have. Perhaps it is the case that players play hands that are only different by 1 or 2 hand strength categories very similarly, making predictions of a player's precise hand strength inherently very difficult. To test this hypothesis, we will look at the percent change in the average values of the features from one hand strength category to the next. 

# In[86]:


hands_without_fold.groupby(['hand_strength']).mean()


# In[87]:


hands_without_fold.groupby(['hand_strength']).mean().pct_change(periods=1).describe()


# In setting the percent change to be calculated between one hand strength category and the previous, we see that there is very little change in the average value of the features. The average % changes in the total bet and total check/bet/call/raise count features are about 4%, 7%, 6%, 0.6% and 9% respectively. This would make it difficult for our model to distinguish between adjacent hand strength categories. 

# In[88]:


hands_without_fold.groupby(['hand_strength']).mean().pct_change(periods=3).describe()


# When we set the percent change to be calculated between hand strength categories 3 levels apart, we see more considerable change in the average value of the features. The average % changes in the total bet and total check/bet/call/raise count features are now 13%, 21%, 17%, 4% and 34% respectively. If we group our hand strengths into bins that each contain 3 hand strength categories, there should be enough change in how the hands are played for our model to produce accurate yet useful predictions. 

# In[89]:


## A function that takes a hand strength value (integer) as input and returns an integer specifying the bin the hand strength belongs to.
## Each key-value pair in the dictionary is a hand strength (integer) mapped to an integer corresponding to a particular hand strength bin.
def hand_strength_bin(x):
    bins = {-2:0, -1:0, 0:0, 1:1, 2:1, 3:1, 4:2, 5:2, 6:2, 7:3, 8:3, 9:3, 10:4, 11:4, 12:4, 14:5, 16:5, 20:5}
    return bins[x]


# In[90]:


## We produce a 'hand strength bin' column by applying the 'hand_strength_bin' function to the 'hand strength' column.
hands_without_fold['hand_strength_bin'] = hands_without_fold['hand_strength'].apply(hand_strength_bin)


# In[91]:


LR_hand_clf = Pipeline([('scale', StandardScaler()),('LR', LogisticRegression())])
parameters_LR = {'LR__C' : [1]}
gs_LR_hand_clf = GridSearchCV(LR_hand_clf, parameters_LR, cv=5, n_jobs=-1, return_train_score = True)
x = hands_without_fold.iloc[:, list(range(16,39))]
y = hands_without_fold['hand_strength_bin']
gs_LR_hand_clf = gs_LR_hand_clf.fit(x,y)


# In[92]:


gs_LR_hand_clf.cv_results_


# After grouping the hand strength categories into 6 bins, our model accuracy has jumped considerably from 19.3% to 41.9%. The similarity between the training set scores and test set scores tells us that our model is underfitting, not overfitting the data. Hence, the strategies to improve model performance from herein are to increase the complexity of our model or to add new, more informative features.
# 
# Firstly, it's possible that the relationships between the log odds of a hand falling in a particular hand strength category and our features are not linear. To test this, we will introduce degree 2 polynomial features into the model (e.g. add a total bet amount^2 feature) and see if prediction accuracy improves.
# 
# Secondly, it's possible that we are missing some interaction terms in our model. In particular, there could be an interaction between bank roll, a feature we have not yet introduced into the model, and each of our predictors. For example, the total raise count feature may have a stronger relationship with the log odds of the response when bankroll is lower versus when it is higher. To test this, we will introduce the bankroll feature into the model and produce interactions terms between each of the features and all the other features and see if prediction accuracy improves.
# 
# Note: little to no colinearity between the independent variables is also an assumption of linear models. However, our primary objective with this analysis is prediction instead of inference, therefore will will not concern ourselves with handling colinearity in the data set for the logistic regression fit. 

# In[93]:


## 'polynomial' is an instance of the 'PolynomialFeatures' class.
## Inside the pipeline, the 'fit' and 'transform' methods of the 'PolynomialFeatures' class instance will be called to produce new columns for the features dataframe that include polynomial features and interaction terms.
LR_hand_clf = Pipeline([('polynomial', PolynomialFeatures()), ('scale', StandardScaler()), ('LR', LogisticRegression())])
parameters_LR = {'LR__C' : [1], 'polynomial__degree': [2]}
gs_LR_hand_clf = GridSearchCV(LR_hand_clf, parameters_LR, cv=5, n_jobs=-1, return_train_score = True)
x = hands_without_fold.iloc[:, [6] + list(range(16,39))]
y = hands_without_fold['hand_strength_bin']
gs_LR_hand_clf = gs_LR_hand_clf.fit(x,y)


# In[94]:


gs_LR_hand_clf.cv_results_


# Including the interaction terms and polynomial features into our model made virtually no difference to the prediction accuracy. With such a negligible difference in performance accuracy, it is better to accept the simpler model withthout the additional features as the better model for our data as it runs a lower risk of having overfit our training and test data. At this point, the only additional improvement we will seek to make is a fine-tuning of the regularization constant hyperparameter. With such similar training and test set prediction accuracies, we are still underfitting out data so it is possible that we have chosen too high of a regularization constant. 

# In[95]:


LR_hand_clf = Pipeline([('scale', StandardScaler()),('LR', LogisticRegression())])
parameters_LR = {'LR__C' : [1, 2, 5, 10]}
gs_LR_hand_clf = GridSearchCV(LR_hand_clf, parameters_LR, cv=5, n_jobs=-1, return_train_score = True)
x = hands_without_fold.iloc[:, list(range(16,39))]
y = hands_without_fold['hand_strength_bin']
gs_LR_hand_clf = gs_LR_hand_clf.fit(x,y)


# In[96]:


gs_LR_hand_clf.cv_results_


# The training and test set prediction accuracies are virtually identical across the entire hyperparameter space. Therefore, the choice of regularization constant was not the cause for underfitting.

# #### Best Logistic Regression Model Prediction Accuracy: ~ 42%.

# ### Multinomial Naive Bayes

# On one hand, it is unlikely that a Multinomial Naive Bayes model will yield a prediction accuracy greater than 42% when the more complex model, Logistic Regression, was still underfitting the data. However, there is a motivation for fitting this model to the data; Naive Bayes is more robust to high leverage points as compared to Logistic Regression because the Naive Bayes model parameters are not derived from a Maximum Likelihood Estimation. As such, it is possible that the Naive Bayes model produces more accurate predictions given that there are a number of high leverage points in our data set. 
# 
# The Multinomial Naive Bayes model requires non-continuous variables as input, thus we will need to bin our bet amount features into intervals. We have a few binning strategies to choose from: uniform, quantile and k-means. Our bet amount features have a large overall range, but most bet amounts are concentrated within a small ranges of values so the uniform and k-means binning strategies would end up producing a few bins with the majority of data points and many bins with hardly any data points at all. Concentrating most of the bet amounts into a single bin would render them useless in distinguishing between different hand strength categories. Therefore, we will use the quantile binning strategy that produces bins with an equal number of data points in each.  
# 
# We will perform a 5-fold cross validation to get an estimate of prediction accuracy on a test set. 
# 

# In[101]:


## We create an instance of the 'KBinsDiscretizer' class and call the 'fit' and 'transform' methods on the class instance, passing a dataframe of the bet amount features as an argument.
discrete_bet_amount_features = KBinsDiscretizer(n_bins=6, encode='ordinal', strategy='quantile').fit_transform(hands_without_fold[['total_bet', 'preflop_bet', 'flop_bet', 'turn_bet', 'river_bet']])


# In[105]:


## The object produced by calling the 'fit' and 'transform' methods is an array of all the bet amounts mapped onto an integer corresponding to a particular bin. 
discrete_bet_amount_features


# In[103]:


## We convert the array produced from the last step into a dataframe.
discrete_bet_amount_features_df = pd.DataFrame(discrete_bet_amount_features, columns = ['total_bet_bin', 'preflop_bet_bin', 'flop_bet_bin', 'turn_bet_bin', 'river_bet_bin'])


# In[104]:


discrete_bet_amount_features_df


# In[106]:


## we concatenate the 'hands_without fold' dataframe with the columns for the binned bet amounts. 
hands_without_fold_naive_bayes = pd.concat([hands_without_fold.reset_index(), discrete_bet_amount_features_df], axis=1)


# In[107]:


hands_without_fold_naive_bayes


# In[108]:


MNB_hand_clf = Pipeline([('MNB_hand_clf', MultinomialNB())])
parameters_MNB = {'MNB_hand_clf__fit_prior' : [True]}
gs_MNB_hand_clf = GridSearchCV(MNB_hand_clf, parameters_MNB, cv=5, n_jobs=-1, return_train_score = True)
x = hands_without_fold_naive_bayes.iloc[:, list(range(17,36)) + list(range(41,46))]
y = hands_without_fold_naive_bayes['hand_strength_bin']
gs_MNB_hand_clf = gs_MNB_hand_clf.fit(x,y)


# In[109]:


gs_MNB_hand_clf.cv_results_

Surprisingly, our 5-fold cross validation mean test set prediction accuracy was 36% -- not too far off from what we achieved with Logistic Regression. Our model is still underfitting the data, which is not surprising given the simplicity of the Naive Bayes hypothesis function. Given that our model is underfitting and not overfitting the data, feature selection to decrease model variance is not a concern right now. We cannot increase the model complexity and we do not have any additional features to incorporate into the model, therefore our only remaining strategy for improving our prediction accuracy is to modify our data such that it better meets the Naive Bayes model assumptions. The core assumption of Naive Bayes is that the features are independent from one another. To test how well our data meets this assumption, we will produce a feature correlation matrix using the spearman rank coefficient as our features do not come from normal distributions. 
# In[110]:


## Call the "corr' method on the dataframe of features.
corr = hands_without_fold_naive_bayes.iloc[:, list(range(17,36)) + list(range(41,46))].corr(method='spearman')

## Produce a correlation heatmap.
corr.style.background_gradient(cmap='coolwarm')


# From the correlation matrix, we can see that the following pairs of features have a correlation >= 0.7: flop/turn/river check count and total check count, flop/turn/river bet count and total bet count, flop/turn call count and total call count, turn bet bin and total bet bin, river check count and river bet bin. To mitigate the feature correlation in the dataset, we will drop the flop/turn/river check count, flop/turn/river bet count, flop/turn call count and turn bet bin features. 
# 
# While our intention is to deflate the importance of correlated features in our model, it is entirely possible, and perhaps likely, that in dropping these features, we will lose some important variance in our dataset and underfit our data further. We perform another 5-fold cross validation to find out. 

# In[111]:


## Dropping the correlated features from our dataframe.
hands_without_fold_nb_independent_features = hands_without_fold_naive_bayes.drop(['flop_x_check_count', 'turn_x_check_count', 'river_x_check_count', 'flop_x_bet_count', 'turn_x_bet_count', 'river_x_bet_count', 'flop_x_call_count', 'turn_x_call_count', 'turn_bet_bin'],  axis = 1)


# In[112]:


MNB_hand_clf = Pipeline([('MNB_hand_clf', MultinomialNB())])
parameters_MNB = {'MNB_hand_clf__fit_prior' : [True]}
gs_MNB_hand_clf = GridSearchCV(MNB_hand_clf, parameters_MNB, cv=5, n_jobs=-1, return_train_score = True)
x = hands_without_fold_nb_independent_features.iloc[:, list(range(17,32)) + [33,34,35,36]]
y = hands_without_fold_nb_independent_features['hand_strength_bin']
gs_MNB_hand_clf = gs_MNB_hand_clf.fit(x,y)


# In[113]:


gs_MNB_hand_clf.cv_results_


# Dropping the correlated features had virtually no effect on the mean test set prediction accuracy. 

# #### Best Multinomial Naive Bayes Model Prediction Accuracy: ~ 36%

# ### AdaBoost (Boosted) Decision Tree

# Similar to the situation with the Naive Bayes model, the AdaBoost (Boosted) Decision Tree model faces the challenge of yielding a prediction accuracy greater than 40% when the more complex model, Logistic Regression, was still underfitting the data. However, there are two motivations for fitting this model to the data: 1) AdaBoost (Boosted) Decision Trees are more robust to high leverage points as compared to Logistic Regression because the AdaBoost (Boosted) Decision Tree model does not rely on a Maximum Likelihood Estimation and 2) AdaBoost (Boosted) Decision Tree models do not require the dataset to meet a feature independence assumption.
# 
# The AdaBoost (Boosted) Decision Tree model can handle both continous and non-continous features as input, therefore no feature conversion is needed. 
# 
# We will perform a 5-fold cross validation to get an estimate of prediction accuracy on a test set. 

# In[114]:


ada_hand_clf = Pipeline([('ada', AdaBoostClassifier())])
parameters_ada = {'ada__n_estimators' : [100]}
gs_ada_hand_clf = GridSearchCV(ada_hand_clf, parameters_ada, cv=5, n_jobs=-1, return_train_score = True)
x = hands_without_fold.iloc[:, list(range(16,39))]
y = hands_without_fold['hand_strength_bin']
gs_ada_hand_clf = gs_ada_hand_clf.fit(x,y)


# In[115]:


gs_ada_hand_clf.cv_results_


# The AdaBoost (Boosted) Decision Tree model managed to perform just as well as the Logistic Regression model. It seems like the robustness to high leverage points and removal of the feature independance assumption made the difference. Though both models had the same prediction accuracy, we will consider the decision tree model our best one because it is the simpler of the two and thus less likely to have overfit our training and test data. 

# #### Best AdaBoost (Boosted) Decision Tree Model Prediction Accuracy: ~ 41.5%

# ## Clustering

# Thus far, we have tried to generate a single model that fits all the player data to predict hand strength. There is an obvious limitation with this strategy which is that the relationship between betting actions and amounts and hand strength likely varies with the type of player. The model that best predicts a particular player type's hand strength might be very different from the single model that best fits all the player types.
# 
# There are two main dimensions along which we can characterize poker players: loose/tight and passive/aggressive. A player's looseness or tightness is defined by the proportion of hands the player flops on in the preflop. A player's passiveness or aggressiveness is defined by the ratio of how many times that player has bet or raised to how many times they have called. 
# 
# We will score each player in the data set on each of these two dimensions and then perform a k-means clustering to group together players who have scored similarily to one another. We will then fit a seperate AdaBoost (Boosted) Decision Tree model to each of these different player groups and see how the prediction accuracies for each of the groups compare to the accuracy of the single AdaBoost (Boosted) Decision Tree model fit to the whole dataset. 
# 
# 

# In[117]:


## A function which takes a string encoding betting actions as input, and returns 1 if the characters "f" or "Q" are in the string, and 0 otherwise.
def if_fold_or_quit(preflop):
    if "f" in preflop or "Q" in preflop:
        return 1
    else:
        return 0


# In[118]:


## An 'if fold' column is created in the 'all_hands' dataframe by applying the 'if_fold_or_quit' function to the 'preflop' column.
all_hands['if_fold'] = all_hands['preflop'].apply(if_fold_or_quit)


# In[119]:


all_hands['if_fold']


# We will only include those players who have played atleast 50 hands into the clustering. We introduce this requirement because otherwise the sample size from which we would calculate a plyers loose/tight and passive/aggressive score would be too small to be representative of a player's play style. 

# In[120]:


## We group the 'all_hands' dataframe by player name.
## Groups (players) with less than 50 played hands (rows) are filtered out.
## We apply different aggregation methods on the different columns of the groupby object.
player_scores = all_hands.groupby(['player_name']).filter(lambda x: len(x) >= 50).groupby(['player_name']).agg(total_bet_count=pd.NamedAgg(column='total bet count', aggfunc=np.sum), total_call_count=pd.NamedAgg(column='total call count', aggfunc=np.sum), total_raise_count=pd.NamedAgg(column='total raise count', aggfunc=np.sum), total_fold_count=pd.NamedAgg(column='if_fold', aggfunc=np.sum), total_hand_count=pd.NamedAgg(column='if_fold', aggfunc='count')).reset_index()


# In[121]:


player_scores


# In[122]:


player_scores['passive/aggressive'] = (player_scores['total_bet_count'] + player_scores['total_raise_count'])/player_scores['total_call_count']
player_scores['loose/tight'] = player_scores['total_fold_count']/player_scores['total_hand_count']


# In[123]:


player_scores


# In[124]:


player_scores.describe()


# The range and spread of the passive/aggressive scores are much higher than the range and spread of the loose/tight scores because the loose/tight scores are necessarily restricted to values between 0 and 1. If we performed k-means clustering on the data as is, the clustering would be much more heavily influenced by the scores on the passive/aggressive dimension. Thus, we will standardize our data on these two dimensions so that both dimensions have equal influence on the results of the clustering.

# In[125]:


ax1 = player_scores.plot.scatter(x='passive/aggressive', y='loose/tight', c='DarkBlue')


# In producing a scatter plot of the player scores on the two dimensions, we see a couple of outlier points where the passive/aggressive scores are around 5, 12  and 32. We will remove these points from the dataset because their inclusion would unduly influence the results of the clustering. These points are so far removed from the normal range of values along this dimension that the k-means algorithm would likely produce cluster centers close to these points and the bulk of the data would end up being assigned to a single or a couple of clusters. 

# In[126]:


## Removing outlier points from the 'player_scores' dataframe.
player_scores_no_outlier = player_scores[player_scores['passive/aggressive'] <= 5]


# In[127]:


ax2 = player_scores_no_outlier.plot.scatter(x='passive/aggressive', y='loose/tight', c='DarkBlue')


# Looking at the same scatter plot without the outlier points, we can see that there aren't any distinct clusters, which makes our job of finding the optimal number of clusters difficult. In order to tackle this problem, we will implement the elbow method whereby we will plot the intra-cluster variation against the number of clusters (K) and look for the point after which the intra-cluster variation seems to level off with additional clusters.

# In[129]:


# We create an instance of the 'StandardScaler' class and call the 'fit' and 'transform' methods of the class on it, passing the dataframe of player scores as an argument.
X = player_scores_no_outlier[['loose/tight', 'passive/aggressive']]
X_std = StandardScaler().fit_transform(X)


# In[130]:


## The output is an array of the standardized player scores. 
X_std


# In[131]:


## Create an instance of the 'KMeans' class.
kmeans = KMeans()

## Create an instance of the 'KElbowVisualizer' class, passing the 'KMeans' class instance and a range of values for K (# of clusters) as arguments.
kmeans_visualizer = KElbowVisualizer(kmeans, k=(1,13))

## We call the 'fit' method on the 'KElbowVisualizer' class instance, which performs a K-means clustering of the standardized player scores for each value of K specified when we instantiated the class above.
kmeans_visualizer.fit(X_std)
kmeans_visualizer.show()


# From the elbow method plot, we can see that 3 clusters for player type are ideal. 

# In[132]:


kmeans = KMeans(n_clusters=kmeans_visualizer.elbow_value_)
kmeans.fit(X_std)


# In[133]:


## We call the 'labels_' attribute of the 'KMeans' class instance, which is an array of the cluster labels for each player.
## We produce a new column on the 'player_scores_no_outlier' dataframe that indicates which k-means cluster each player belongs to from this array.
player_scores_no_outlier['cluster'] = kmeans.labels_


# In[134]:


## We produce a groupby object by grouping the 'player_scores_no_outlier' dataframe by cluster label. 
player_scores_no_outlier_clusters = player_scores_no_outlier.groupby(['cluster'])

## We produce lists of all the player names belonging to each of the 3 k-means clusters.
cluster_0_players_names = list(player_scores_no_outlier_clusters.get_group(0)['player_name'])
cluster_1_players_names = list(player_scores_no_outlier_clusters.get_group(1)['player_name'])
cluster_2_players_names = list(player_scores_no_outlier_clusters.get_group(2)['player_name'])


# In[135]:


# We produce 3 separate dataframes, where each contains the hands played of only those players that fall within one of the 3 k-means clusters.
hands_without_fold_cluster_0 = hands_without_fold[hands_without_fold['player_name'].isin(cluster_0_players_names)]
hands_without_fold_cluster_1 = hands_without_fold[hands_without_fold['player_name'].isin(cluster_1_players_names)]
hands_without_fold_cluster_2 = hands_without_fold[hands_without_fold['player_name'].isin(cluster_2_players_names)]


# In[136]:


## Fitting the decision tree model to the cluster 0 player hand data. 
ada_hand_clf_cluster_0 = Pipeline([('ada', AdaBoostClassifier())])
parameters_ada_cluster_0 = {'ada__n_estimators' : [100]}
gs_ada_hand_clf_cluster_0 = GridSearchCV(ada_hand_clf_cluster_0, parameters_ada_cluster_0, cv=5, n_jobs=-1, return_train_score = True)
cluster_0_features = hands_without_fold_cluster_0.iloc[:, list(range(16,39))]
cluster_0_response = hands_without_fold_cluster_0['hand_strength_bin']
gs_ada_hand_clf_cluster_0 = gs_ada_hand_clf_cluster_0.fit(cluster_0_features, cluster_0_response)


# In[137]:


gs_ada_hand_clf_cluster_0.cv_results_


# In[138]:


## Fitting the decision tree model to the cluster 1 player hand data. 
ada_hand_clf_cluster_1 = Pipeline([('ada', AdaBoostClassifier())])
parameters_ada_cluster_1 = {'ada__n_estimators' : [100]}
gs_ada_hand_clf_cluster_1 = GridSearchCV(ada_hand_clf_cluster_1, parameters_ada_cluster_1, cv=5, n_jobs=-1, return_train_score = True)
cluster_1_features = hands_without_fold_cluster_1.iloc[:, list(range(16,39))]
cluster_1_response = hands_without_fold_cluster_1['hand_strength_bin']
gs_ada_hand_clf_cluster_1 = gs_ada_hand_clf_cluster_1.fit(cluster_1_features, cluster_1_response)


# In[139]:


gs_ada_hand_clf_cluster_1.cv_results_


# In[140]:


## Fitting the decision tree model to the cluster 2 player hand data. 
ada_hand_clf_cluster_2 = Pipeline([('ada', AdaBoostClassifier())])
parameters_ada_cluster_2 = {'ada__n_estimators' : [100]}
gs_ada_hand_clf_cluster_2 = GridSearchCV(ada_hand_clf_cluster_2, parameters_ada_cluster_2, cv=5, n_jobs=-1, return_train_score = True)
cluster_2_features = hands_without_fold_cluster_2.iloc[:, list(range(16,39))]
cluster_2_response = hands_without_fold_cluster_2['hand_strength_bin']
gs_ada_hand_clf_cluster_2 = gs_ada_hand_clf_cluster_2.fit(cluster_2_features, cluster_2_response)


# In[141]:


gs_ada_hand_clf_cluster_2.cv_results_


# Surprisingly, fitting separate decision tree models to each of the different player types did not increase prediction accuracy. The models fit to the cluster 0 and cluter 1 player data had the same accuracy as the single model fit to the whole dataset. For the cluster 2 players, the model yielded a prediction accuracy about 6% lower than the prediction accuracy for the model fit to the entire dataset. 

# ## Conclusion

# The AdaBoost (Boosted) Decision Tree model was the best fit to the full set of player data, yielding a prediction accuracy of 41.5%. Unfortunately, fitting a separate AdaBoost (Boosted) Decision Tree model to each player type did not improve the prediction accuracy. 
# 
# We recognize that predicting pocket hand strength from betting actions and betting amounts is an inherently difficult task given the nature of the game. Firstly, players bluff and play like they have a really strong hand even when they don't. Thus, more aggressive betting actions and higher bet amounts aren't always indicative of a stronger pocket hand. Secondly, players may have a weak pocket hand, but a strong overall 5-card hand when the pocket cards are combined with the board cards revealed at the flop, turn or river. Again, more aggressive betting actions and higher bet amounts wouldn't always always be indicative of a stronger pocket hand. 
# 
# Still, there are some avenues through which the performance of our models could be improved. In the future, opposing player's betting actions could be included as features to predict hand strength. 
