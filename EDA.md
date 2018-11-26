### [EDA](/EDA.md)  | [References](/References.md) | 

## Introduction and EDA

### PROBLEM STATEMENT
How to predict the probability of a given twitter account to be a bot (bot: no direct human involvement in generating tweets)
(alternative: predicting the probability of a given tweet to be a bot tweet)
  1. Solve as a classification problem of classifying an account to be bot or not-bot
  2. Solve as an unsupervised problem of clustering twitter accounts into 2 (or several) clusters  
  
###  DATA
  1. Data Collection
    1.1. Approach 1: Collect and Labelling Bot / Real Users
      1.1.1. API (Tweets)

Size of Data
Time Period
Keywords (optional)
API (Account)
Followers
Use Botometer for labelling bot / real users
Approach 2: Collect Bot and Real Users Separately
Bot Accounts
Identify bots accounts - get their followers - get the followers of their followers (as the bots’ followers tend to be bots)
Manually-verified bot accounts
Real Users:
verified twitter profile accounts
Approach 3: Annotated Datasets
For instance: https://botometer.iuni.iu.edu/bot-repository/datasets.html
Description of Raw Data (Tweets)
An example of json + Description on Information Included
Describe Key Features (tweet)
favorite_count
name
screen_name
retweet
user.description
profile_background_url
profile_image_url
user.created_at
geo_enabled & location
Describe Key Features (account)
Account followers
Account friends
Account following
Account retweet count
Feature Engineering
Account tweeting rate and intervals
Account activity rate and intervals
Number of days since join
Challenging Features:
Top frequency one-word phrase and two-word phrase
Hashtags & frequencies
Identical responses to the same triggers
Shared websites & frequencies (of the websites)
Preliminary EDA
Plot graphs, distribution, correlations, etc.
Train / Test / Validation Set and Standardization

### EDA + PROPOSED MODELS
EDA using BotoMeter Classification as Response (known responses)
Baseline Model: Logistic Regression
Other Models
KNN
Random Forest
SVM (if data size is small)
Downside of Using BotoMeter’s Result as Classification: we might end up fitting Botometer’s algorithm

EDA Using Unsupervised Learning (unknown responses)
Baseline Model: K-Means, k=2
Gaussian Mixture Model (GMM)
Comparison with BotoMeter classification
