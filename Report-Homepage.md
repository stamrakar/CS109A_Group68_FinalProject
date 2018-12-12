---
title: CS109A Data Science Final Project
notebook: Report-Homepage.ipynb
nav_include: 1
---



### Machine Learning & Analysis for Twitter Bot Detection 
#### Contributors: Group #68: Nisrine Elhauz, Huan Liu, Fayzan Talpur, Samasara Tamrakar 
#### Harvard University, Fall 2018

<center><img style="float: center; padding-right:" src="https://raw.githubusercontent.com/fayzantalpur/DS1-Twitter-Bot-Detection/master/Images%20and%20Graphs/Twitter_Bot_Image.png" width="280"></center>

#### Motivation: 

The primary goal of this project is to detect Twitter bots by being able to differentiate the source of a tweet between a Twitter bot or human. To accomplish this, we must categorize the two main types of Twitter accounts that we are interested in for analysis. The first type being Twitter accounts that have direct human involvement but with a fake persona and specific purposes (e.g. spreading fake news, manipulation of online rating and review systems, etc.). The second type being “bot” Twitter accounts, that lack direct human involvement in generating tweets. The “bot” accounts, irrespective of their intention (whether to reduce human activity or malicious intent), will all be regarded as “bot” Twitter accounts as long as there is no direct involvement of a human. “No direct human involvement” can be defined as the actions of tweeting / retweeting / liking that are not the result of direct human operations.


#### Problem Statement:
How to predict the probability of a given twitter account to be a bot (bot: no direct human involvement in generating tweets)
(alternative: predicting the probability of a given tweet to be a bot tweet)
  1. Solve as a classification problem of classifying an account to be bot or not-bot
  2. Solve as an unsupervised problem of clustering twitter accounts into 2 (or several) clusters
