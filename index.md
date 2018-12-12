---
title: CS109A Data Science Final Project
---

### Machine Learning & Analysis for Twitter Bot Detection 
#### Contributors: Group #68: Nisrine Elhauz, Huan Liu, Fayzan Talpur, Samasara Tamrakar 
#### Harvard University, Fall 2018

![Image Text](https://i.ibb.co/KDd8fKN/Twitter-Logo-Blue.png)

<a href="https://imgbb.com/"><img src="https://i.ibb.co/KDd8fKN/Twitter-Logo-Blue.png" alt="Twitter-Logo-Blue" border="0" /></a>

![Image Text](https://cfcdnpull-creativefreedoml.netdna-ssl.com/wp-content/uploads/2017/06/Twitter-logo-2012.png)

#### Motivation: 

The primary goal of this project is to detect Twitter bots by being able to differentiate the source of a tweet between a Twitter bot or human. To accomplish this, we must categorize the two main types of Twitter accounts that we are interested in for analysis. The first type being Twitter accounts that have direct human involvement but with a fake persona and specific purposes (e.g. spreading fake news, manipulation of online rating and review systems, etc.). The second type being “bot” Twitter accounts, that lack direct human involvement in generating tweets. The “bot” accounts, irrespective of their intention (whether to reduce human activity or malicious intent), will all be regarded as “bot” Twitter accounts as long as there is no direct involvement of a human. “No direct human involvement” can be defined as the actions of tweeting / retweeting / liking that are not the result of direct human operations.


#### Problem Statement:

How to detect Twitter Bots using tweets data from Twitter developer API by using machine learning techniques. Our objective is to determine whether the source of tweets are from accounts that are bot users [1] or non-bot users [0].  (we define bot as: no direct human involvement in generating tweets) <br/>
  1. Start by collection data using Twitter API
  2. Perform feature engineering and preprocessing techniques to aggregate tweet features to account level features
  3. Use Data visualization to understand the trend and patterns. 
  4. Solve as a classification problem of classifying an account to be bot or not-bot
  5. Solve as an unsupervised problem of clustering twitter accounts into 2 (or several) clusters
  6. Conclude by comparing the models
