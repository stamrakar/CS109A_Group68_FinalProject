---
nav_include: 1
title: Test Notebook
notebook: Final_Project_TestFile.ipynb
---

## Machine Learning Analysis for Twitter Bot Detection

<img style="float: center; padding-right:" src="https://raw.githubusercontent.com/fayzantalpur/DS1-Twitter-Bot-Detection/master/Images%20and%20Graphs/Twitter_Bot_Image.png">

  
### CS109A: Harvard University, Fall 2018
### Final Project: Milestone 4 
### Contributors: Group #68: Nisrine Elhauz, Huan Liu, Fayzan Talpur, Samasara Tamrakar
___

<a id ='TOC'></a>
#### Table of Contents
1. [Introduction](#Introduction) <br/>
    1.1 [Motivation](#Motivation) <br/>
    1.2 [Problem Statement](#Problem-Statement) <br/>
2. [Data](#Data) <br/>
    2.1 [Data in Twitter API](#Data-in-Twitter-API) <br/>
    2.2 [Collection of Data](#Collection-of-Data) <br/>
    2.3 [Data Labelling: Using Botometer](#Labelling-Botometer) <br/>
    2.4 [Data Labelling: Manually](#Labelling-Manual) <br/>
    2.5 [Data Collection: Most Recent 100 Tweets per Each User](#Data-Collection-Recent) <br/>
    2.6 [Description of Raw Data (Tweets)](#Description-of-Raw-Data) <br/>  
3. [Exploratory Data Analysis](#Exploratory-Data-Analysis) <br/>
    3.1 [Data Wrangling and Cleansing](#Data-Wrangling-Cleansing) <br/>
    3.2 [Feature Engineering](#Feature-Engineering) <br/>
    3.3 [Important Features](#Important-Features) <br/>
    3.4 [Relations in Data](#Relations-in-Data) <br/>
    3.5 [Standardization](#Standardization) <br/>   
4. [Models](#Models) <br/>
    4.1 [Baseline Model](#Baseline-Model) <br/>
    4.2 [Logistic Regression](#Logistic-Regression) <br/>
    4.3 [KNN](#KNN) <br/>
    4.4 [Random Forest](#Random-Forest) <br/>
    4.5 [SVM](#SVM) <br/>
    4.6 [RNN](#RNN) <br/>
    4.7.[K-Means Clustering](#KMeans-Clustering)<br/>
5. [Results and Conclusion](#Results-and-Conclusion) <br/>
    5.1 [Summary of Results](#Summary-of-Results) <br/>
    5.2 [Noteworthy Findings](#Noteworthy-Findings) <br/>
    5.3 [Conclusion and Future Work](#Conclusion-and-Future-Work) <br/>
6. [Literature Review and Related Work](#Literature-Review-and-Related-Work) <br/>
    6.1 [Bots in the Twittersphere](#Bots-in-the-Twittersphere) <br/>
    6.2 [How Twitter Bots Help Fuel Political Feuds](#How-Twitter-Bots-Help-Fuel-Political-Feuds) <br/>
    6.3 [The spread of low-credibility content by social bots](#The-spread-of-low-credibility-content-by-social-bots) <br/>
    6.4 [Twitter Topic Modeling by Tweet Aggregation](#Twitter-Topic-Modeling-by-Tweet-Aggregation) <br/>
    6.5 [The tweepy Python Library](#tweepy-library) <br/>
    6.6 [Twitter's Developer Resources](#twitter-developer) <br/> 

___

[Back to TOC](#TOC) <br/>
<a id ='Introduction'></a>
### 1- Introduction

The main objective of the project is explore twitter dataset using twitter API and try to create a learning algorithim that can differentiate between bot and human.

[Back to TOC](#TOC) <br/>
<a id ='Motivation'></a>
#### 1.1 - Motivation <br/>

<mark> Some Text Here </mark>

[Back to TOC](#TOC) <br/>
<a id ='Problem-Statement'></a>
#### 1.2 - Problem Statement <br/>
How to detect Twitter Bots using tweets data from Twitter developer API by using machine learning techniques. Our objective is to determine whether the source of tweets are from accounts that are bot users [1] or non-bot users [0].  (we define bot as: no direct human involvement in generating tweets) <br/>
1. Start by collection data using Twitter API, then use feature engineering and preprocessing techniques to extract relevant data.
2. Use Data visualization to understand the trend and patterns. 
3. Solve as a classification problem of classifying an account to be bot or not-bot
4. Solve as an unsupervised problem of clustering twitter accounts into 2 (or several) clusters
5. Conclude by comparing the models



```python
#@title 
# Import Libraries, Global Options and Styles
import requests
from IPython.core.display import HTML
styles = requests.get(
    "https://raw.githubusercontent.com/Harvard-IACS/2018-CS109A/master/content/styles/cs109.css").text
HTML(styles)
%matplotlib inline

# Import libraries
import warnings
warnings.filterwarnings('ignore')

#import tweepy
import random
random.seed(112358)

%matplotlib inline
import numpy as np
import scipy as sp
import json as json
import pandas as pd
#import jsonpickle
import time


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from pandas.plotting import scatter_matrix
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV

import statsmodels.api as sm
from statsmodels.api import OLS

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import pandas as pd
pd.set_option('display.width', 1500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)

import seaborn.apionly as sns
sns.set(style="darkgrid")
sns.set_context("poster")
```


[Back to TOC](#TOC) <br/>
<a id ='Data'></a>
### 2 - Data <br/>

We started with three potential approaches to collect data for bot detection:

##### Approach 1: Collect Tweets then Label Bot / Real Users: 
*Approach* - Collect Tweets via Twitter API, extract accounts, then use Botometer API to label bot / real-user<br/>
*Pros* - flexible in collecting the specific data we are interested <br/>
*Cons* - using Botometer to label accounts might result in a fitting of Botometer's algorithms

##### Approach 2: Collect Bot and Real Users Separately
*Approach* - Manually select / verify bots account, use verified twitter profile accounts for the real user dataset, then use Twitter API to collect Tweets from the selected accounts <br/>
*Pros* - very accurate response (bot / real user) <br/>
*Cons* - time consuming and therefore small data size

##### Approach 3: Use Existing Annotated Dataset
*Approach* - Use existing datasets that have already labelled the tweets as bot / real-user <br/>
*Pros* - convenient <br/>
*Cons* - less flexible in collecting tweets with specific topic; results highly rely on the labelling accuracy

After evaluating the three approaches, we decided to collect our own tweets and use Botometer to label the bot / real-user. We decided to use the following approach to collect and label our data:

##### Step 1: Collection of Data : Collect over 1000 Tweets using Keywords
##### Step 2: Data Labelling: Using Botometer
##### Step 3: Data Labelling: Manual Verification for Each Account (Until Reach 50 Bots 50 Real Users)
##### Step 4: Data Collection - Get 3,200 (max) Most Recent Tweets from Each Verified Bot / User

[Back to TOC](#TOC) <br/>
<a id ='Data-in-Twitter-API'></a>
#### 2.1 - Data Source: Twitter API

Twitter offers a public API to interact with its platform. The API has different tiers representing different level of access to twitter platform. The API used in this project is the free twitter search API. The main points about this API are as following

1. Search tweets by keywods, usernames, locations, named places, etc.
2. its not clear if search criteria is limited to tweet itself or the username or a place.
3. The return data comes in  JSON format with different containers representing different aspects of tweet.

Since this is a free API, there are many limitations here as well. Some of these limitations are as following
1. Returned Data is a random sample of matching keywords.
2. Limited to 5,000 tweets per keyword.
3. Limit of 180 requests in a 15 minute period.
4. The respective data comes from last 7 days.

[Back to TOC](#TOC) <br/>
<a id ='Collection-of-Data'></a>
#### 2.2 - Collection of Data : Collect over 1000 Tweets using Keywords

We first collected some tweets that contains one of the following keywords that are likely to lead to controversial topics:  <br/>
>  1) Immigration <br/>
>  2) Brexit <br/>
>  3) bitcoin <br/>

We used keywords of more controversial topics as those are more likely to have non-obvious bots, which are more difficult to detect. We inferred this based on Baraniuk's article, "How Twitter Bots Help Fuel Political Feuds". 

We requested every 2 seconds for 100 tweets each for 15 request and received 1277 tweets.



```python
# http://www.tweepy.org/
import tweepy

# Replace the API_KEY and API_SECRET with your application's key and secret.
auth = tweepy.AppAuthHandler("API KEY", "SECRET")

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
 
if (not api):
    print ("Can't Authenticate")
    sys.exit(-1)
```




```python
# The following code was adapted from sample code provided by TFs / Profs for this project

def collect_tweets(maxTs, requestCount, filename):
    searchQuery = 'Immgration OR Brexit OR bitcoin'  # this is what we're searching for
    maxTweets = maxTs # some arbitrary large number
    tweetsPerQry = 100  # this is the max the API permits
    fName = filename # we'll store the tweets in a text file.

    # If results from a specific ID onwards are reqd, set since_id to that ID.
    # else default to no lower limit, go as far back as API allows
    sinceId = None

    # If results only below a specific ID are, set max_id to that ID.
    # else default to no upper limit, start from the most recent tweet matching the search query.
    max_id = -1
    error_count = 0

    request_count = 0

    tweetCount = 0
    
    
    print("Downloading max {0} tweets".format(maxTweets))
    with open(fName, 'w') as f:
        while tweetCount < maxTweets and request_count < requestCount:
            try:
                if (max_id <= 0):
                    if (not sinceId):
                        new_tweets = api.search(q=searchQuery, count=tweetsPerQry)
                    else:
                        new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                                since_id=sinceId)
                else:
                    if (not sinceId):
                        new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                                max_id=str(max_id - 1))
                    else:
                        new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                                max_id=str(max_id - 1),
                                                since_id=sinceId)
                if not new_tweets:
                    print("No more tweets found")
                    break
                for tweet in new_tweets:
                    f.write(jsonpickle.encode(tweet._json, unpicklable=False) +
                            '\n')
                tweetCount += len(new_tweets)
                print("Downloaded {0} tweets".format(tweetCount))
                max_id = new_tweets[-1].id
                request_count += 1
                time.sleep(2)
            except tweepy.TweepError as e:
                # Just exit if any error
                error_count += 1
                print("some error : " + str(e))
                time.sleep(2)
                if error_count >= 5:
                    print("too many errors ....break.")
                    break

    print ("Downloaded {0} tweets, Saved to {1}".format(tweetCount, fName))
```




```python
# we first collected a small sample data of about 1000 (which we will use botometer to encode)
collect_tweets(1500, 13, 'immigration_brexit_bitcoin.json')
```


A small sample dataset (from which we will manually verify 25 bots and 25 real user accounts): <br/>
We requested every 2 seconds for 100 tweet each for 13 reqyests and received 1277 tweets.



```python
# then we collect a larger sample (which we will use botometer to encode)
collect_tweets(12000, 120, 'immigration_brexit_bitcoin_extended.json')
```




```python
# load the file
raw_df_core = pd.read_json('immigration_brexit_bitcoin.json')
raw_df_extended = pd.read_json('immigration_brexit_bitcoin_extended.json', lines=True)
```




```python
# take a look at the separate data
display(raw_df_core.shape)
display(raw_df_extended.shape)
```




```python
# combine the two data sets
raw_df = pd.concat([raw_df_core, raw_df_extended], ignore_index=True)
```




```python
# take a look at the combined data
display(raw_df.head(5))
display(raw_df.shape)
```




```python
raw_df.shape
```




```python
# delete duplicate accounts
raw_df = raw_df.drop_duplicates(subset='id_str')
raw_df.shape
```




```python
# save as csv
raw_df.to_csv('immigration_brexit_bitcoin_full.csv')

# save as json
raw_df.to_json('immigration_brexit_bitcoin_full.json')
```


[Back to TOC](#TOC) <br/>
<a id ='Labelling-Botometer'></a>
#### 2.3 - Data Labelling: Using Botometer

<mark> add some text <mark>



```python
#load the data
raw_df = pd.read_json('immigration_brexit_bitcoin_full.json')
raw_df.head(5)
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>contributors</th>
      <th>coordinates</th>
      <th>created_at</th>
      <th>entities</th>
      <th>extended_entities</th>
      <th>favorite_count</th>
      <th>favorited</th>
      <th>geo</th>
      <th>id_str</th>
      <th>in_reply_to_screen_name</th>
      <th>in_reply_to_status_id</th>
      <th>in_reply_to_status_id_str</th>
      <th>in_reply_to_user_id</th>
      <th>in_reply_to_user_id_str</th>
      <th>is_quote_status</th>
      <th>lang</th>
      <th>metadata</th>
      <th>place</th>
      <th>possibly_sensitive</th>
      <th>quoted_status</th>
      <th>quoted_status_id</th>
      <th>quoted_status_id_str</th>
      <th>retweet_count</th>
      <th>retweeted</th>
      <th>retweeted_status</th>
      <th>source</th>
      <th>text</th>
      <th>truncated</th>
      <th>user</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>None</td>
      <td>2018-12-04 06:10:17</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [{'dis...</td>
      <td>None</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1069836265804259328</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>en</td>
      <td>{'iso_language_code': 'en', 'result_type': 're...</td>
      <td>None</td>
      <td>0.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>&lt;a href="https://ifttt.com" rel="nofollow"&gt;IFT...</td>
      <td>Coinpot Faucets That Pay Out Free Bitcoin Inst...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>None</td>
      <td>2018-12-04 06:10:16</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>None</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1069836265581961216</td>
      <td>brexit_politics</td>
      <td>1.069767e+18</td>
      <td>1.069767e+18</td>
      <td>8.981611e+17</td>
      <td>8.981611e+17</td>
      <td>False</td>
      <td>en</td>
      <td>{'iso_language_code': 'en', 'result_type': 're...</td>
      <td>None</td>
      <td>NaN</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>@brexit_politics Because the cocoa plant is na...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>NaN</td>
      <td>None</td>
      <td>2018-12-04 06:10:14</td>
      <td>{'hashtags': [{'indices': [37, 44], 'text': 'F...</td>
      <td>None</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1069836254169235456</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>en</td>
      <td>{'iso_language_code': 'en', 'result_type': 're...</td>
      <td>None</td>
      <td>NaN</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3</td>
      <td>False</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>RT @StuckOnCrypto: Phenomenal gains! #Factom $...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>100</th>
      <td>NaN</td>
      <td>None</td>
      <td>2018-12-04 06:09:22</td>
      <td>{'hashtags': [{'indices': [14, 21], 'text': 'B...</td>
      <td>None</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1069836035872428032</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>en</td>
      <td>{'iso_language_code': 'en', 'result_type': 're...</td>
      <td>None</td>
      <td>NaN</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2</td>
      <td>False</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>&lt;a href="http://forwardas.one" rel="nofollow"&gt;...</td>
      <td>RT @bbc5live: #Brexit – give us your speech to...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
    <tr>
      <th>1000</th>
      <td>NaN</td>
      <td>None</td>
      <td>2018-12-04 06:00:22</td>
      <td>{'hashtags': [], 'media': [{'display_url': 'pi...</td>
      <td>{'media': [{'display_url': 'pic.twitter.com/2w...</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1069833771778744320</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>tr</td>
      <td>{'iso_language_code': 'tr', 'result_type': 're...</td>
      <td>None</td>
      <td>0.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>&lt;a href="http://publicize.wp.com/" rel="nofoll...</td>
      <td>Money Button CEO’su: “Bitcoin problemi Bitcoin...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
    </tr>
  </tbody>
</table>
</div>





```python
# add account id to dataframe
raw_df['id'] = raw_df['user'].map(lambda d: d['id'])
```




```python
# set up botometer
# the code below was adapted from 
# https://github.com/IUNetSci/botometer-python

import botometer

mashape_key = "mashape key"
twitter_app_auth = {
    'consumer_key': 'CONSUMER KEY',
    'consumer_secret': 'CONSUMER SECRET',
    'access_token': 'API KEY',
    'access_token_secret': 'API SECRET',
  }
bom = botometer.Botometer(wait_on_ratelimit=True,
                          mashape_key=mashape_key,
                          **twitter_app_auth)
```




```python
# retrieve response objects from Botometer
botometer_results = {}
count = 0
for index, user_id in raw_df['id'].iteritems():
    botometer_results[index] = bom.check_account(user_id)
    print (count)
    count +=1
    time.sleep(3)
```




```python
# convert to series
botometer_series = pd.Series(botometer_results)
```




```python
# add results to a new column
raw_df['botometer_result'] = botometer_series
```




```python
# a quick look at botometer results
raw_df['botometer_result'][0]
```




```python
# extract universal score (botometer score)
raw_df['boto_univ'] = raw_df['botometer_result'].map(lambda s: s['cap']['universal'])
raw_df['boto_univ'].describe()
```




```python
# encode bot / non-bot via score of 0.5 threshold
threshold = 0.5
raw_df['class_boto'] = np.where(raw_df['boto_univ']>threshold, 1, 0)
```




```python
# examine number of 'bots' as identified by Botometer
sum(raw_df['class_boto'])
```




```python
# save as csv
raw_df.to_csv('immigration_brexit_bitcoin_boto.csv')

# save as json
raw_df.to_json('immigration_brexit_bitcoin_boto.json')
```


[Back to TOC](#TOC) <br/>
<a id ='Labelling-Manual'></a>
#### 2.4 - Data Labelling: Manual Verification for Each Account (Until Reach 50 Bots 50 Real Users)

We verified accounts by manually searching the username to check whether they were a bot or not using our best judgement. Usually, "Verified" Twitter accounts are often real users or obvious bots (e.g. Netflix). <br/>

Meanwhile, we only want to look at English tweets. <br/>

The following rules are used for manual Twitter account verification: <br/>
1) Constant retweets of media (especially only retweets no replies)  <br/>
2) Strong concentration on a specific topic <br/>
3) Significantly large number of tweets <br/>
4) Significantly large number of replying - not humanly possible speed <br/>
<mark> (add more verification rules) <mark>
  
During identification......<mark> add text <mark> <br/>
 
After examing the accounts associated with the tweets, we selected 50 bot accounts and 50 real user accounts that we feel confident in their classification.
  



```python
# load the core / small dataset, from which we will manually identify 25 bots and 25 non-bot accounts
raw_df_core = pd.read_json('immigration_brexit_bitcoin.json')
raw_df_core.head(5)
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>contributors</th>
      <th>coordinates</th>
      <th>created_at</th>
      <th>entities</th>
      <th>extended_entities</th>
      <th>favorite_count</th>
      <th>favorited</th>
      <th>geo</th>
      <th>id_str</th>
      <th>in_reply_to_screen_name</th>
      <th>in_reply_to_status_id</th>
      <th>in_reply_to_status_id_str</th>
      <th>in_reply_to_user_id</th>
      <th>in_reply_to_user_id_str</th>
      <th>is_quote_status</th>
      <th>lang</th>
      <th>metadata</th>
      <th>place</th>
      <th>possibly_sensitive</th>
      <th>quoted_status</th>
      <th>quoted_status_id</th>
      <th>quoted_status_id_str</th>
      <th>retweet_count</th>
      <th>retweeted</th>
      <th>retweeted_status</th>
      <th>source</th>
      <th>text</th>
      <th>truncated</th>
      <th>user</th>
      <th>botometer_result</th>
      <th>boto_univ</th>
      <th>class_boto</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>None</td>
      <td>2018-12-04 06:10:17</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [{'dis...</td>
      <td>None</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1069836265804259328</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>en</td>
      <td>{'iso_language_code': 'en', 'result_type': 're...</td>
      <td>None</td>
      <td>0.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>&lt;a href="https://ifttt.com" rel="nofollow"&gt;IFT...</td>
      <td>Coinpot Faucets That Pay Out Free Bitcoin Inst...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
      <td>{'cap': {'english': 0.5650534221, 'universal':...</td>
      <td>0.661264</td>
      <td>1</td>
      <td>389263463</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>None</td>
      <td>2018-12-04 06:10:16</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>None</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1069836265581961216</td>
      <td>brexit_politics</td>
      <td>1.069767e+18</td>
      <td>1.069767e+18</td>
      <td>8.981611e+17</td>
      <td>8.981611e+17</td>
      <td>False</td>
      <td>en</td>
      <td>{'iso_language_code': 'en', 'result_type': 're...</td>
      <td>None</td>
      <td>NaN</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>@brexit_politics Because the cocoa plant is na...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
      <td>{'cap': {'english': 0.0467737825, 'universal':...</td>
      <td>0.199883</td>
      <td>0</td>
      <td>1049565246711623680</td>
    </tr>
    <tr>
      <th>10</th>
      <td>NaN</td>
      <td>None</td>
      <td>2018-12-04 06:10:14</td>
      <td>{'hashtags': [{'indices': [37, 44], 'text': 'F...</td>
      <td>None</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1069836254169235456</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>en</td>
      <td>{'iso_language_code': 'en', 'result_type': 're...</td>
      <td>None</td>
      <td>NaN</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3</td>
      <td>False</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>RT @StuckOnCrypto: Phenomenal gains! #Factom $...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
      <td>{'cap': {'english': 0.0017254577000000001, 'un...</td>
      <td>0.002308</td>
      <td>0</td>
      <td>213374157</td>
    </tr>
    <tr>
      <th>100</th>
      <td>NaN</td>
      <td>None</td>
      <td>2018-12-04 06:09:22</td>
      <td>{'hashtags': [{'indices': [14, 21], 'text': 'B...</td>
      <td>None</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1069836035872428032</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>en</td>
      <td>{'iso_language_code': 'en', 'result_type': 're...</td>
      <td>None</td>
      <td>NaN</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2</td>
      <td>False</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>&lt;a href="http://forwardas.one" rel="nofollow"&gt;...</td>
      <td>RT @bbc5live: #Brexit – give us your speech to...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
      <td>{'cap': {'english': 0.4522790023, 'universal':...</td>
      <td>0.559158</td>
      <td>1</td>
      <td>1008465833046298625</td>
    </tr>
    <tr>
      <th>1000</th>
      <td>NaN</td>
      <td>None</td>
      <td>2018-12-04 06:00:22</td>
      <td>{'hashtags': [], 'media': [{'display_url': 'pi...</td>
      <td>{'media': [{'display_url': 'pic.twitter.com/2w...</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>1069833771778744320</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>tr</td>
      <td>{'iso_language_code': 'tr', 'result_type': 're...</td>
      <td>None</td>
      <td>0.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>None</td>
      <td>&lt;a href="http://publicize.wp.com/" rel="nofoll...</td>
      <td>Money Button CEO’su: “Bitcoin problemi Bitcoin...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
      <td>{'cap': {'english': 0.4827005207, 'universal':...</td>
      <td>0.199883</td>
      <td>0</td>
      <td>1189983314</td>
    </tr>
  </tbody>
</table>
</div>





```python
# to verify each user, we only need "screen_name"
raw_df_core['screen_name'] = raw_df_core['user'].map(lambda d: d['screen_name'])
```




```python
# add botometer score to the core dataset
raw_df_core = pd.merge(raw_df_core, raw_df[['class_boto','class_verified', 'boto_univ', 'screen_name']], left_on='screen_name', right_on='screen_name')
```




```python
# form a simple dataframe with only screen_name and Botometer score for references (so we can manually verify accounts)
# create 'class_verified for verified score'
raw_df_verify = raw_df_core.loc[:,['screen_name', 'class_boto', 'boto_univ','class_verified']]
```




```python
# delete duplicate rows
raw_df_verify.drop_duplicates(subset='screen_name')
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>screen_name</th>
      <th>class_boto</th>
      <th>boto_univ</th>
      <th>class_verified</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ICVeo</td>
      <td>1</td>
      <td>0.661264</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SamGuile2</td>
      <td>0</td>
      <td>0.199883</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>BadassJack101</td>
      <td>0</td>
      <td>0.002308</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100</th>
      <td>FAO_Scotbot</td>
      <td>1</td>
      <td>0.559158</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1000</th>
      <td>Habereglencee</td>
      <td>0</td>
      <td>0.199883</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1001</th>
      <td>cryptanalyser</td>
      <td>0</td>
      <td>0.104603</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1002</th>
      <td>boaleo</td>
      <td>0</td>
      <td>0.180303</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1003</th>
      <td>kasootsuuka</td>
      <td>0</td>
      <td>0.415826</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1004</th>
      <td>Diana_1aLectura</td>
      <td>0</td>
      <td>0.009037</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1005</th>
      <td>MasterNodesPro</td>
      <td>0</td>
      <td>0.221216</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1006</th>
      <td>CryptoGulp</td>
      <td>0</td>
      <td>0.038677</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1007</th>
      <td>morocotacoin</td>
      <td>0</td>
      <td>0.067479</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1008</th>
      <td>mattremains</td>
      <td>0</td>
      <td>0.180303</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1009</th>
      <td>CryptoNewswire</td>
      <td>0</td>
      <td>0.034358</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>101</th>
      <td>OshiWatanabe</td>
      <td>0</td>
      <td>0.008048</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1010</th>
      <td>FilthyRemainer</td>
      <td>0</td>
      <td>0.145776</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1012</th>
      <td>KingCrypto_</td>
      <td>0</td>
      <td>0.325413</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1013</th>
      <td>OOvBadviseurs</td>
      <td>0</td>
      <td>0.048615</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1014</th>
      <td>CriptoPasion</td>
      <td>0</td>
      <td>0.048615</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1015</th>
      <td>current_price_v</td>
      <td>0</td>
      <td>0.475301</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1016</th>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1017</th>
      <td>takeoff_tech</td>
      <td>1</td>
      <td>0.905228</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1018</th>
      <td>prathapkumarr</td>
      <td>0</td>
      <td>0.269683</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1019</th>
      <td>WorldCoinIndex</td>
      <td>0</td>
      <td>0.023812</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>102</th>
      <td>JmdsCourse</td>
      <td>0</td>
      <td>0.060559</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1020</th>
      <td>GerryJBrown</td>
      <td>0</td>
      <td>0.004304</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1021</th>
      <td>WhatIsBrexit</td>
      <td>0</td>
      <td>0.009037</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1022</th>
      <td>alanthelondoner</td>
      <td>0</td>
      <td>0.004304</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1023</th>
      <td>quickmarketcap</td>
      <td>0</td>
      <td>0.026940</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1024</th>
      <td>riskinfo</td>
      <td>1</td>
      <td>0.661264</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>96</th>
      <td>andvaccaro</td>
      <td>0</td>
      <td>0.002730</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>961</th>
      <td>jbhearn</td>
      <td>0</td>
      <td>0.003354</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>964</th>
      <td>webscale_bot</td>
      <td>0</td>
      <td>0.269683</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>967</th>
      <td>Mystic_lilac</td>
      <td>0</td>
      <td>0.067479</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>968</th>
      <td>KoinBulteni</td>
      <td>0</td>
      <td>0.034358</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>969</th>
      <td>Judge_Morville</td>
      <td>0</td>
      <td>0.005213</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>970</th>
      <td>cchapman1957</td>
      <td>0</td>
      <td>0.038677</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>971</th>
      <td>BeyondSimian</td>
      <td>0</td>
      <td>0.003626</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>973</th>
      <td>trevorb205</td>
      <td>0</td>
      <td>0.005213</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>974</th>
      <td>bigstushaw</td>
      <td>0</td>
      <td>0.008048</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>975</th>
      <td>ZAiFX</td>
      <td>0</td>
      <td>0.014566</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>976</th>
      <td>rayski1966</td>
      <td>0</td>
      <td>0.010165</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>977</th>
      <td>megacardtr</td>
      <td>0</td>
      <td>0.130656</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>978</th>
      <td>NexiruV2</td>
      <td>0</td>
      <td>0.475301</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>979</th>
      <td>billionairebman</td>
      <td>0</td>
      <td>0.244463</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>980</th>
      <td>BtcPulse</td>
      <td>0</td>
      <td>0.093608</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>981</th>
      <td>CoinWatcherBot</td>
      <td>0</td>
      <td>0.005213</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>982</th>
      <td>VittaGam</td>
      <td>0</td>
      <td>0.093608</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>983</th>
      <td>MeowchelK</td>
      <td>0</td>
      <td>0.004725</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>984</th>
      <td>robertjoseph</td>
      <td>0</td>
      <td>0.004725</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>985</th>
      <td>r_Buttcoin</td>
      <td>0</td>
      <td>0.003941</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>986</th>
      <td>Adnyuk</td>
      <td>0</td>
      <td>0.004725</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>989</th>
      <td>kriptoparahaber</td>
      <td>0</td>
      <td>0.130656</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>99</th>
      <td>John_Watkin5</td>
      <td>0</td>
      <td>0.016452</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>990</th>
      <td>precohoje</td>
      <td>0</td>
      <td>0.162308</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>992</th>
      <td>Guyston</td>
      <td>0</td>
      <td>0.009037</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>996</th>
      <td>Lendo_io</td>
      <td>0</td>
      <td>0.116936</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>997</th>
      <td>MuchBitcoin</td>
      <td>0</td>
      <td>0.002571</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>998</th>
      <td>GailMeeke</td>
      <td>1</td>
      <td>0.504016</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>999</th>
      <td>CryptoUpdates8</td>
      <td>1</td>
      <td>0.504016</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>988 rows × 4 columns</p>
</div>





```python
# save as csv (so we can manually verify and input results in excel)
raw_df_verify.to_csv('boto_to_verify.csv')
```


[Back to TOC](#TOC) <br/>
<a id ='Data-Collection-Recent'></a>
#### 2.5 - Data Collection - Get 200 (max) Most Recent Tweets from Each Account

For each tweet, we requested users' most recent 200 tweets using api.user_timeline via tweepy. <br/>



```python
# get the list of bot names and non-bot names
users_list = raw_df.loc[raw_df_verify['class_boto']==0]['screen_name'].tolist()
bots_list = raw_df.loc[raw_df_verify['class_boto']==1]['screen_name'].tolist()

# get the list of names
names = users_list + bots_list
```




```python
def get_tweets(names, fName, t_count, verify_df):
    # INPUT:
    # names: list of screen_name
    # fName: file name, .json
    # t_count: maximum number of tweets for each user
    # verify_df: a dataframe with 1) screen name; 2) class_bot; 3) class_verified; 4) boto_univ
    # OUTPUT:
    # tweets: pd dataframe of all the tweets
    
    # get tweets
    with open(fName, 'w') as f:
        tweetCount = 0
        for name in names:
            try:
                tweets = api.user_timeline(screen_name=name, count=t_count, tweet_mode='extended')
                for tweet in tweets:
                    f.write(jsonpickle.encode(tweet._json, unpicklable=False) + '\n')
                print("Downloaded {} tweets.".format(len(tweets)))
                tweetCount += len(tweets)
                time.sleep(5)
            except tweepy.TweepError as e:
                # Just exit if any error
                error_count += 1
                print("some error : " + str(e))
                if error_count >= 5:
                    print("too many errors ....break.")
                    break
    print ("Downloaded {0} tweets, Saved to {1}".format(tweetCount, fName))
    
    # add botometer 
    tmp_df = pd.read_json(fName, lines=True)
    tmp_df['user_screen_name'] = tmp_df['user'].map(lambda d: d['screen_name'])
    tmp_boto_df = pd.merge(tmp_df, verify_df[['class_boto','class_verified', 'boto_univ', 'screen_name']], left_on='user_screen_name', right_on='screen_name')
    tmp_boto_df = tmp_boto_df.drop(columns=['user_screen_name'])
    return tmp_boto_df
```




```python
# get max 200 tweets for each user
tweets_df = get_tweets(names=names, fName='tweets.json', t_count=200, verify_df=raw_df_verify) #the fName and corresponding data will be updated later
```


[Back to TOC](#TOC) <br/>
<a id ='Description-of-Raw-Data'></a>
#### 2.6 - Description of Raw Data (Tweets)

<mark>(TO BE UPDATED)<mark> <br/>
  
<mark> Updated some field descriptions below from [Tweets Data Dictionary](https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/tweet-object): <mark> <br/>
<mark> add more if necessary <mark>
   
 
Among the data we collected as json files from the tweepy api.search, the data set contains objects such as ‘user’, which includes comprehensive information of user accounts. Additionally, detailed information about each individual tweet was also collected.

The following describes some of the fields from the raw data collected, that we thought were relevant:

**Tweet Data**

> *created_at*: UTC time when this Tweet was created  <br/>
>  *display_text_range*: delineates different sections of body for mentions/tweets/entities <br/>
>  *extended_entities*: contains a single media array of media objects <br/>
>  *favorite_count*: approximately how many times Tweet has been liked by Twitter users <br/>
>  *favorited*: Indicates whether this Tweet has been liked by the authenticating user <br/>
>  *full_text*: actual UTF-8 text of the status update  <br/>
>  *id*: The integer representation of the unique identifier for this Tweet <br/>
>  *in_reply_to_screen_name*: If Tweet is a reply, field will contain the screen name of the original Tweet’s author<br/>
>  *is_quote_status*: Indicates whether this is a Quoted Tweet  <br/>
>  *possibly_sensitive*: only surfaces when a Tweet contains a link. The meaning of the field doesn’t pertain to the Tweet content itself, but instead it is an indicator that the URL contained in the Tweet may contain content or media identified as sensitive content.<br/>
>  *retweet_count*: Number of times this Tweet has been retweeted <br/>
>  *retweeted*: Indicates whether this Tweet has been Retweeted by the authenticating user  <br/>
>  *retweeted_status*: Retweets can be distinguished from typical Tweets by the existence of a retweeted_status attribute. This attribute contains a representation of the original Tweet that was retweeted <br/>
>  *truncated*: Indicates whether value of text parameter was truncated, e.g., as a result of a retweet exceeding the original Tweet text length limit of 140 characters. Truncated text will end in ellipsis, like this ... <br/>
>  *screen_name*: screen name, handle, or alias of user who posted the Tweet <br/>  
   
   

**User Data**
  
>  *user_created_at*: UTC datetime that the user account was created on Twitter  <br/>
>  *user_description*:  user-defined UTF-8 string describing their account <br/>
>  *user_favourites_count*: number of Tweets this user has liked in the account’s lifetime <br/>
>  *user_followers_count*:  number of followers this account currently has<br/>   
>  *user_friends_count*: number of users this account is following (AKA their “followings”)<br/>
>  *user_listed_count*: number of public lists that this user is a member of<br/>
>  *user_location*: The user-defined location for this account’s profile. Not necessarily a location, nor machine-parseable<br/>
>  *user_name*: name of the user, as they’ve defined it. Not necessarily a person’s name<br/>
>  *user_profile_background_image_url*: HTTP-based URL pointing to the background image the user has uploaded for their profile <br/>   
>  *user_screen_name*: screen name, handle, or alias that this user identifies themselves with. screen_names are unique but subject to change, use id_str as a user identifier if possible.<br/>
>  *user_statuses_count*: number of Tweets (including retweets) issued by the user<br/>
>  *user_description_len*: length of user_description field string <br/>



Botometer’s response object returned bot-scores in various different categories. This included categories such as the Complete Automation Probability, which determines how likely the account is a bot. The bot-scores, on a scale, determines if a given account is closer to a bot or a real user. Then. from the json data we gathered through the tweepy api.search, we extracted user account id to retrieve their corresponding Botometer scores. 


[Back to TOC](#TOC) <br/>
<a id ='Exploratory-Data-Analysis'></a>
### 3.1 - Exploratory Data Analysis
Include only features with value and drop features with mostly null value

[Back to TOC](#TOC) <br/>
<a id ='Data-Wrangling'></a>
#### 3.1 - Data Wrangling & Cleansing

parsing features, Include only features with value and drop features with mostly null value



```python
# read the dataset
tweets_df = pd.read_json('tweets.json', lines=True)
```




```python
# explode 'entities', 'user'
# although it would be interesting to see 'retweeted_status', it might be a bit too complicated
# especially when the # of reweets of the retweeted post is availabel directly ('retweet_count')
# it might be more efficient just to add a new column showing if a tweet contains retweet
def explode(df):
    dicts = ['user', 'entities']
    for d in dicts:
        keys = list(df.iloc[0]['user'].keys())
        for key in keys:
            df[str(d) + '_' + key] = df[d].map(lambda x: x[key] if key in x and x[key] else None)    
    return df
```




```python
# parse
tweets_df = explode(tweets_df)
tweets_df.head()
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>contributors</th>
      <th>coordinates</th>
      <th>created_at</th>
      <th>display_text_range</th>
      <th>entities</th>
      <th>extended_entities</th>
      <th>favorite_count</th>
      <th>favorited</th>
      <th>full_text</th>
      <th>geo</th>
      <th>id</th>
      <th>id_str</th>
      <th>in_reply_to_screen_name</th>
      <th>in_reply_to_status_id</th>
      <th>in_reply_to_status_id_str</th>
      <th>in_reply_to_user_id</th>
      <th>in_reply_to_user_id_str</th>
      <th>is_quote_status</th>
      <th>lang</th>
      <th>place</th>
      <th>possibly_sensitive</th>
      <th>quoted_status</th>
      <th>quoted_status_id</th>
      <th>quoted_status_id_str</th>
      <th>quoted_status_permalink</th>
      <th>retweet_count</th>
      <th>retweeted</th>
      <th>retweeted_status</th>
      <th>source</th>
      <th>truncated</th>
      <th>user</th>
      <th>withheld_in_countries</th>
      <th>user_contributors_enabled</th>
      <th>user_created_at</th>
      <th>user_default_profile</th>
      <th>user_default_profile_image</th>
      <th>user_description</th>
      <th>user_entities</th>
      <th>user_favourites_count</th>
      <th>user_follow_request_sent</th>
      <th>user_followers_count</th>
      <th>user_following</th>
      <th>user_friends_count</th>
      <th>user_geo_enabled</th>
      <th>user_has_extended_profile</th>
      <th>user_id</th>
      <th>user_id_str</th>
      <th>user_is_translation_enabled</th>
      <th>user_is_translator</th>
      <th>user_lang</th>
      <th>...</th>
      <th>user_protected</th>
      <th>user_screen_name</th>
      <th>user_statuses_count</th>
      <th>user_time_zone</th>
      <th>user_translator_type</th>
      <th>user_url</th>
      <th>user_utc_offset</th>
      <th>user_verified</th>
      <th>entities_contributors_enabled</th>
      <th>entities_created_at</th>
      <th>entities_default_profile</th>
      <th>entities_default_profile_image</th>
      <th>entities_description</th>
      <th>entities_entities</th>
      <th>entities_favourites_count</th>
      <th>entities_follow_request_sent</th>
      <th>entities_followers_count</th>
      <th>entities_following</th>
      <th>entities_friends_count</th>
      <th>entities_geo_enabled</th>
      <th>entities_has_extended_profile</th>
      <th>entities_id</th>
      <th>entities_id_str</th>
      <th>entities_is_translation_enabled</th>
      <th>entities_is_translator</th>
      <th>entities_lang</th>
      <th>entities_listed_count</th>
      <th>entities_location</th>
      <th>entities_name</th>
      <th>entities_notifications</th>
      <th>entities_profile_background_color</th>
      <th>entities_profile_background_image_url</th>
      <th>entities_profile_background_image_url_https</th>
      <th>entities_profile_background_tile</th>
      <th>entities_profile_banner_url</th>
      <th>entities_profile_image_url</th>
      <th>entities_profile_image_url_https</th>
      <th>entities_profile_link_color</th>
      <th>entities_profile_sidebar_border_color</th>
      <th>entities_profile_sidebar_fill_color</th>
      <th>entities_profile_text_color</th>
      <th>entities_profile_use_background_image</th>
      <th>entities_protected</th>
      <th>entities_screen_name</th>
      <th>entities_statuses_count</th>
      <th>entities_time_zone</th>
      <th>entities_translator_type</th>
      <th>entities_url</th>
      <th>entities_utc_offset</th>
      <th>entities_verified</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-12-06 09:42:52</td>
      <td>[0, 140]</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>RT @David_Dowse1: Have you seen @jeremycorbyn ...</td>
      <td>NaN</td>
      <td>1070614540746977281</td>
      <td>1070614540746977280</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>en</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>278</td>
      <td>False</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
      <td>NaN</td>
      <td>None</td>
      <td>Tue Sep 16 11:29:33 +0000 2014</td>
      <td>True</td>
      <td>None</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>{'description': {'urls': []}}</td>
      <td>13997.0</td>
      <td>None</td>
      <td>3080</td>
      <td>None</td>
      <td>3073.0</td>
      <td>None</td>
      <td>None</td>
      <td>2772143657</td>
      <td>2772143657</td>
      <td>None</td>
      <td>None</td>
      <td>en</td>
      <td>...</td>
      <td>None</td>
      <td>do_mck</td>
      <td>108501</td>
      <td>None</td>
      <td>none</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-12-06 09:42:45</td>
      <td>[0, 139]</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>RT @MrHickmott: In your street there lives som...</td>
      <td>NaN</td>
      <td>1070614514238922753</td>
      <td>1070614514238922752</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>en</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1586</td>
      <td>False</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
      <td>NaN</td>
      <td>None</td>
      <td>Tue Sep 16 11:29:33 +0000 2014</td>
      <td>True</td>
      <td>None</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>{'description': {'urls': []}}</td>
      <td>13997.0</td>
      <td>None</td>
      <td>3080</td>
      <td>None</td>
      <td>3073.0</td>
      <td>None</td>
      <td>None</td>
      <td>2772143657</td>
      <td>2772143657</td>
      <td>None</td>
      <td>None</td>
      <td>en</td>
      <td>...</td>
      <td>None</td>
      <td>do_mck</td>
      <td>108501</td>
      <td>None</td>
      <td>none</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-12-06 09:42:25</td>
      <td>[0, 140]</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>RT @JumMurphy: The Unite general secretary, Le...</td>
      <td>NaN</td>
      <td>1070614428859727872</td>
      <td>1070614428859727872</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>en</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>False</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
      <td>NaN</td>
      <td>None</td>
      <td>Tue Sep 16 11:29:33 +0000 2014</td>
      <td>True</td>
      <td>None</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>{'description': {'urls': []}}</td>
      <td>13997.0</td>
      <td>None</td>
      <td>3080</td>
      <td>None</td>
      <td>3073.0</td>
      <td>None</td>
      <td>None</td>
      <td>2772143657</td>
      <td>2772143657</td>
      <td>None</td>
      <td>None</td>
      <td>en</td>
      <td>...</td>
      <td>None</td>
      <td>do_mck</td>
      <td>108501</td>
      <td>None</td>
      <td>none</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-12-06 09:42:07</td>
      <td>[0, 139]</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>RT @ArthurStramash: “The Scottish Parliament, ...</td>
      <td>NaN</td>
      <td>1070614351076298753</td>
      <td>1070614351076298752</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>en</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>37</td>
      <td>False</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
      <td>NaN</td>
      <td>None</td>
      <td>Tue Sep 16 11:29:33 +0000 2014</td>
      <td>True</td>
      <td>None</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>{'description': {'urls': []}}</td>
      <td>13997.0</td>
      <td>None</td>
      <td>3080</td>
      <td>None</td>
      <td>3073.0</td>
      <td>None</td>
      <td>None</td>
      <td>2772143657</td>
      <td>2772143657</td>
      <td>None</td>
      <td>None</td>
      <td>en</td>
      <td>...</td>
      <td>None</td>
      <td>do_mck</td>
      <td>108501</td>
      <td>None</td>
      <td>none</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-12-06 09:41:45</td>
      <td>[0, 115]</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>RT @Politicalpolls_: If there was a second EU ...</td>
      <td>NaN</td>
      <td>1070614262857588736</td>
      <td>1070614262857588736</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>en</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1263</td>
      <td>False</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
      <td>NaN</td>
      <td>None</td>
      <td>Tue Sep 16 11:29:33 +0000 2014</td>
      <td>True</td>
      <td>None</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>{'description': {'urls': []}}</td>
      <td>13997.0</td>
      <td>None</td>
      <td>3080</td>
      <td>None</td>
      <td>3073.0</td>
      <td>None</td>
      <td>None</td>
      <td>2772143657</td>
      <td>2772143657</td>
      <td>None</td>
      <td>None</td>
      <td>en</td>
      <td>...</td>
      <td>None</td>
      <td>do_mck</td>
      <td>108501</td>
      <td>None</td>
      <td>none</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 116 columns</p>
</div>





```python
# append botometer univ_value and classifications
tweets_df = pd.merge(tweets_df, raw_df_verify[['screen_name', 'class_boto', 'boto_univ', 'class_verified']], left_on='user_screen_name', right_on='screen_name')
len(tweets_df.columns.values)
```





    120





```python
# heatmap to visualize the missing data in different columns
sns.set(style="darkgrid")
sns.set_context("poster")

def get_heatmap(df, imgName='NaN_heatmap.png'):
    #This function gives heatmap of all NaN values or only zero
    plt.figure(figsize=(20,10))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False,cmap="YlGnBu").set_title('Missing Data in Column 0 to 20')

    plt.tight_layout()
    
    # save image for report, need to run cell  
    plt.savefig(imgName)
    
    return plt.show()
```




```python
#plotting first null values
get_heatmap(tweets_df.ix[:,0:21], imgName='NaN_heatmap_col0_20.png')
```



![png](Final_Project_TestFile_files/Final_Project_TestFile_53_0.png)




```python
# obviously there are many columns are mostly missing values
# we want to drop the columns that miss more than 50% of the time
threshold = len(tweets_df.columns.values)*0.5
tweets_df_clean = tweets_df.dropna(thresh = threshold, axis='columns')
```




```python
# take a look at the columns left (reduced from 116 to 59 columns)
display(len(tweets_df_clean.columns.values))
display(tweets_df_clean.head(5))
```



    63



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>created_at</th>
      <th>display_text_range</th>
      <th>entities</th>
      <th>extended_entities</th>
      <th>favorite_count</th>
      <th>favorited</th>
      <th>full_text</th>
      <th>id</th>
      <th>id_str</th>
      <th>in_reply_to_screen_name</th>
      <th>in_reply_to_status_id</th>
      <th>in_reply_to_status_id_str</th>
      <th>in_reply_to_user_id</th>
      <th>in_reply_to_user_id_str</th>
      <th>is_quote_status</th>
      <th>lang</th>
      <th>possibly_sensitive</th>
      <th>quoted_status</th>
      <th>quoted_status_id</th>
      <th>quoted_status_id_str</th>
      <th>quoted_status_permalink</th>
      <th>retweet_count</th>
      <th>retweeted</th>
      <th>retweeted_status</th>
      <th>source</th>
      <th>truncated</th>
      <th>user</th>
      <th>user_created_at</th>
      <th>user_default_profile</th>
      <th>user_default_profile_image</th>
      <th>user_description</th>
      <th>user_entities</th>
      <th>user_favourites_count</th>
      <th>user_followers_count</th>
      <th>user_friends_count</th>
      <th>user_geo_enabled</th>
      <th>user_has_extended_profile</th>
      <th>user_id</th>
      <th>user_id_str</th>
      <th>user_lang</th>
      <th>user_listed_count</th>
      <th>user_location</th>
      <th>user_name</th>
      <th>user_profile_background_color</th>
      <th>user_profile_background_image_url</th>
      <th>user_profile_background_image_url_https</th>
      <th>user_profile_background_tile</th>
      <th>user_profile_banner_url</th>
      <th>user_profile_image_url</th>
      <th>user_profile_image_url_https</th>
      <th>user_profile_link_color</th>
      <th>user_profile_sidebar_border_color</th>
      <th>user_profile_sidebar_fill_color</th>
      <th>user_profile_text_color</th>
      <th>user_profile_use_background_image</th>
      <th>user_screen_name</th>
      <th>user_statuses_count</th>
      <th>user_translator_type</th>
      <th>user_url</th>
      <th>screen_name</th>
      <th>class_boto</th>
      <th>boto_univ</th>
      <th>class_verified</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-12-06 09:42:52</td>
      <td>[0, 140]</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>RT @David_Dowse1: Have you seen @jeremycorbyn ...</td>
      <td>1070614540746977281</td>
      <td>1070614540746977280</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>en</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>278</td>
      <td>False</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
      <td>Tue Sep 16 11:29:33 +0000 2014</td>
      <td>True</td>
      <td>None</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>{'description': {'urls': []}}</td>
      <td>13997.0</td>
      <td>3080</td>
      <td>3073.0</td>
      <td>None</td>
      <td>None</td>
      <td>2772143657</td>
      <td>2772143657</td>
      <td>en</td>
      <td>3.0</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>C0DEED</td>
      <td>http://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>https://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>None</td>
      <td>https://pbs.twimg.com/profile_banners/27721436...</td>
      <td>http://pbs.twimg.com/profile_images/5245241660...</td>
      <td>https://pbs.twimg.com/profile_images/524524166...</td>
      <td>1DA1F2</td>
      <td>C0DEED</td>
      <td>DDEEF6</td>
      <td>333333</td>
      <td>True</td>
      <td>do_mck</td>
      <td>108501</td>
      <td>none</td>
      <td>None</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-12-06 09:42:45</td>
      <td>[0, 139]</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>RT @MrHickmott: In your street there lives som...</td>
      <td>1070614514238922753</td>
      <td>1070614514238922752</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>en</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1586</td>
      <td>False</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
      <td>Tue Sep 16 11:29:33 +0000 2014</td>
      <td>True</td>
      <td>None</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>{'description': {'urls': []}}</td>
      <td>13997.0</td>
      <td>3080</td>
      <td>3073.0</td>
      <td>None</td>
      <td>None</td>
      <td>2772143657</td>
      <td>2772143657</td>
      <td>en</td>
      <td>3.0</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>C0DEED</td>
      <td>http://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>https://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>None</td>
      <td>https://pbs.twimg.com/profile_banners/27721436...</td>
      <td>http://pbs.twimg.com/profile_images/5245241660...</td>
      <td>https://pbs.twimg.com/profile_images/524524166...</td>
      <td>1DA1F2</td>
      <td>C0DEED</td>
      <td>DDEEF6</td>
      <td>333333</td>
      <td>True</td>
      <td>do_mck</td>
      <td>108501</td>
      <td>none</td>
      <td>None</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-12-06 09:42:25</td>
      <td>[0, 140]</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>RT @JumMurphy: The Unite general secretary, Le...</td>
      <td>1070614428859727872</td>
      <td>1070614428859727872</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>en</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>False</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
      <td>Tue Sep 16 11:29:33 +0000 2014</td>
      <td>True</td>
      <td>None</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>{'description': {'urls': []}}</td>
      <td>13997.0</td>
      <td>3080</td>
      <td>3073.0</td>
      <td>None</td>
      <td>None</td>
      <td>2772143657</td>
      <td>2772143657</td>
      <td>en</td>
      <td>3.0</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>C0DEED</td>
      <td>http://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>https://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>None</td>
      <td>https://pbs.twimg.com/profile_banners/27721436...</td>
      <td>http://pbs.twimg.com/profile_images/5245241660...</td>
      <td>https://pbs.twimg.com/profile_images/524524166...</td>
      <td>1DA1F2</td>
      <td>C0DEED</td>
      <td>DDEEF6</td>
      <td>333333</td>
      <td>True</td>
      <td>do_mck</td>
      <td>108501</td>
      <td>none</td>
      <td>None</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-12-06 09:42:07</td>
      <td>[0, 139]</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>RT @ArthurStramash: “The Scottish Parliament, ...</td>
      <td>1070614351076298753</td>
      <td>1070614351076298752</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>en</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>37</td>
      <td>False</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
      <td>Tue Sep 16 11:29:33 +0000 2014</td>
      <td>True</td>
      <td>None</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>{'description': {'urls': []}}</td>
      <td>13997.0</td>
      <td>3080</td>
      <td>3073.0</td>
      <td>None</td>
      <td>None</td>
      <td>2772143657</td>
      <td>2772143657</td>
      <td>en</td>
      <td>3.0</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>C0DEED</td>
      <td>http://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>https://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>None</td>
      <td>https://pbs.twimg.com/profile_banners/27721436...</td>
      <td>http://pbs.twimg.com/profile_images/5245241660...</td>
      <td>https://pbs.twimg.com/profile_images/524524166...</td>
      <td>1DA1F2</td>
      <td>C0DEED</td>
      <td>DDEEF6</td>
      <td>333333</td>
      <td>True</td>
      <td>do_mck</td>
      <td>108501</td>
      <td>none</td>
      <td>None</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-12-06 09:41:45</td>
      <td>[0, 115]</td>
      <td>{'hashtags': [], 'symbols': [], 'urls': [], 'u...</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>RT @Politicalpolls_: If there was a second EU ...</td>
      <td>1070614262857588736</td>
      <td>1070614262857588736</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>en</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1263</td>
      <td>False</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>False</td>
      <td>{'contributors_enabled': False, 'created_at': ...</td>
      <td>Tue Sep 16 11:29:33 +0000 2014</td>
      <td>True</td>
      <td>None</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>{'description': {'urls': []}}</td>
      <td>13997.0</td>
      <td>3080</td>
      <td>3073.0</td>
      <td>None</td>
      <td>None</td>
      <td>2772143657</td>
      <td>2772143657</td>
      <td>en</td>
      <td>3.0</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>C0DEED</td>
      <td>http://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>https://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>None</td>
      <td>https://pbs.twimg.com/profile_banners/27721436...</td>
      <td>http://pbs.twimg.com/profile_images/5245241660...</td>
      <td>https://pbs.twimg.com/profile_images/524524166...</td>
      <td>1DA1F2</td>
      <td>C0DEED</td>
      <td>DDEEF6</td>
      <td>333333</td>
      <td>True</td>
      <td>do_mck</td>
      <td>108501</td>
      <td>none</td>
      <td>None</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# duplicated columns
col_duplicate = ['entities','user', 'id_str', 'lang', 'user_lang', 'user_id', 'user_id_str']
# we dropped 'lang' as we only use english accounts for our dataset
# 'entities' and 'user' have already been parsed

# columns that we are obviously not itnerested
col_not_interested = ['source', 'user_entities']
# retweeted_status is the tweet object of the retweet - perhaps 
```




```python
# drop duplicated columns and columns that we are not interested
tweets_df = tweets_df_clean.drop(columns= (col_duplicate + col_not_interested))
```


[Back to TOC](#TOC) <br/>
<a id ='Data-Cleansing'></a>
#### 3.2 - Feature Engineering
feature engineering<br/>
<mark> add text - code to be updated <mark>

[Back to TOC](#TOC) <br/>
<a id ='Data-Cleansing'></a>
##### 3.2.1 - Feature Engineering - Tweet Features
feature engineering of tweet features: <br/>
1) text_rt: text of the retweet <br/>
2) text_tweet: text of the tweet (when there is no retweet) <br/>
3) encode tweet features <br/>



```python
tweets_df.head()
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>created_at</th>
      <th>display_text_range</th>
      <th>extended_entities</th>
      <th>favorite_count</th>
      <th>favorited</th>
      <th>full_text</th>
      <th>id</th>
      <th>in_reply_to_screen_name</th>
      <th>in_reply_to_status_id</th>
      <th>in_reply_to_status_id_str</th>
      <th>in_reply_to_user_id</th>
      <th>in_reply_to_user_id_str</th>
      <th>is_quote_status</th>
      <th>possibly_sensitive</th>
      <th>quoted_status</th>
      <th>quoted_status_id</th>
      <th>quoted_status_id_str</th>
      <th>quoted_status_permalink</th>
      <th>retweet_count</th>
      <th>retweeted</th>
      <th>retweeted_status</th>
      <th>truncated</th>
      <th>user_created_at</th>
      <th>user_default_profile</th>
      <th>user_default_profile_image</th>
      <th>user_description</th>
      <th>user_favourites_count</th>
      <th>user_followers_count</th>
      <th>user_friends_count</th>
      <th>user_geo_enabled</th>
      <th>user_has_extended_profile</th>
      <th>user_listed_count</th>
      <th>user_location</th>
      <th>user_name</th>
      <th>user_profile_background_color</th>
      <th>user_profile_background_image_url</th>
      <th>user_profile_background_image_url_https</th>
      <th>user_profile_background_tile</th>
      <th>user_profile_banner_url</th>
      <th>user_profile_image_url</th>
      <th>user_profile_image_url_https</th>
      <th>user_profile_link_color</th>
      <th>user_profile_sidebar_border_color</th>
      <th>user_profile_sidebar_fill_color</th>
      <th>user_profile_text_color</th>
      <th>user_profile_use_background_image</th>
      <th>user_screen_name</th>
      <th>user_statuses_count</th>
      <th>user_translator_type</th>
      <th>user_url</th>
      <th>screen_name</th>
      <th>class_boto</th>
      <th>boto_univ</th>
      <th>class_verified</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-12-06 09:42:52</td>
      <td>[0, 140]</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>RT @David_Dowse1: Have you seen @jeremycorbyn ...</td>
      <td>1070614540746977281</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>278</td>
      <td>False</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>False</td>
      <td>Tue Sep 16 11:29:33 +0000 2014</td>
      <td>True</td>
      <td>None</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>13997.0</td>
      <td>3080</td>
      <td>3073.0</td>
      <td>None</td>
      <td>None</td>
      <td>3.0</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>C0DEED</td>
      <td>http://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>https://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>None</td>
      <td>https://pbs.twimg.com/profile_banners/27721436...</td>
      <td>http://pbs.twimg.com/profile_images/5245241660...</td>
      <td>https://pbs.twimg.com/profile_images/524524166...</td>
      <td>1DA1F2</td>
      <td>C0DEED</td>
      <td>DDEEF6</td>
      <td>333333</td>
      <td>True</td>
      <td>do_mck</td>
      <td>108501</td>
      <td>none</td>
      <td>None</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-12-06 09:42:45</td>
      <td>[0, 139]</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>RT @MrHickmott: In your street there lives som...</td>
      <td>1070614514238922753</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1586</td>
      <td>False</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>False</td>
      <td>Tue Sep 16 11:29:33 +0000 2014</td>
      <td>True</td>
      <td>None</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>13997.0</td>
      <td>3080</td>
      <td>3073.0</td>
      <td>None</td>
      <td>None</td>
      <td>3.0</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>C0DEED</td>
      <td>http://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>https://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>None</td>
      <td>https://pbs.twimg.com/profile_banners/27721436...</td>
      <td>http://pbs.twimg.com/profile_images/5245241660...</td>
      <td>https://pbs.twimg.com/profile_images/524524166...</td>
      <td>1DA1F2</td>
      <td>C0DEED</td>
      <td>DDEEF6</td>
      <td>333333</td>
      <td>True</td>
      <td>do_mck</td>
      <td>108501</td>
      <td>none</td>
      <td>None</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-12-06 09:42:25</td>
      <td>[0, 140]</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>RT @JumMurphy: The Unite general secretary, Le...</td>
      <td>1070614428859727872</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>False</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>False</td>
      <td>Tue Sep 16 11:29:33 +0000 2014</td>
      <td>True</td>
      <td>None</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>13997.0</td>
      <td>3080</td>
      <td>3073.0</td>
      <td>None</td>
      <td>None</td>
      <td>3.0</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>C0DEED</td>
      <td>http://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>https://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>None</td>
      <td>https://pbs.twimg.com/profile_banners/27721436...</td>
      <td>http://pbs.twimg.com/profile_images/5245241660...</td>
      <td>https://pbs.twimg.com/profile_images/524524166...</td>
      <td>1DA1F2</td>
      <td>C0DEED</td>
      <td>DDEEF6</td>
      <td>333333</td>
      <td>True</td>
      <td>do_mck</td>
      <td>108501</td>
      <td>none</td>
      <td>None</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-12-06 09:42:07</td>
      <td>[0, 139]</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>RT @ArthurStramash: “The Scottish Parliament, ...</td>
      <td>1070614351076298753</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>37</td>
      <td>False</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>False</td>
      <td>Tue Sep 16 11:29:33 +0000 2014</td>
      <td>True</td>
      <td>None</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>13997.0</td>
      <td>3080</td>
      <td>3073.0</td>
      <td>None</td>
      <td>None</td>
      <td>3.0</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>C0DEED</td>
      <td>http://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>https://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>None</td>
      <td>https://pbs.twimg.com/profile_banners/27721436...</td>
      <td>http://pbs.twimg.com/profile_images/5245241660...</td>
      <td>https://pbs.twimg.com/profile_images/524524166...</td>
      <td>1DA1F2</td>
      <td>C0DEED</td>
      <td>DDEEF6</td>
      <td>333333</td>
      <td>True</td>
      <td>do_mck</td>
      <td>108501</td>
      <td>none</td>
      <td>None</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-12-06 09:41:45</td>
      <td>[0, 115]</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>RT @Politicalpolls_: If there was a second EU ...</td>
      <td>1070614262857588736</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1263</td>
      <td>False</td>
      <td>{'contributors': None, 'coordinates': None, 'c...</td>
      <td>False</td>
      <td>Tue Sep 16 11:29:33 +0000 2014</td>
      <td>True</td>
      <td>None</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>13997.0</td>
      <td>3080</td>
      <td>3073.0</td>
      <td>None</td>
      <td>None</td>
      <td>3.0</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>C0DEED</td>
      <td>http://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>https://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>None</td>
      <td>https://pbs.twimg.com/profile_banners/27721436...</td>
      <td>http://pbs.twimg.com/profile_images/5245241660...</td>
      <td>https://pbs.twimg.com/profile_images/524524166...</td>
      <td>1DA1F2</td>
      <td>C0DEED</td>
      <td>DDEEF6</td>
      <td>333333</td>
      <td>True</td>
      <td>do_mck</td>
      <td>108501</td>
      <td>none</td>
      <td>None</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>





```python
# although using tweet_mode='extended', we are still not getting the full text
# therefore, we tried to get full_text from retweeted_status
tweets_df['text_rt'] = tweets_df['retweeted_status'].map(lambda x: x['full_text'] if x and not isinstance(x, float) and ('full_text' in x) else None)
tweets_df['text_tweet'] = tweets_df['full_text'].where(tweets_df['text_rt'].map(lambda x: x is None), None)
tweets_df[['text_tweet', 'text_rt']].head(20)
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text_tweet</th>
      <th>text_rt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>None</td>
      <td>Have you seen @jeremycorbyn savaging the torie...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>None</td>
      <td>In your street there lives someone who moves t...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>None</td>
      <td>The Unite general secretary, Len McCluskey war...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>None</td>
      <td>“The Scottish Parliament, speaking for the Sco...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>None</td>
      <td>If there was a second EU referendum \n\nHow wo...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>None</td>
      <td>@MyScotlandpage @Indy_Quint @ferguson2811 Oh n...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>None</td>
      <td>@Indy_Quint @ferguson2811 Meghan has stopped t...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>None</td>
      <td>@Indy_Quint @iluvfilms What a waste of a freez...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>None</td>
      <td>@Indy_Quint @goanabootbiddy This has brightene...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>None</td>
      <td>I think Prince Phillip has been dead for weeks...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>@gaf_young @alicmurray @sparkyhamill @DavidJFH...</td>
      <td>None</td>
    </tr>
    <tr>
      <th>11</th>
      <td>None</td>
      <td>@alicmurray @do_mck @sparkyhamill @DavidJFHall...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Don’t be fooled, Stephen !\nThis guy @PaulWill...</td>
      <td>None</td>
    </tr>
    <tr>
      <th>13</th>
      <td>None</td>
      <td>@kezdugdale Too right Kez!\nBTW, he wouldn't b...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Ha ha 😂\nA TORY ...\nSuggesting other people a...</td>
      <td>None</td>
    </tr>
    <tr>
      <th>15</th>
      <td>None</td>
      <td>Humans are so heartless https://t.co/HnnNrFO0cO</td>
    </tr>
    <tr>
      <th>16</th>
      <td>None</td>
      <td>@AlexNiff @do_mck @coopuk During the irish &amp;am...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>None</td>
      <td>@jeremybement @do_mck Happy Birthday wee man, ...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>None</td>
      <td>@fatweegee @sparkyhamill @WingsScotland Racist...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>None</td>
      <td>@sparkyhamill @WingsScotland so our tories and...</td>
    </tr>
  </tbody>
</table>
</div>





```python
# encode tweet features

# 1 = has extende entities; 0 = don't have extende entities
tweets_df['extended_entities'] = tweets_df['extended_entities'].map(lambda x: 0 if x==None else 1) 

# 1 = favorited - True; 0 = favorited - False
tweets_df['favorited'] = tweets_df['favorited'].map(lambda x: 0 if x==False else 1) 

# 1 = is_quote_status - True - True; 0 = is_quote_status - False
tweets_df['is_quote_status'] = tweets_df['is_quote_status'].map(lambda x: 0 if x==False else 1) 

# -1 = None; else - actual value
#tweets_df['possibly_sensitive'] = tweets_df['possibly_sensitive'].map(lambda x: x if x>=0 else -1) 

# 1 = reply to at least one user; 0 = not reply to another user
tweets_df['in_reply_to_screen_name'] = tweets_df['in_reply_to_screen_name'].map(lambda x: 1 if x else 0)

# 1 = retweeted-true; 0 = retweeted-false
tweets_df['retweeted'] = tweets_df['retweeted'].map(lambda x: 1 if x==True else 0) 

# 1 = tweet includes retweet; 0 = tweet does not include retweet
tweets_df['retweeted_status'] = tweets_df['retweeted_status'].map(lambda x: 0 if x==None else 1)

# 0 = none or information not available
tweets_df['user_listed_count'] = tweets_df['user_listed_count'].map(lambda x: x if x>0 else 0)

# 1 = truncated-true; 0 = truncated-false
tweets_df['truncated'] = tweets_df['truncated'].map(lambda x: 0 if x==False else 1) 

# replace nan with 0 for the following features (as for these features, missing values usually means 0)
for f in ['user_favourites_count', 'user_followers_count', 'user_friends_count']:
    tweets_df[f] = tweets_df[f].replace(np.nan, 0, regex=True)

tweets_df.head(5)
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>created_at</th>
      <th>display_text_range</th>
      <th>extended_entities</th>
      <th>favorite_count</th>
      <th>favorited</th>
      <th>full_text</th>
      <th>id</th>
      <th>in_reply_to_screen_name</th>
      <th>in_reply_to_status_id</th>
      <th>in_reply_to_status_id_str</th>
      <th>in_reply_to_user_id</th>
      <th>in_reply_to_user_id_str</th>
      <th>is_quote_status</th>
      <th>possibly_sensitive</th>
      <th>quoted_status</th>
      <th>quoted_status_id</th>
      <th>quoted_status_id_str</th>
      <th>quoted_status_permalink</th>
      <th>retweet_count</th>
      <th>retweeted</th>
      <th>retweeted_status</th>
      <th>truncated</th>
      <th>user_created_at</th>
      <th>user_default_profile</th>
      <th>user_default_profile_image</th>
      <th>user_description</th>
      <th>user_favourites_count</th>
      <th>user_followers_count</th>
      <th>user_friends_count</th>
      <th>user_geo_enabled</th>
      <th>user_has_extended_profile</th>
      <th>user_listed_count</th>
      <th>user_location</th>
      <th>user_name</th>
      <th>user_profile_background_color</th>
      <th>user_profile_background_image_url</th>
      <th>user_profile_background_image_url_https</th>
      <th>user_profile_background_tile</th>
      <th>user_profile_banner_url</th>
      <th>user_profile_image_url</th>
      <th>user_profile_image_url_https</th>
      <th>user_profile_link_color</th>
      <th>user_profile_sidebar_border_color</th>
      <th>user_profile_sidebar_fill_color</th>
      <th>user_profile_text_color</th>
      <th>user_profile_use_background_image</th>
      <th>user_screen_name</th>
      <th>user_statuses_count</th>
      <th>user_translator_type</th>
      <th>user_url</th>
      <th>screen_name</th>
      <th>class_boto</th>
      <th>boto_univ</th>
      <th>class_verified</th>
      <th>text_rt</th>
      <th>text_tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-12-06 09:42:52</td>
      <td>[0, 140]</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>RT @David_Dowse1: Have you seen @jeremycorbyn ...</td>
      <td>1070614540746977281</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>278</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Tue Sep 16 11:29:33 +0000 2014</td>
      <td>True</td>
      <td>None</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>13997.0</td>
      <td>3080</td>
      <td>3073.0</td>
      <td>None</td>
      <td>None</td>
      <td>3.0</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>C0DEED</td>
      <td>http://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>https://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>None</td>
      <td>https://pbs.twimg.com/profile_banners/27721436...</td>
      <td>http://pbs.twimg.com/profile_images/5245241660...</td>
      <td>https://pbs.twimg.com/profile_images/524524166...</td>
      <td>1DA1F2</td>
      <td>C0DEED</td>
      <td>DDEEF6</td>
      <td>333333</td>
      <td>True</td>
      <td>do_mck</td>
      <td>108501</td>
      <td>none</td>
      <td>None</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
      <td>Have you seen @jeremycorbyn savaging the torie...</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-12-06 09:42:45</td>
      <td>[0, 139]</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>RT @MrHickmott: In your street there lives som...</td>
      <td>1070614514238922753</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1586</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Tue Sep 16 11:29:33 +0000 2014</td>
      <td>True</td>
      <td>None</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>13997.0</td>
      <td>3080</td>
      <td>3073.0</td>
      <td>None</td>
      <td>None</td>
      <td>3.0</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>C0DEED</td>
      <td>http://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>https://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>None</td>
      <td>https://pbs.twimg.com/profile_banners/27721436...</td>
      <td>http://pbs.twimg.com/profile_images/5245241660...</td>
      <td>https://pbs.twimg.com/profile_images/524524166...</td>
      <td>1DA1F2</td>
      <td>C0DEED</td>
      <td>DDEEF6</td>
      <td>333333</td>
      <td>True</td>
      <td>do_mck</td>
      <td>108501</td>
      <td>none</td>
      <td>None</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
      <td>In your street there lives someone who moves t...</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-12-06 09:42:25</td>
      <td>[0, 140]</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>RT @JumMurphy: The Unite general secretary, Le...</td>
      <td>1070614428859727872</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Tue Sep 16 11:29:33 +0000 2014</td>
      <td>True</td>
      <td>None</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>13997.0</td>
      <td>3080</td>
      <td>3073.0</td>
      <td>None</td>
      <td>None</td>
      <td>3.0</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>C0DEED</td>
      <td>http://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>https://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>None</td>
      <td>https://pbs.twimg.com/profile_banners/27721436...</td>
      <td>http://pbs.twimg.com/profile_images/5245241660...</td>
      <td>https://pbs.twimg.com/profile_images/524524166...</td>
      <td>1DA1F2</td>
      <td>C0DEED</td>
      <td>DDEEF6</td>
      <td>333333</td>
      <td>True</td>
      <td>do_mck</td>
      <td>108501</td>
      <td>none</td>
      <td>None</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
      <td>The Unite general secretary, Len McCluskey war...</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-12-06 09:42:07</td>
      <td>[0, 139]</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>RT @ArthurStramash: “The Scottish Parliament, ...</td>
      <td>1070614351076298753</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>37</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Tue Sep 16 11:29:33 +0000 2014</td>
      <td>True</td>
      <td>None</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>13997.0</td>
      <td>3080</td>
      <td>3073.0</td>
      <td>None</td>
      <td>None</td>
      <td>3.0</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>C0DEED</td>
      <td>http://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>https://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>None</td>
      <td>https://pbs.twimg.com/profile_banners/27721436...</td>
      <td>http://pbs.twimg.com/profile_images/5245241660...</td>
      <td>https://pbs.twimg.com/profile_images/524524166...</td>
      <td>1DA1F2</td>
      <td>C0DEED</td>
      <td>DDEEF6</td>
      <td>333333</td>
      <td>True</td>
      <td>do_mck</td>
      <td>108501</td>
      <td>none</td>
      <td>None</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
      <td>“The Scottish Parliament, speaking for the Sco...</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-12-06 09:41:45</td>
      <td>[0, 115]</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>RT @Politicalpolls_: If there was a second EU ...</td>
      <td>1070614262857588736</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1263</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Tue Sep 16 11:29:33 +0000 2014</td>
      <td>True</td>
      <td>None</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>13997.0</td>
      <td>3080</td>
      <td>3073.0</td>
      <td>None</td>
      <td>None</td>
      <td>3.0</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>C0DEED</td>
      <td>http://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>https://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>None</td>
      <td>https://pbs.twimg.com/profile_banners/27721436...</td>
      <td>http://pbs.twimg.com/profile_images/5245241660...</td>
      <td>https://pbs.twimg.com/profile_images/524524166...</td>
      <td>1DA1F2</td>
      <td>C0DEED</td>
      <td>DDEEF6</td>
      <td>333333</td>
      <td>True</td>
      <td>do_mck</td>
      <td>108501</td>
      <td>none</td>
      <td>None</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
      <td>If there was a second EU referendum \n\nHow wo...</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>



[Back to TOC](#TOC) <br/>
<a id ='Data-Cleansing'></a>
##### 3.2.2 - Feature Engineering - User Features
feature engineering of user features: <br/>
1) length of user description <br/>
2) tweet frequencies (the mean, std, min, and max time between tweets for each account) <br/>
3) account age (seconds from the account creation time to the latest tweet time) <br/>



```python
# account feature engineering
# create an intermedium df with all account-related data from tweets

users_description_len_df = tweets_df.drop_duplicates(subset=['screen_name'])
users_description_len_df['user_description_len'] = users_description_len_df['user_description'].map(lambda x: len(x) if x!=None else 0)
```




```python
# account feature engineering
# get tweets interval stats (in seconds)

def create_tweet_time_stats(created_at_series):
    times = created_at_series['created_at'].sort_values().diff().dt.total_seconds()[1:]
    cols = ['tweet_time_mean', 'tweet_time_std', 'tweet_time_min', 'tweet_time_max']
    return pd.Series([times.mean(), times.std(), times.min(), times.max()], index=cols)

tweet_time_stats_df = tweets_df[['screen_name', 'created_at']].groupby('screen_name').apply(create_tweet_time_stats).reset_index()
tweet_time_stats_df.head()
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>screen_name</th>
      <th>tweet_time_mean</th>
      <th>tweet_time_std</th>
      <th>tweet_time_min</th>
      <th>tweet_time_max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0ttaM</td>
      <td>17824.869347</td>
      <td>34897.555475</td>
      <td>3.0</td>
      <td>161802.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Abills</td>
      <td>17956.708543</td>
      <td>33535.836124</td>
      <td>0.0</td>
      <td>182940.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AltinaDair</td>
      <td>264.718593</td>
      <td>391.140551</td>
      <td>0.0</td>
      <td>1273.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BSD_Masternode</td>
      <td>1799.994975</td>
      <td>4.411300</td>
      <td>1785.0</td>
      <td>1815.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BTCticker</td>
      <td>877.402010</td>
      <td>866.659009</td>
      <td>0.0</td>
      <td>1819.0</td>
    </tr>
  </tbody>
</table>
</div>





```python
# account feature engineering
# get account age (in seconds)

reference_date = tweets_df['created_at'].max()
user_account_age_df = tweets_df[['screen_name', 'user_created_at']].groupby('screen_name').min().reset_index()
user_account_age_df['account_age'] = user_account_age_df['user_created_at'].map(lambda d: (reference_date - pd.to_datetime(d)).total_seconds())
del user_account_age_df['user_created_at']
user_account_age_df.head()
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>screen_name</th>
      <th>account_age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0ttaM</td>
      <td>141919299.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Abills</td>
      <td>264813954.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AltinaDair</td>
      <td>34952734.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BSD_Masternode</td>
      <td>44056297.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BTCticker</td>
      <td>179151892.0</td>
    </tr>
  </tbody>
</table>
</div>





```python
# account feature engineering
# create a new dataframe with engineered features that are associated with each user
users_df = pd.DataFrame(tweets_df['screen_name']).drop_duplicates(subset='screen_name')
users_df = pd.merge(users_df, tweet_time_stats_df, left_on='screen_name', right_on='screen_name')
users_df = pd.merge(users_df, users_description_len_df[['screen_name', 'user_description_len']], left_on='screen_name', right_on='screen_name')
users_df = pd.merge(users_df, user_account_age_df, left_on='screen_name', right_on='screen_name')
users_df.head(5)
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>screen_name</th>
      <th>tweet_time_mean</th>
      <th>tweet_time_std</th>
      <th>tweet_time_min</th>
      <th>tweet_time_max</th>
      <th>user_description_len</th>
      <th>account_age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>do_mck</td>
      <td>428.195980</td>
      <td>2753.761741</td>
      <td>1.0</td>
      <td>30530.0</td>
      <td>87</td>
      <td>133222476.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GerryJBrown</td>
      <td>8666.525253</td>
      <td>17894.700903</td>
      <td>10.0</td>
      <td>112888.0</td>
      <td>124</td>
      <td>245459350.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>alanthelondoner</td>
      <td>1312.376884</td>
      <td>5778.957185</td>
      <td>0.0</td>
      <td>45423.0</td>
      <td>158</td>
      <td>236809466.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>nosheepzone</td>
      <td>877.386935</td>
      <td>4431.100708</td>
      <td>0.0</td>
      <td>36806.0</td>
      <td>160</td>
      <td>208451124.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>flyingdavy</td>
      <td>3937.609137</td>
      <td>9659.666613</td>
      <td>1.0</td>
      <td>57959.0</td>
      <td>145</td>
      <td>133897253.0</td>
    </tr>
  </tbody>
</table>
</div>





```python
# merge the account information back to the dataset
tweets_df = pd.merge(tweets_df, users_df, left_on='screen_name', right_on='screen_name')
tweets_df.head(5)
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>created_at</th>
      <th>display_text_range</th>
      <th>extended_entities</th>
      <th>favorite_count</th>
      <th>favorited</th>
      <th>full_text</th>
      <th>id</th>
      <th>in_reply_to_screen_name</th>
      <th>in_reply_to_status_id</th>
      <th>in_reply_to_status_id_str</th>
      <th>in_reply_to_user_id</th>
      <th>in_reply_to_user_id_str</th>
      <th>is_quote_status</th>
      <th>possibly_sensitive</th>
      <th>quoted_status</th>
      <th>quoted_status_id</th>
      <th>quoted_status_id_str</th>
      <th>quoted_status_permalink</th>
      <th>retweet_count</th>
      <th>retweeted</th>
      <th>retweeted_status</th>
      <th>truncated</th>
      <th>user_created_at</th>
      <th>user_default_profile</th>
      <th>user_default_profile_image</th>
      <th>user_description</th>
      <th>user_favourites_count</th>
      <th>user_followers_count</th>
      <th>user_friends_count</th>
      <th>user_geo_enabled</th>
      <th>user_has_extended_profile</th>
      <th>user_listed_count</th>
      <th>user_location</th>
      <th>user_name</th>
      <th>user_profile_background_color</th>
      <th>user_profile_background_image_url</th>
      <th>user_profile_background_image_url_https</th>
      <th>user_profile_background_tile</th>
      <th>user_profile_banner_url</th>
      <th>user_profile_image_url</th>
      <th>user_profile_image_url_https</th>
      <th>user_profile_link_color</th>
      <th>user_profile_sidebar_border_color</th>
      <th>user_profile_sidebar_fill_color</th>
      <th>user_profile_text_color</th>
      <th>user_profile_use_background_image</th>
      <th>user_screen_name</th>
      <th>user_statuses_count</th>
      <th>user_translator_type</th>
      <th>user_url</th>
      <th>screen_name</th>
      <th>class_boto</th>
      <th>boto_univ</th>
      <th>class_verified</th>
      <th>text_rt</th>
      <th>text_tweet</th>
      <th>tweet_time_mean</th>
      <th>tweet_time_std</th>
      <th>tweet_time_min</th>
      <th>tweet_time_max</th>
      <th>user_description_len</th>
      <th>account_age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-12-06 09:42:52</td>
      <td>[0, 140]</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>RT @David_Dowse1: Have you seen @jeremycorbyn ...</td>
      <td>1070614540746977281</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>278</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Tue Sep 16 11:29:33 +0000 2014</td>
      <td>True</td>
      <td>None</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>13997.0</td>
      <td>3080</td>
      <td>3073.0</td>
      <td>None</td>
      <td>None</td>
      <td>3.0</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>C0DEED</td>
      <td>http://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>https://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>None</td>
      <td>https://pbs.twimg.com/profile_banners/27721436...</td>
      <td>http://pbs.twimg.com/profile_images/5245241660...</td>
      <td>https://pbs.twimg.com/profile_images/524524166...</td>
      <td>1DA1F2</td>
      <td>C0DEED</td>
      <td>DDEEF6</td>
      <td>333333</td>
      <td>True</td>
      <td>do_mck</td>
      <td>108501</td>
      <td>none</td>
      <td>None</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
      <td>Have you seen @jeremycorbyn savaging the torie...</td>
      <td>None</td>
      <td>428.19598</td>
      <td>2753.761741</td>
      <td>1.0</td>
      <td>30530.0</td>
      <td>87</td>
      <td>133222476.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-12-06 09:42:45</td>
      <td>[0, 139]</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>RT @MrHickmott: In your street there lives som...</td>
      <td>1070614514238922753</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1586</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Tue Sep 16 11:29:33 +0000 2014</td>
      <td>True</td>
      <td>None</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>13997.0</td>
      <td>3080</td>
      <td>3073.0</td>
      <td>None</td>
      <td>None</td>
      <td>3.0</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>C0DEED</td>
      <td>http://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>https://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>None</td>
      <td>https://pbs.twimg.com/profile_banners/27721436...</td>
      <td>http://pbs.twimg.com/profile_images/5245241660...</td>
      <td>https://pbs.twimg.com/profile_images/524524166...</td>
      <td>1DA1F2</td>
      <td>C0DEED</td>
      <td>DDEEF6</td>
      <td>333333</td>
      <td>True</td>
      <td>do_mck</td>
      <td>108501</td>
      <td>none</td>
      <td>None</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
      <td>In your street there lives someone who moves t...</td>
      <td>None</td>
      <td>428.19598</td>
      <td>2753.761741</td>
      <td>1.0</td>
      <td>30530.0</td>
      <td>87</td>
      <td>133222476.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-12-06 09:42:25</td>
      <td>[0, 140]</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>RT @JumMurphy: The Unite general secretary, Le...</td>
      <td>1070614428859727872</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Tue Sep 16 11:29:33 +0000 2014</td>
      <td>True</td>
      <td>None</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>13997.0</td>
      <td>3080</td>
      <td>3073.0</td>
      <td>None</td>
      <td>None</td>
      <td>3.0</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>C0DEED</td>
      <td>http://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>https://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>None</td>
      <td>https://pbs.twimg.com/profile_banners/27721436...</td>
      <td>http://pbs.twimg.com/profile_images/5245241660...</td>
      <td>https://pbs.twimg.com/profile_images/524524166...</td>
      <td>1DA1F2</td>
      <td>C0DEED</td>
      <td>DDEEF6</td>
      <td>333333</td>
      <td>True</td>
      <td>do_mck</td>
      <td>108501</td>
      <td>none</td>
      <td>None</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
      <td>The Unite general secretary, Len McCluskey war...</td>
      <td>None</td>
      <td>428.19598</td>
      <td>2753.761741</td>
      <td>1.0</td>
      <td>30530.0</td>
      <td>87</td>
      <td>133222476.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-12-06 09:42:07</td>
      <td>[0, 139]</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>RT @ArthurStramash: “The Scottish Parliament, ...</td>
      <td>1070614351076298753</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>37</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Tue Sep 16 11:29:33 +0000 2014</td>
      <td>True</td>
      <td>None</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>13997.0</td>
      <td>3080</td>
      <td>3073.0</td>
      <td>None</td>
      <td>None</td>
      <td>3.0</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>C0DEED</td>
      <td>http://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>https://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>None</td>
      <td>https://pbs.twimg.com/profile_banners/27721436...</td>
      <td>http://pbs.twimg.com/profile_images/5245241660...</td>
      <td>https://pbs.twimg.com/profile_images/524524166...</td>
      <td>1DA1F2</td>
      <td>C0DEED</td>
      <td>DDEEF6</td>
      <td>333333</td>
      <td>True</td>
      <td>do_mck</td>
      <td>108501</td>
      <td>none</td>
      <td>None</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
      <td>“The Scottish Parliament, speaking for the Sco...</td>
      <td>None</td>
      <td>428.19598</td>
      <td>2753.761741</td>
      <td>1.0</td>
      <td>30530.0</td>
      <td>87</td>
      <td>133222476.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-12-06 09:41:45</td>
      <td>[0, 115]</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>RT @Politicalpolls_: If there was a second EU ...</td>
      <td>1070614262857588736</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1263</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Tue Sep 16 11:29:33 +0000 2014</td>
      <td>True</td>
      <td>None</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>13997.0</td>
      <td>3080</td>
      <td>3073.0</td>
      <td>None</td>
      <td>None</td>
      <td>3.0</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>C0DEED</td>
      <td>http://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>https://abs.twimg.com/images/themes/theme1/bg.png</td>
      <td>None</td>
      <td>https://pbs.twimg.com/profile_banners/27721436...</td>
      <td>http://pbs.twimg.com/profile_images/5245241660...</td>
      <td>https://pbs.twimg.com/profile_images/524524166...</td>
      <td>1DA1F2</td>
      <td>C0DEED</td>
      <td>DDEEF6</td>
      <td>333333</td>
      <td>True</td>
      <td>do_mck</td>
      <td>108501</td>
      <td>none</td>
      <td>None</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
      <td>If there was a second EU referendum \n\nHow wo...</td>
      <td>None</td>
      <td>428.19598</td>
      <td>2753.761741</td>
      <td>1.0</td>
      <td>30530.0</td>
      <td>87</td>
      <td>133222476.0</td>
    </tr>
  </tbody>
</table>
</div>



[Back to TOC](#TOC) <br/>
<a id ='Data-Cleansing'></a>
##### 3.2.3 - Feature Engineering - Finalize and Clean Up Data
drop the columns that are no longer interesting / useful



```python
# delete columns that no longer useful
col_del = ['display_text_range', 'in_reply_to_status_id_str', 'in_reply_to_user_id_str','in_reply_to_status_id', 
           'in_reply_to_user_id', 'quoted_status', 'quoted_status_id', 'quoted_status_id_str',
          'quoted_status_permalink', 'user_url', 'user_translator_type', 'user_default_profile_image',
          'user_default_profile', 'user_geo_enabled', 'user_has_extended_profile', 'user_profile_background_tile',
          'user_profile_image_url', 'user_profile_image_url_https', 'full_text', 'created_at', 
          'user_created_at', 'user_profile_background_image_url', 'user_profile_background_image_url_https',
          'user_profile_banner_url', 'user_profile_link_color', 'user_profile_sidebar_border_color',
           'possibly_sensitive', 'user_profile_sidebar_fill_color', 'user_profile_text_color', 'user_screen_name',
          'user_profile_background_color']

tweets_df = tweets_df.drop(columns=col_del)
tweets_df.head(5)
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>extended_entities</th>
      <th>favorite_count</th>
      <th>favorited</th>
      <th>id</th>
      <th>in_reply_to_screen_name</th>
      <th>is_quote_status</th>
      <th>retweet_count</th>
      <th>retweeted</th>
      <th>retweeted_status</th>
      <th>truncated</th>
      <th>user_description</th>
      <th>user_favourites_count</th>
      <th>user_followers_count</th>
      <th>user_friends_count</th>
      <th>user_listed_count</th>
      <th>user_location</th>
      <th>user_name</th>
      <th>user_profile_use_background_image</th>
      <th>user_statuses_count</th>
      <th>screen_name</th>
      <th>class_boto</th>
      <th>boto_univ</th>
      <th>class_verified</th>
      <th>text_rt</th>
      <th>text_tweet</th>
      <th>tweet_time_mean</th>
      <th>tweet_time_std</th>
      <th>tweet_time_min</th>
      <th>tweet_time_max</th>
      <th>user_description_len</th>
      <th>account_age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1070614540746977281</td>
      <td>0</td>
      <td>0</td>
      <td>278</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>13997.0</td>
      <td>3080</td>
      <td>3073.0</td>
      <td>3.0</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>True</td>
      <td>108501</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
      <td>Have you seen @jeremycorbyn savaging the torie...</td>
      <td>None</td>
      <td>428.19598</td>
      <td>2753.761741</td>
      <td>1.0</td>
      <td>30530.0</td>
      <td>87</td>
      <td>133222476.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1070614514238922753</td>
      <td>0</td>
      <td>0</td>
      <td>1586</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>13997.0</td>
      <td>3080</td>
      <td>3073.0</td>
      <td>3.0</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>True</td>
      <td>108501</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
      <td>In your street there lives someone who moves t...</td>
      <td>None</td>
      <td>428.19598</td>
      <td>2753.761741</td>
      <td>1.0</td>
      <td>30530.0</td>
      <td>87</td>
      <td>133222476.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1070614428859727872</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>13997.0</td>
      <td>3080</td>
      <td>3073.0</td>
      <td>3.0</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>True</td>
      <td>108501</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
      <td>The Unite general secretary, Len McCluskey war...</td>
      <td>None</td>
      <td>428.19598</td>
      <td>2753.761741</td>
      <td>1.0</td>
      <td>30530.0</td>
      <td>87</td>
      <td>133222476.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1070614351076298753</td>
      <td>0</td>
      <td>0</td>
      <td>37</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>13997.0</td>
      <td>3080</td>
      <td>3073.0</td>
      <td>3.0</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>True</td>
      <td>108501</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
      <td>“The Scottish Parliament, speaking for the Sco...</td>
      <td>None</td>
      <td>428.19598</td>
      <td>2753.761741</td>
      <td>1.0</td>
      <td>30530.0</td>
      <td>87</td>
      <td>133222476.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1070614262857588736</td>
      <td>0</td>
      <td>0</td>
      <td>1263</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>13997.0</td>
      <td>3080</td>
      <td>3073.0</td>
      <td>3.0</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>True</td>
      <td>108501</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
      <td>If there was a second EU referendum \n\nHow wo...</td>
      <td>None</td>
      <td>428.19598</td>
      <td>2753.761741</td>
      <td>1.0</td>
      <td>30530.0</td>
      <td>87</td>
      <td>133222476.0</td>
    </tr>
  </tbody>
</table>
</div>





```python
# create list of columns names for different categories
col_boto = list(tweets_df[['class_boto', 'boto_univ']].columns.values)
col_response = ['class_boto']
col_verified = ['class_verified']
col_pred_text = list(tweets_df.select_dtypes(['object']).columns.values)
col_id = ['id']
col_pred_numerical = list(tweets_df.select_dtypes(['float64', 'int64']).drop(columns=['boto_univ', 'class_verified', 'id']).columns.values)
```




```python
# take a look at the more structured / cleaned up data
display(tweets_df.describe())
display(tweets_df.shape)
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>extended_entities</th>
      <th>favorite_count</th>
      <th>favorited</th>
      <th>id</th>
      <th>in_reply_to_screen_name</th>
      <th>is_quote_status</th>
      <th>retweet_count</th>
      <th>retweeted</th>
      <th>retweeted_status</th>
      <th>truncated</th>
      <th>user_favourites_count</th>
      <th>user_followers_count</th>
      <th>user_friends_count</th>
      <th>user_listed_count</th>
      <th>user_statuses_count</th>
      <th>boto_univ</th>
      <th>class_verified</th>
      <th>tweet_time_mean</th>
      <th>tweet_time_std</th>
      <th>tweet_time_min</th>
      <th>tweet_time_max</th>
      <th>user_description_len</th>
      <th>account_age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9990.0</td>
      <td>9990.000000</td>
      <td>9990.0</td>
      <td>9.990000e+03</td>
      <td>9990.000000</td>
      <td>9990.000000</td>
      <td>9990.000000</td>
      <td>9990.0</td>
      <td>9990.0</td>
      <td>9990.0</td>
      <td>9990.000000</td>
      <td>9990.000000</td>
      <td>9990.000000</td>
      <td>9990.000000</td>
      <td>9.990000e+03</td>
      <td>9990.000000</td>
      <td>9990.000000</td>
      <td>9990.000000</td>
      <td>9990.000000</td>
      <td>9990.000000</td>
      <td>9990.000000</td>
      <td>9990.000000</td>
      <td>9.990000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.0</td>
      <td>0.475676</td>
      <td>0.0</td>
      <td>1.067710e+18</td>
      <td>0.128729</td>
      <td>0.097297</td>
      <td>515.657858</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>13858.428729</td>
      <td>4783.830330</td>
      <td>4090.616116</td>
      <td>67.701301</td>
      <td>6.820953e+04</td>
      <td>0.091691</td>
      <td>0.480480</td>
      <td>6729.440777</td>
      <td>10984.707421</td>
      <td>659.235035</td>
      <td>78520.945946</td>
      <td>98.034535</td>
      <td>1.458152e+08</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.0</td>
      <td>2.276985</td>
      <td>0.0</td>
      <td>4.692141e+15</td>
      <td>0.334916</td>
      <td>0.296377</td>
      <td>4585.243177</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>22003.006032</td>
      <td>17187.454729</td>
      <td>15820.466275</td>
      <td>180.271814</td>
      <td>1.858952e+05</td>
      <td>0.166270</td>
      <td>0.499644</td>
      <td>9030.632130</td>
      <td>14817.549069</td>
      <td>1518.890057</td>
      <td>102493.616994</td>
      <td>48.927004</td>
      <td>1.009992e+08</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.041673e+18</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.740000e+02</td>
      <td>0.002730</td>
      <td>0.000000</td>
      <td>244.291457</td>
      <td>0.738549</td>
      <td>0.000000</td>
      <td>1273.000000</td>
      <td>0.000000</td>
      <td>1.292807e+07</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.067765e+18</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.000000</td>
      <td>192.000000</td>
      <td>20.000000</td>
      <td>3.000000</td>
      <td>4.580000e+03</td>
      <td>0.005213</td>
      <td>0.000000</td>
      <td>877.402010</td>
      <td>1034.589860</td>
      <td>0.000000</td>
      <td>7199.000000</td>
      <td>54.000000</td>
      <td>4.489316e+07</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.069833e+18</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1044.000000</td>
      <td>937.000000</td>
      <td>446.000000</td>
      <td>15.000000</td>
      <td>1.715000e+04</td>
      <td>0.021041</td>
      <td>0.000000</td>
      <td>2405.587940</td>
      <td>4058.401580</td>
      <td>2.000000</td>
      <td>38723.000000</td>
      <td>100.000000</td>
      <td>1.341711e+08</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.070294e+18</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>35.000000</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>17826.000000</td>
      <td>2743.000000</td>
      <td>2255.000000</td>
      <td>40.000000</td>
      <td>5.219900e+04</td>
      <td>0.075193</td>
      <td>1.000000</td>
      <td>8466.336683</td>
      <td>11861.843283</td>
      <td>7.000000</td>
      <td>107988.000000</td>
      <td>146.000000</td>
      <td>2.408530e+08</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.0</td>
      <td>126.000000</td>
      <td>0.0</td>
      <td>1.070615e+18</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>278544.000000</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>75918.000000</td>
      <td>118699.000000</td>
      <td>108323.000000</td>
      <td>1147.000000</td>
      <td>1.206346e+06</td>
      <td>0.661264</td>
      <td>1.000000</td>
      <td>35088.540816</td>
      <td>53243.158872</td>
      <td>7187.000000</td>
      <td>390659.000000</td>
      <td>160.000000</td>
      <td>3.456110e+08</td>
    </tr>
  </tbody>
</table>
</div>



    (9990, 31)




```python
# delete numerical columns that have mean or std equals 0 (which implies same values for the columns)
col_name_del = []
for col in col_pred_numerical:
    if tweets_df[col].mean() == 0 or tweets_df[col].std() == 0:
        del tweets_df[col]
        col_name_del.append(col)
        col_pred_numerical.remove(col)
display(tweets_df.shape)
print ('{} are deleted as they only have one values across all the rows.'.format(str(col_name_del)))
```



    (9990, 27)


    ['extended_entities', 'favorited', 'retweeted', 'truncated'] are deleted as they only have one values across all the rows.




```python
# save columns
c_list_names = ['col_pred_numerical', 'col_boto', 'col_response', 'col_pred_text', 'col_id', 'col_verified']
c_list = [col_pred_numerical, col_boto, col_response, col_pred_text, col_id, col_verified]
for c_name, c in zip(c_list_names, c_list):
    with open(c_name+'.txt', 'w') as fp:
        ls_str = ",".join(col_pred_numerical)
        fp.write(ls_str)
```


<mark> TO DO: add a bit more explainations <mark>

[Back to TOC](#TOC) <br/>
<a id ='Important-Features'></a>
#### 3.3 - Advanced Feature Engineering - NLP Features

<mark> add descriptions for each feature  <mark>



```python
col_nlp_text = ['tweet_len_mean', 'tweet_len_std', 'tweet_word_mean', 'tweet_word_std',
                'retweet_len_mean', 'retweet_len_std', 'retweet_word_mean', 'retweet_word_std']

with open('col_nlp_text.txt', 'w') as fp:
        ls_str = ",".join(col_nlp_text)
        fp.write(ls_str)
```




```python
# function to get tweet length
def get_tweet_lens(tweet_series):
  return tweet_series.dropna().map(lambda s: len(s))
```




```python
# function to get length of each word. filtering out hashtags, @, and links
def get_tweet_word_lens(tweet_series):
  tweets = tweet_series.dropna().values.tolist()
  words = [w for s in [t.split() for t in tweets] for w in s]
  filtered_words = filter(lambda w: not (w.startswith('@') or w.startswith('#') or w.startswith('http')), words)
  word_len = np.array([len(w) for w in filtered_words])
  return word_len
```




```python
# function to create feature
def tweet_text_features(df):
  cols = col_nlp_text
  tweet_lens = get_tweet_lens(df['text_tweet'])
  tweet_word_lens = get_tweet_word_lens(df['text_tweet'])
  retweet_lens = get_tweet_lens(df['text_rt'])
  retweet_word_lens = get_tweet_word_lens(df['text_rt'])
  
  return pd.Series((tweet_lens.mean(), tweet_lens.std(), 
                    tweet_word_lens.mean(), tweet_word_lens.std(),
                    retweet_lens.mean(), retweet_lens.std(), 
                    retweet_word_lens.mean(), retweet_word_lens.std()), index=cols)
```




```python
# get text features
text_df = tweets_df.groupby("screen_name").apply(tweet_text_features).reset_index()
```




```python
# merge text features with tweets_df
tweets_df = pd.merge(tweets_df, text_df, left_on='screen_name', right_on='screen_name')
```




```python
tweets_df.head()
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>favorite_count</th>
      <th>id</th>
      <th>in_reply_to_screen_name</th>
      <th>is_quote_status</th>
      <th>retweet_count</th>
      <th>retweeted_status</th>
      <th>user_description</th>
      <th>user_favourites_count</th>
      <th>user_followers_count</th>
      <th>user_friends_count</th>
      <th>user_listed_count</th>
      <th>user_location</th>
      <th>user_name</th>
      <th>user_profile_use_background_image</th>
      <th>user_statuses_count</th>
      <th>screen_name</th>
      <th>class_boto</th>
      <th>boto_univ</th>
      <th>class_verified</th>
      <th>text_rt</th>
      <th>text_tweet</th>
      <th>tweet_time_mean</th>
      <th>tweet_time_std</th>
      <th>tweet_time_min</th>
      <th>tweet_time_max</th>
      <th>user_description_len</th>
      <th>account_age</th>
      <th>tweet_len_mean</th>
      <th>tweet_len_std</th>
      <th>tweet_word_mean</th>
      <th>tweet_word_std</th>
      <th>retweet_len_mean</th>
      <th>retweet_len_std</th>
      <th>retweet_word_mean</th>
      <th>retweet_word_std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1070614540746977281</td>
      <td>0</td>
      <td>0</td>
      <td>278</td>
      <td>1</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>13997.0</td>
      <td>3080</td>
      <td>3073.0</td>
      <td>3.0</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>True</td>
      <td>108501</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
      <td>Have you seen @jeremycorbyn savaging the torie...</td>
      <td>None</td>
      <td>428.19598</td>
      <td>2753.761741</td>
      <td>1.0</td>
      <td>30530.0</td>
      <td>87</td>
      <td>133222476.0</td>
      <td>162.0</td>
      <td>65.630785</td>
      <td>4.068966</td>
      <td>2.468777</td>
      <td>163.222222</td>
      <td>79.950073</td>
      <td>4.608677</td>
      <td>2.542836</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1070614514238922753</td>
      <td>0</td>
      <td>0</td>
      <td>1586</td>
      <td>1</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>13997.0</td>
      <td>3080</td>
      <td>3073.0</td>
      <td>3.0</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>True</td>
      <td>108501</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
      <td>In your street there lives someone who moves t...</td>
      <td>None</td>
      <td>428.19598</td>
      <td>2753.761741</td>
      <td>1.0</td>
      <td>30530.0</td>
      <td>87</td>
      <td>133222476.0</td>
      <td>162.0</td>
      <td>65.630785</td>
      <td>4.068966</td>
      <td>2.468777</td>
      <td>163.222222</td>
      <td>79.950073</td>
      <td>4.608677</td>
      <td>2.542836</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1070614428859727872</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>1</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>13997.0</td>
      <td>3080</td>
      <td>3073.0</td>
      <td>3.0</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>True</td>
      <td>108501</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
      <td>The Unite general secretary, Len McCluskey war...</td>
      <td>None</td>
      <td>428.19598</td>
      <td>2753.761741</td>
      <td>1.0</td>
      <td>30530.0</td>
      <td>87</td>
      <td>133222476.0</td>
      <td>162.0</td>
      <td>65.630785</td>
      <td>4.068966</td>
      <td>2.468777</td>
      <td>163.222222</td>
      <td>79.950073</td>
      <td>4.608677</td>
      <td>2.542836</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1070614351076298753</td>
      <td>0</td>
      <td>0</td>
      <td>37</td>
      <td>1</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>13997.0</td>
      <td>3080</td>
      <td>3073.0</td>
      <td>3.0</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>True</td>
      <td>108501</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
      <td>“The Scottish Parliament, speaking for the Sco...</td>
      <td>None</td>
      <td>428.19598</td>
      <td>2753.761741</td>
      <td>1.0</td>
      <td>30530.0</td>
      <td>87</td>
      <td>133222476.0</td>
      <td>162.0</td>
      <td>65.630785</td>
      <td>4.068966</td>
      <td>2.468777</td>
      <td>163.222222</td>
      <td>79.950073</td>
      <td>4.608677</td>
      <td>2.542836</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1070614262857588736</td>
      <td>0</td>
      <td>0</td>
      <td>1263</td>
      <td>1</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>13997.0</td>
      <td>3080</td>
      <td>3073.0</td>
      <td>3.0</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>True</td>
      <td>108501</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
      <td>If there was a second EU referendum \n\nHow wo...</td>
      <td>None</td>
      <td>428.19598</td>
      <td>2753.761741</td>
      <td>1.0</td>
      <td>30530.0</td>
      <td>87</td>
      <td>133222476.0</td>
      <td>162.0</td>
      <td>65.630785</td>
      <td>4.068966</td>
      <td>2.468777</td>
      <td>163.222222</td>
      <td>79.950073</td>
      <td>4.608677</td>
      <td>2.542836</td>
    </tr>
  </tbody>
</table>
</div>



[Back to TOC](#TOC) <br/>
<a id ='Important-Features'></a>
#### 3.4 - Important Features

<mark> text to be refined - analysis and graphs <mark>



```python
tweets_df.head(5)
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>favorite_count</th>
      <th>id</th>
      <th>in_reply_to_screen_name</th>
      <th>is_quote_status</th>
      <th>retweet_count</th>
      <th>retweeted_status</th>
      <th>user_description</th>
      <th>user_favourites_count</th>
      <th>user_followers_count</th>
      <th>user_friends_count</th>
      <th>user_listed_count</th>
      <th>user_location</th>
      <th>user_name</th>
      <th>user_profile_use_background_image</th>
      <th>user_statuses_count</th>
      <th>screen_name</th>
      <th>class_boto</th>
      <th>boto_univ</th>
      <th>class_verified</th>
      <th>text_rt</th>
      <th>text_tweet</th>
      <th>tweet_time_mean</th>
      <th>tweet_time_std</th>
      <th>tweet_time_min</th>
      <th>tweet_time_max</th>
      <th>user_description_len</th>
      <th>account_age</th>
      <th>tweet_len_mean</th>
      <th>tweet_len_std</th>
      <th>tweet_word_mean</th>
      <th>tweet_word_std</th>
      <th>retweet_len_mean</th>
      <th>retweet_len_std</th>
      <th>retweet_word_mean</th>
      <th>retweet_word_std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1070614540746977281</td>
      <td>0</td>
      <td>0</td>
      <td>278</td>
      <td>1</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>13997.0</td>
      <td>3080</td>
      <td>3073.0</td>
      <td>3.0</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>True</td>
      <td>108501</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
      <td>Have you seen @jeremycorbyn savaging the torie...</td>
      <td>None</td>
      <td>428.19598</td>
      <td>2753.761741</td>
      <td>1.0</td>
      <td>30530.0</td>
      <td>87</td>
      <td>133222476.0</td>
      <td>162.0</td>
      <td>65.630785</td>
      <td>4.068966</td>
      <td>2.468777</td>
      <td>163.222222</td>
      <td>79.950073</td>
      <td>4.608677</td>
      <td>2.542836</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1070614514238922753</td>
      <td>0</td>
      <td>0</td>
      <td>1586</td>
      <td>1</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>13997.0</td>
      <td>3080</td>
      <td>3073.0</td>
      <td>3.0</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>True</td>
      <td>108501</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
      <td>In your street there lives someone who moves t...</td>
      <td>None</td>
      <td>428.19598</td>
      <td>2753.761741</td>
      <td>1.0</td>
      <td>30530.0</td>
      <td>87</td>
      <td>133222476.0</td>
      <td>162.0</td>
      <td>65.630785</td>
      <td>4.068966</td>
      <td>2.468777</td>
      <td>163.222222</td>
      <td>79.950073</td>
      <td>4.608677</td>
      <td>2.542836</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1070614428859727872</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>1</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>13997.0</td>
      <td>3080</td>
      <td>3073.0</td>
      <td>3.0</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>True</td>
      <td>108501</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
      <td>The Unite general secretary, Len McCluskey war...</td>
      <td>None</td>
      <td>428.19598</td>
      <td>2753.761741</td>
      <td>1.0</td>
      <td>30530.0</td>
      <td>87</td>
      <td>133222476.0</td>
      <td>162.0</td>
      <td>65.630785</td>
      <td>4.068966</td>
      <td>2.468777</td>
      <td>163.222222</td>
      <td>79.950073</td>
      <td>4.608677</td>
      <td>2.542836</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1070614351076298753</td>
      <td>0</td>
      <td>0</td>
      <td>37</td>
      <td>1</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>13997.0</td>
      <td>3080</td>
      <td>3073.0</td>
      <td>3.0</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>True</td>
      <td>108501</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
      <td>“The Scottish Parliament, speaking for the Sco...</td>
      <td>None</td>
      <td>428.19598</td>
      <td>2753.761741</td>
      <td>1.0</td>
      <td>30530.0</td>
      <td>87</td>
      <td>133222476.0</td>
      <td>162.0</td>
      <td>65.630785</td>
      <td>4.068966</td>
      <td>2.468777</td>
      <td>163.222222</td>
      <td>79.950073</td>
      <td>4.608677</td>
      <td>2.542836</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1070614262857588736</td>
      <td>0</td>
      <td>0</td>
      <td>1263</td>
      <td>1</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>13997.0</td>
      <td>3080</td>
      <td>3073.0</td>
      <td>3.0</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>True</td>
      <td>108501</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
      <td>If there was a second EU referendum \n\nHow wo...</td>
      <td>None</td>
      <td>428.19598</td>
      <td>2753.761741</td>
      <td>1.0</td>
      <td>30530.0</td>
      <td>87</td>
      <td>133222476.0</td>
      <td>162.0</td>
      <td>65.630785</td>
      <td>4.068966</td>
      <td>2.468777</td>
      <td>163.222222</td>
      <td>79.950073</td>
      <td>4.608677</td>
      <td>2.542836</td>
    </tr>
  </tbody>
</table>
</div>





```python
# separte bots and non-bots for easy plotting
tweets_0 = tweets_df.loc[tweets_df['class_verified']==0]
tweets_1 = tweets_df.loc[tweets_df['class_verified']==1]
```




```python
# scatter plot
def scatterplot (col1, col2, xlimit, ylimit):
    plt.scatter(tweets_1[col1], tweets_1[col2], s=5, color='salmon', label='bot', alpha=0.75)
    plt.scatter(tweets_0[col1], tweets_0[col2], s=5, color='royalblue', label='non-bot', alpha=0.75)
    plt.xlabel(str(col1))
    plt.ylabel(str(col2))
    plt.xlim(xlimit)
    plt.ylim(ylimit)
    plt.legend()
    title = str(col1) + ' vs ' + str(col2)
    plt.title(title)
    plt.savefig(str(title)+'.png')
```




```python
# histogram
def hist_plot(col, xlabel, ylabel, title):
    plt.hist(col)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(str(title)+'.png')
    return None
```




```python
# quick plots
plt.figure(figsize=(6,4))
hist_plot(tweets_0['tweet_time_min'], 'tweets_minimum_time_interval','count', 'minimum time interval among all tweets for each real user')
```



![png](Final_Project_TestFile_files/Final_Project_TestFile_89_0.png)




```python
# quick plots
plt.figure(figsize=(6,4))
hist_plot(tweets_1['tweet_time_min'], 'tweets_minimum_time_interval','count', 'minimum time interval among all tweets for each bot')
```



![png](Final_Project_TestFile_files/Final_Project_TestFile_90_0.png)




```python
# quick plots
plt.figure(figsize=(6,4))
hist_plot(tweets_0['tweet_time_std'], 'tweets_minimum_time_interval','count', 'std time interval among all tweets for each real user')
```



![png](Final_Project_TestFile_files/Final_Project_TestFile_91_0.png)




```python
# quick plots
plt.figure(figsize=(6,4))
hist_plot(tweets_1['tweet_time_std'], 'tweets_minimum_time_interval','count', 'std time interval among all tweets for each bot')
```



![png](Final_Project_TestFile_files/Final_Project_TestFile_92_0.png)


From the four plots above, it looks like the bots have significantly smaller std of tweet interval times (which implies more uniform patter) than actual users. <br/>

<mark> need to show same range for comparisons <mark>



```python
# quick plots
plt.figure(figsize=(6,4))
scatterplot('user_description_len', 'tweet_time_std', [0,200], [0,50000])
```



![png](Final_Project_TestFile_files/Final_Project_TestFile_94_0.png)


It was quite obvious from the plot above that non-bot users tend to have account description with mostly around maximum allowed characters. In contrast, the bots tend to have significantly shorter descriptions and much more even distribution. <br/>

The standard deviation of tweet time interval for each account seems be able to tell bot and non-bot apart very well - the bots tend to have significantly smaller tweet_time_std, while the non-bots tend to have much larger std.



```python
# quick plots
plt.figure(figsize=(6,4))
scatterplot('account_age', 'tweet_time_min', [0,400000000], [0,10])
```



![png](Final_Project_TestFile_files/Final_Project_TestFile_96_0.png)


it seems that bots tend to have longer min tweet intervals with newer accounts, and longer tweet time itnerval with older accounts. In contrast, it seems non-bot users have shorter minimum tweet intervals with newer accounts; however, the minimum tweet interval increase with the increase of account age.



```python
# quick plots
plt.figure(figsize=(6,4))
scatterplot('account_age', 'user_statuses_count', [0,400000000], [0,100000])
```



![png](Final_Project_TestFile_files/Final_Project_TestFile_98_0.png)


It seems the bot tend to have significantly shorter account age and significantly more tweets than non-bot accounts with the same account age.

[Back to TOC](#TOC) <br/>
<a id ='Relations-in-Data'></a>
#### 3.5 - Relations in Data

<mark> draft draft - NEED UPDATES <mark>



```python
# correlation matrix
# to be updated

fig, ax = plt.subplots()

col_corr = ['user_followers_count', 'user_friends_count', 'user_description_len', 'retweet_count', 'user_favourites_count']
labels_corr = ['follower', 'friends', 'user_description', 'retweet', 'favorite']
ax.matshow(tweets_df[col_corr].corr())
ax.set_xticklabels([''] + labels_corr)
ax.set_yticklabels([''] + labels_corr);
```



![png](Final_Project_TestFile_files/Final_Project_TestFile_101_0.png)




```python
# to be refined

scatter_matrix(tweets_df[col_pred_numerical], alpha=0.5, figsize=(25,20));
```




```python
# correlation matrix - to be udpated
pd.DataFrame(tweets_df[col_corr].corr())
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_followers_count</th>
      <th>user_friends_count</th>
      <th>user_description_len</th>
      <th>retweet_count</th>
      <th>user_favourites_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>user_followers_count</th>
      <td>1.000000</td>
      <td>0.991854</td>
      <td>-0.017684</td>
      <td>-0.019123</td>
      <td>-0.094587</td>
    </tr>
    <tr>
      <th>user_friends_count</th>
      <td>0.991854</td>
      <td>1.000000</td>
      <td>0.000681</td>
      <td>-0.013643</td>
      <td>-0.072485</td>
    </tr>
    <tr>
      <th>user_description_len</th>
      <td>-0.017684</td>
      <td>0.000681</td>
      <td>1.000000</td>
      <td>0.020562</td>
      <td>0.329083</td>
    </tr>
    <tr>
      <th>retweet_count</th>
      <td>-0.019123</td>
      <td>-0.013643</td>
      <td>0.020562</td>
      <td>1.000000</td>
      <td>0.101225</td>
    </tr>
    <tr>
      <th>user_favourites_count</th>
      <td>-0.094587</td>
      <td>-0.072485</td>
      <td>0.329083</td>
      <td>0.101225</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



[Back to TOC](#TOC) <br/>
<a id ='Standardization'></a>
#### 3.6- Standardization

standardize numerical features



```python
# current feature types
tweets_df.dtypes
```





    favorite_count                         int64
    id                                     int64
    in_reply_to_screen_name                int64
    is_quote_status                        int64
    retweet_count                          int64
    retweeted_status                       int64
    user_description                      object
    user_favourites_count                float64
    user_followers_count                   int64
    user_friends_count                   float64
    user_listed_count                    float64
    user_location                         object
    user_name                             object
    user_profile_use_background_image     object
    user_statuses_count                    int64
    screen_name                           object
    class_boto                            object
    boto_univ                            float64
    class_verified                       float64
    text_rt                               object
    text_tweet                            object
    tweet_time_mean                      float64
    tweet_time_std                       float64
    tweet_time_min                       float64
    tweet_time_max                       float64
    user_description_len                   int64
    account_age                          float64
    tweet_len_mean                       float64
    tweet_len_std                        float64
    tweet_word_mean                      float64
    tweet_word_std                       float64
    retweet_len_mean                     float64
    retweet_len_std                      float64
    retweet_word_mean                    float64
    retweet_word_std                     float64
    dtype: object





```python
from sklearn import preprocessing

def standardize(df):
    col_names = df.columns.values
    scaler = preprocessing.StandardScaler()
    df_std = pd.DataFrame(scaler.fit_transform(df), columns=col_names)
    return df_std
```




```python
# create a new copy with numercial columns standardized
tweets_df[col_pred_numerical] = standardize(tweets_df[col_pred_numerical])
```




```python
# check if the copy 
display(tweets_df.describe())
display(tweets_df.head())
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>favorite_count</th>
      <th>id</th>
      <th>in_reply_to_screen_name</th>
      <th>is_quote_status</th>
      <th>retweet_count</th>
      <th>retweeted_status</th>
      <th>user_favourites_count</th>
      <th>user_followers_count</th>
      <th>user_friends_count</th>
      <th>user_listed_count</th>
      <th>user_statuses_count</th>
      <th>boto_univ</th>
      <th>class_verified</th>
      <th>tweet_time_mean</th>
      <th>tweet_time_std</th>
      <th>tweet_time_min</th>
      <th>tweet_time_max</th>
      <th>user_description_len</th>
      <th>account_age</th>
      <th>tweet_len_mean</th>
      <th>tweet_len_std</th>
      <th>tweet_word_mean</th>
      <th>tweet_word_std</th>
      <th>retweet_len_mean</th>
      <th>retweet_len_std</th>
      <th>retweet_word_mean</th>
      <th>retweet_word_std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9.990000e+03</td>
      <td>9.990000e+03</td>
      <td>9.990000e+03</td>
      <td>9.990000e+03</td>
      <td>9.990000e+03</td>
      <td>9990.0</td>
      <td>9.990000e+03</td>
      <td>9.990000e+03</td>
      <td>9.990000e+03</td>
      <td>9.990000e+03</td>
      <td>9.990000e+03</td>
      <td>9990.000000</td>
      <td>9990.000000</td>
      <td>9.990000e+03</td>
      <td>9.990000e+03</td>
      <td>9.990000e+03</td>
      <td>9.990000e+03</td>
      <td>9.990000e+03</td>
      <td>9990.000000</td>
      <td>9790.000000</td>
      <td>9790.000000</td>
      <td>9790.000000</td>
      <td>9790.000000</td>
      <td>5590.000000</td>
      <td>5590.000000</td>
      <td>5590.000000</td>
      <td>5590.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.138006e-17</td>
      <td>1.067710e+18</td>
      <td>-1.138006e-17</td>
      <td>-2.276013e-17</td>
      <td>-8.535048e-18</td>
      <td>0.0</td>
      <td>-2.276013e-17</td>
      <td>1.138006e-17</td>
      <td>3.983022e-17</td>
      <td>-1.138006e-17</td>
      <td>-2.276013e-17</td>
      <td>0.091691</td>
      <td>0.480480</td>
      <td>1.536309e-16</td>
      <td>-2.276013e-17</td>
      <td>2.276013e-17</td>
      <td>1.024206e-16</td>
      <td>8.535048e-17</td>
      <td>0.000000</td>
      <td>118.687034</td>
      <td>46.989872</td>
      <td>4.766072</td>
      <td>2.638058</td>
      <td>191.909435</td>
      <td>80.416308</td>
      <td>4.697703</td>
      <td>2.608362</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.000050e+00</td>
      <td>4.692141e+15</td>
      <td>1.000050e+00</td>
      <td>1.000050e+00</td>
      <td>1.000050e+00</td>
      <td>0.0</td>
      <td>1.000050e+00</td>
      <td>1.000050e+00</td>
      <td>1.000050e+00</td>
      <td>1.000050e+00</td>
      <td>1.000050e+00</td>
      <td>0.166270</td>
      <td>0.499644</td>
      <td>1.000050e+00</td>
      <td>1.000050e+00</td>
      <td>1.000050e+00</td>
      <td>1.000050e+00</td>
      <td>1.000050e+00</td>
      <td>1.000050</td>
      <td>45.197972</td>
      <td>34.712563</td>
      <td>0.687878</td>
      <td>0.570048</td>
      <td>19.645234</td>
      <td>7.968795</td>
      <td>0.101774</td>
      <td>0.086884</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.089164e-01</td>
      <td>1.041673e+18</td>
      <td>-3.843802e-01</td>
      <td>-3.283054e-01</td>
      <td>-1.124659e-01</td>
      <td>0.0</td>
      <td>-6.298740e-01</td>
      <td>-2.780557e-01</td>
      <td>-2.585778e-01</td>
      <td>-3.755700e-01</td>
      <td>-3.643929e-01</td>
      <td>0.002730</td>
      <td>0.000000</td>
      <td>-7.181639e-01</td>
      <td>-7.413182e-01</td>
      <td>-4.340459e-01</td>
      <td>-7.537232e-01</td>
      <td>-2.003790e+00</td>
      <td>-1.315791</td>
      <td>14.050000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>1.361082</td>
      <td>147.967213</td>
      <td>65.687936</td>
      <td>4.501567</td>
      <td>2.365080</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-2.089164e-01</td>
      <td>1.067765e+18</td>
      <td>-3.843802e-01</td>
      <td>-3.283054e-01</td>
      <td>-1.124659e-01</td>
      <td>0.0</td>
      <td>-6.297831e-01</td>
      <td>-2.671751e-01</td>
      <td>-2.573135e-01</td>
      <td>-3.589277e-01</td>
      <td>-3.423041e-01</td>
      <td>0.005213</td>
      <td>0.000000</td>
      <td>-6.480534e-01</td>
      <td>-6.715426e-01</td>
      <td>-4.340459e-01</td>
      <td>-6.959020e-01</td>
      <td>-9.000498e-01</td>
      <td>-0.999287</td>
      <td>94.000000</td>
      <td>15.132380</td>
      <td>4.350230</td>
      <td>2.335403</td>
      <td>175.037037</td>
      <td>73.715149</td>
      <td>4.615609</td>
      <td>2.557134</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-2.089164e-01</td>
      <td>1.069833e+18</td>
      <td>-3.843802e-01</td>
      <td>-3.283054e-01</td>
      <td>-1.124659e-01</td>
      <td>0.0</td>
      <td>-5.824236e-01</td>
      <td>-2.238274e-01</td>
      <td>-2.303850e-01</td>
      <td>-2.923582e-01</td>
      <td>-2.746820e-01</td>
      <td>0.021041</td>
      <td>0.000000</td>
      <td>-4.788224e-01</td>
      <td>-4.674628e-01</td>
      <td>-4.327291e-01</td>
      <td>-3.883163e-01</td>
      <td>4.017339e-02</td>
      <td>-0.115295</td>
      <td>118.500000</td>
      <td>36.380043</td>
      <td>4.611089</td>
      <td>2.536109</td>
      <td>201.076923</td>
      <td>79.579214</td>
      <td>4.689600</td>
      <td>2.615336</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>-2.089164e-01</td>
      <td>1.070294e+18</td>
      <td>-3.843802e-01</td>
      <td>-3.283054e-01</td>
      <td>-1.048324e-01</td>
      <td>0.0</td>
      <td>1.803285e-01</td>
      <td>-1.187455e-01</td>
      <td>-1.160337e-01</td>
      <td>-1.536718e-01</td>
      <td>-8.613091e-02</td>
      <td>0.075193</td>
      <td>1.000000</td>
      <td>1.923434e-01</td>
      <td>5.919871e-02</td>
      <td>-4.294371e-01</td>
      <td>2.875157e-01</td>
      <td>9.803966e-01</td>
      <td>0.941024</td>
      <td>138.085000</td>
      <td>78.848861</td>
      <td>4.945022</td>
      <td>2.922850</td>
      <td>203.521127</td>
      <td>85.912335</td>
      <td>4.775988</td>
      <td>2.657264</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.513018e+01</td>
      <td>1.070615e+18</td>
      <td>2.601591e+00</td>
      <td>3.045944e+00</td>
      <td>6.063850e+01</td>
      <td>0.0</td>
      <td>2.820645e+00</td>
      <td>6.628141e+00</td>
      <td>6.588782e+00</td>
      <td>5.987363e+00</td>
      <td>6.122768e+00</td>
      <td>0.661264</td>
      <td>1.000000</td>
      <td>3.140480e+00</td>
      <td>2.852062e+00</td>
      <td>4.297936e+00</td>
      <td>3.045591e+00</td>
      <td>1.266551e+00</td>
      <td>1.978291</td>
      <td>222.770000</td>
      <td>105.807394</td>
      <td>8.035714</td>
      <td>5.134353</td>
      <td>222.006536</td>
      <td>102.230390</td>
      <td>4.910280</td>
      <td>2.749550</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>favorite_count</th>
      <th>id</th>
      <th>in_reply_to_screen_name</th>
      <th>is_quote_status</th>
      <th>retweet_count</th>
      <th>retweeted_status</th>
      <th>user_description</th>
      <th>user_favourites_count</th>
      <th>user_followers_count</th>
      <th>user_friends_count</th>
      <th>user_listed_count</th>
      <th>user_location</th>
      <th>user_name</th>
      <th>user_profile_use_background_image</th>
      <th>user_statuses_count</th>
      <th>screen_name</th>
      <th>class_boto</th>
      <th>boto_univ</th>
      <th>class_verified</th>
      <th>text_rt</th>
      <th>text_tweet</th>
      <th>tweet_time_mean</th>
      <th>tweet_time_std</th>
      <th>tweet_time_min</th>
      <th>tweet_time_max</th>
      <th>user_description_len</th>
      <th>account_age</th>
      <th>tweet_len_mean</th>
      <th>tweet_len_std</th>
      <th>tweet_word_mean</th>
      <th>tweet_word_std</th>
      <th>retweet_len_mean</th>
      <th>retweet_len_std</th>
      <th>retweet_word_mean</th>
      <th>retweet_word_std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.208916</td>
      <td>1070614540746977281</td>
      <td>-0.38438</td>
      <td>-0.328305</td>
      <td>-0.051834</td>
      <td>0.0</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>0.006298</td>
      <td>-0.099137</td>
      <td>-0.064326</td>
      <td>-0.358928</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>True</td>
      <td>0.216754</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
      <td>Have you seen @jeremycorbyn savaging the torie...</td>
      <td>None</td>
      <td>-0.697798</td>
      <td>-0.555514</td>
      <td>-0.433388</td>
      <td>-0.468257</td>
      <td>-0.225542</td>
      <td>-0.124688</td>
      <td>162.0</td>
      <td>65.630785</td>
      <td>4.068966</td>
      <td>2.468777</td>
      <td>163.222222</td>
      <td>79.950073</td>
      <td>4.608677</td>
      <td>2.542836</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.208916</td>
      <td>1070614514238922753</td>
      <td>-0.38438</td>
      <td>-0.328305</td>
      <td>0.233444</td>
      <td>0.0</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>0.006298</td>
      <td>-0.099137</td>
      <td>-0.064326</td>
      <td>-0.358928</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>True</td>
      <td>0.216754</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
      <td>In your street there lives someone who moves t...</td>
      <td>None</td>
      <td>-0.697798</td>
      <td>-0.555514</td>
      <td>-0.433388</td>
      <td>-0.468257</td>
      <td>-0.225542</td>
      <td>-0.124688</td>
      <td>162.0</td>
      <td>65.630785</td>
      <td>4.068966</td>
      <td>2.468777</td>
      <td>163.222222</td>
      <td>79.950073</td>
      <td>4.608677</td>
      <td>2.542836</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.208916</td>
      <td>1070614428859727872</td>
      <td>-0.38438</td>
      <td>-0.328305</td>
      <td>-0.109194</td>
      <td>0.0</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>0.006298</td>
      <td>-0.099137</td>
      <td>-0.064326</td>
      <td>-0.358928</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>True</td>
      <td>0.216754</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
      <td>The Unite general secretary, Len McCluskey war...</td>
      <td>None</td>
      <td>-0.697798</td>
      <td>-0.555514</td>
      <td>-0.433388</td>
      <td>-0.468257</td>
      <td>-0.225542</td>
      <td>-0.124688</td>
      <td>162.0</td>
      <td>65.630785</td>
      <td>4.068966</td>
      <td>2.468777</td>
      <td>163.222222</td>
      <td>79.950073</td>
      <td>4.608677</td>
      <td>2.542836</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.208916</td>
      <td>1070614351076298753</td>
      <td>-0.38438</td>
      <td>-0.328305</td>
      <td>-0.104396</td>
      <td>0.0</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>0.006298</td>
      <td>-0.099137</td>
      <td>-0.064326</td>
      <td>-0.358928</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>True</td>
      <td>0.216754</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
      <td>“The Scottish Parliament, speaking for the Sco...</td>
      <td>None</td>
      <td>-0.697798</td>
      <td>-0.555514</td>
      <td>-0.433388</td>
      <td>-0.468257</td>
      <td>-0.225542</td>
      <td>-0.124688</td>
      <td>162.0</td>
      <td>65.630785</td>
      <td>4.068966</td>
      <td>2.468777</td>
      <td>163.222222</td>
      <td>79.950073</td>
      <td>4.608677</td>
      <td>2.542836</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.208916</td>
      <td>1070614262857588736</td>
      <td>-0.38438</td>
      <td>-0.328305</td>
      <td>0.162997</td>
      <td>0.0</td>
      <td>INDEPENDENCE OR BUST ..... Tweets and retweets...</td>
      <td>0.006298</td>
      <td>-0.099137</td>
      <td>-0.064326</td>
      <td>-0.358928</td>
      <td>SCOTLAND</td>
      <td>Do McK</td>
      <td>True</td>
      <td>0.216754</td>
      <td>do_mck</td>
      <td>0</td>
      <td>0.005213</td>
      <td>0.0</td>
      <td>If there was a second EU referendum \n\nHow wo...</td>
      <td>None</td>
      <td>-0.697798</td>
      <td>-0.555514</td>
      <td>-0.433388</td>
      <td>-0.468257</td>
      <td>-0.225542</td>
      <td>-0.124688</td>
      <td>162.0</td>
      <td>65.630785</td>
      <td>4.068966</td>
      <td>2.468777</td>
      <td>163.222222</td>
      <td>79.950073</td>
      <td>4.608677</td>
      <td>2.542836</td>
    </tr>
  </tbody>
</table>
</div>




```python
# save to json
tweets_df.to_json('50_accounts_200_tweets_each_final_std.json')
```


[Back to TOC](#TOC) <br/>
<a id ='Models'></a>
### 4 - Models

<mark> Some Text Here </mark> <br>
<mark> for threshold, we are using 0.5 across all models </mark>



```python
# read the data
tweets_df = pd.read_json('50_accounts_200_tweets_each_final_std.json')
```




```python
# Train/Test split 
'''
change as needed, do we want test_size of .25?
'''
train_tweets_df, test_tweets_df = train_test_split(tweets_df, test_size=.25, 
                                                       stratify=tweets_df.class_verified, random_state=99)
```




```python
with open('col_pred_numerical.txt', 'r') as fp:
  col_pred_numerical = fp.read().split(',')
with open('col_boto.txt', 'r') as fp:
  col_pred_numerical = fp.read().split(',')
with open('col_response.txt', 'r') as fp:
  col_pred_numerical = fp.read().split(',')
with open('col_pred_text.txt', 'r') as fp:
  col_pred_numerical = fp.read().split(',')
with open('col_id.txt', 'r') as fp:
  col_pred_numerical = fp.read().split(',')
with open('col_verified.txt', 'r') as fp:
  col_pred_numerical = fp.read().split(',')
```




```python
# write a function to split the data
def split_data(df):
    # num_pred: standardized numerical predictors - what we will be using for most of the models
    # text_pred: text features that are associated with the tweets - only useful for NLP
    # response: response - manually verified classification. 1=bot; 0=non-bot
    # ids: 'id'
    # boto: botometer values
    num_pred, text_pred, response = df[col_pred_numerical], df[col_pred_text], df['class_verified']
    ids, boto = df['id'], df[col_boto]
    return num_pred, text_pred, response, ids, boto
```




```python
# get the predictors, responses, and other features from train and test set
xtrain, xtrain_text, ytrain, train_id, train_boto = split_data(train_tweets_df)
xtest, xtest_text, ytest, test_id, test_boto = split_data(test_tweets_df)
```




```python
# create a dictioary to store all our models
models_list = {}
```


[Back to TOC](#TOC) <br/>
<a id ='Baseline-Model'></a>
#### 4.1 - Baseline Model - Naive Bayes

<mark> TO DO, optional?  </mark>

[Back to TOC](#TOC) <br/>
<a id ='Baseline-Model'></a>
#### 4.2 - Linear Regression

<mark> TO DO: Samsara </mark>



```python
# multiple linear regression(no poly)on numerical predictors
X_train = sm.add_constant(xtrain)
X_test = sm.add_constant(xtest)
y_train = ytrain.reshape(-1,1)
y_test = ytest.reshape(-1,1)
```




```python
# Fit and summarize OLS model
model = OLS(y_train, X_train)
results = model.fit()

results.summary()
```





<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.484</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.483</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   413.1</td>
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 10 Dec 2018</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>01:19:31</td>     <th>  Log-Likelihood:    </th> <td> -2950.1</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>  7492</td>      <th>  AIC:               </th> <td>   5936.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  7474</td>      <th>  BIC:               </th> <td>   6061.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    17</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
             <td></td>                <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                   <td>    0.4792</td> <td>    0.004</td> <td>  115.450</td> <td> 0.000</td> <td>    0.471</td> <td>    0.487</td>
</tr>
<tr>
  <th>extended_entities</th>       <td>    0.0384</td> <td>    0.005</td> <td>    7.679</td> <td> 0.000</td> <td>    0.029</td> <td>    0.048</td>
</tr>
<tr>
  <th>favorite_count</th>          <td>    0.0048</td> <td>    0.004</td> <td>    1.174</td> <td> 0.241</td> <td>   -0.003</td> <td>    0.013</td>
</tr>
<tr>
  <th>in_reply_to_screen_name</th> <td>   -0.1882</td> <td>    0.006</td> <td>  -33.468</td> <td> 0.000</td> <td>   -0.199</td> <td>   -0.177</td>
</tr>
<tr>
  <th>is_quote_status</th>         <td>   -0.0361</td> <td>    0.004</td> <td>   -8.228</td> <td> 0.000</td> <td>   -0.045</td> <td>   -0.028</td>
</tr>
<tr>
  <th>retweet_count</th>           <td>   -0.0064</td> <td>    0.004</td> <td>   -1.497</td> <td> 0.134</td> <td>   -0.015</td> <td>    0.002</td>
</tr>
<tr>
  <th>retweeted_status</th>        <td>   -0.2210</td> <td>    0.007</td> <td>  -32.012</td> <td> 0.000</td> <td>   -0.235</td> <td>   -0.207</td>
</tr>
<tr>
  <th>user_favourites_count</th>   <td>   -0.0990</td> <td>    0.005</td> <td>  -19.563</td> <td> 0.000</td> <td>   -0.109</td> <td>   -0.089</td>
</tr>
<tr>
  <th>user_followers_count</th>    <td>   -0.4058</td> <td>    0.038</td> <td>  -10.554</td> <td> 0.000</td> <td>   -0.481</td> <td>   -0.330</td>
</tr>
<tr>
  <th>user_friends_count</th>      <td>    0.3922</td> <td>    0.044</td> <td>    9.012</td> <td> 0.000</td> <td>    0.307</td> <td>    0.477</td>
</tr>
<tr>
  <th>user_listed_count</th>       <td>    0.0433</td> <td>    0.007</td> <td>    5.982</td> <td> 0.000</td> <td>    0.029</td> <td>    0.058</td>
</tr>
<tr>
  <th>user_statuses_count</th>     <td>   -0.1154</td> <td>    0.026</td> <td>   -4.447</td> <td> 0.000</td> <td>   -0.166</td> <td>   -0.065</td>
</tr>
<tr>
  <th>tweet_time_mean</th>         <td>   -0.1016</td> <td>    0.013</td> <td>   -7.589</td> <td> 0.000</td> <td>   -0.128</td> <td>   -0.075</td>
</tr>
<tr>
  <th>tweet_time_std</th>          <td>   -0.3413</td> <td>    0.021</td> <td>  -16.401</td> <td> 0.000</td> <td>   -0.382</td> <td>   -0.300</td>
</tr>
<tr>
  <th>tweet_time_min</th>          <td>    0.0445</td> <td>    0.006</td> <td>    7.927</td> <td> 0.000</td> <td>    0.033</td> <td>    0.055</td>
</tr>
<tr>
  <th>tweet_time_max</th>          <td>    0.3274</td> <td>    0.021</td> <td>   15.827</td> <td> 0.000</td> <td>    0.287</td> <td>    0.368</td>
</tr>
<tr>
  <th>user_description_len</th>    <td>    0.0457</td> <td>    0.005</td> <td>    8.849</td> <td> 0.000</td> <td>    0.036</td> <td>    0.056</td>
</tr>
<tr>
  <th>account_age</th>             <td>   -0.0050</td> <td>    0.005</td> <td>   -0.980</td> <td> 0.327</td> <td>   -0.015</td> <td>    0.005</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>32.539</td> <th>  Durbin-Watson:     </th> <td>   1.963</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  32.840</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.160</td> <th>  Prob(JB):          </th> <td>7.39e-08</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.056</td> <th>  Cond. No.          </th> <td>    27.1</td>
</tr>
</table>





```python
y_hat_train = results.predict()
y_hat_test = results.predict(exog=X_test)

# get Train & Test R^2
print('Train R^2 = {}'.format(results.rsquared))
print('Test R^2 = {}'.format(r2_score(test_tweets_df['class_verified'], y_hat_test)))
```


    Train R^2 = 0.4844590700814966
    Test R^2 = 0.4832952963654841




```python
# accuracy score
ols_train_acc = accuracy_score(y_train, results.predict(X_train).round())
ols_test_acc = accuracy_score(y_test, results.predict(X_test).round())
print("Training accuracy is {:.4}%".format(ols_train_acc*100))
print("Test accuracy is {:.4} %".format(ols_test_acc*100))
```


    Training accuracy is 82.84%
    Test accuracy is 82.39 %




```python
# save model to the list
models_list["ols"] = model
```


[Back to TOC](#TOC) <br/>
<a id ='Baseline-Model'></a>
#### 4.2a - Ridge

<mark> huan: I think we'll need to include some kind of dimension reduction techniques therefore added lasso and ridge, thinking about pca as well <mark>



```python
alphas = np.array([.01, .05, .1, .5, 1, 5, 10, 50, 100])
fitted_ridge = RidgeCV(alphas=alphas, cv=5).fit(X_train, y_train)
```




```python
# accuracy score
ridge_train_acc = accuracy_score(y_train, fitted_ridge.predict(X_train).round())
ridge_test_acc = accuracy_score(y_test, fitted_ridge.predict(X_test).round())
print("Training accuracy is {:.4}%".format(ridge_train_acc*100))
print("Test accuracy is {:.4} %".format(ridge_test_acc*100))
```


    Training accuracy is 82.84%
    Test accuracy is 82.39 %




```python
# save model to the list
models_list["ridge"] = fitted_ridge
```


[Back to TOC](#TOC) <br/>
<a id ='Baseline-Model'></a>
#### 4.2b - Lasso
<mark> huan: I think we'll need to include some kind of dimension reduction techniques therefore added lasso and ridge, thinking about pca as well <mark>



```python
fitted_lasso = LassoCV(alphas=alphas, max_iter=100000, cv=5).fit(X_train, y_train)
```




```python
# accuracy score
lasso_train_acc = accuracy_score(y_train, fitted_lasso.predict(X_train).round())
lasso_test_acc = accuracy_score(y_test, fitted_lasso.predict(X_test).round())
print("Training accuracy is {:.4}%".format(lasso_train_acc*100))
print("Test accuracy is {:.4} %".format(lasso_test_acc*100))
```


    Training accuracy is 82.29%
    Test accuracy is 82.03 %




```python
# save model to the list
models_list["lasso"] = fitted_lasso
```


[Back to TOC](#TOC) <br/>
<a id ='Baseline-Model'></a>
#### 4.2c - Comparison of OLS, Ridge, and Lasso

<mark> TO DO </mark>

[Back to TOC](#TOC) <br/>
<a id ='Baseline-Model'></a>
#### 4.3 - Logistic Regression

<mark> Nisrine </mark>



```python
X_train = sm.add_constant(xtrain)
X_test = sm.add_constant(xtest)

logistic_model = LogisticRegression().fit(X_train, ytrain)

logistic_model_score = logistic_model.score(X_test, ytest)

models_list["simple_logistic"] = logistic_model

print("Train set score: {0:4.4}%".format(logistic_model.score(X_train, ytrain)*100))
print("Test set score: {0:4.4}%".format(logistic_model.score(X_test, ytest)*100))
```


    Train set score: 81.02%
    Test set score: 80.94%


[Back to TOC](#TOC) <br/>
<a id ='Logistic-Regression'></a>
#### 4.3a - Logistic Regression with cross validation

<mark> Nisrine, team please check </mark>



```python
logistic_model_cv = LogisticRegressionCV(Cs=[1,10,100,1000,10000], cv=3, penalty='l2', 
                                       solver='newton-cg').fit(X_train,ytrain)

models_list["simple_logistic_Cross_Validation"] = logistic_model_cv

print("Train set score with Cross Validation: {0:4.4}%".format(logistic_model_cv.score(X_train, ytrain)*100))
print("Test set score with Cross Validation: {0:4.4}%".format(logistic_model_cv.score(X_test, ytest)*100))
```


    Train set score with Cross Validation: 81.07%
    Test set score with Cross Validation: 81.06%


[Back to TOC](#TOC) <br/>
<a id ='Logistic-Regression'></a>
#### 4.3b - Logistic Regression with polynomial degree 3

<mark> Nisrine, team please check </mark>



```python
X_train_poly = PolynomialFeatures(degree=3, include_bias=False).fit_transform(X_train)

logistic_model_poly_cv = LogisticRegressionCV(Cs=[1,10,100,1000,10000], cv=3, penalty='l2', 
                                       solver='newton-cg').fit(X_train_poly,ytrain)

models_list["poly_logistic_cv"] = logistic_model_poly_cv

X_test_poly = PolynomialFeatures(degree=3, include_bias=False).fit_transform(X_test)
print("Train set score with Polynomial Features (degree=3) and with Cross Validation: {0:4.4}%".
      format(logistic_model_poly_cv.score(X_train_poly, ytrain)*100))
print("Test set score with Polynomial Features (degree=3) and with Cross Validation: {0:4.4}%".
      format(logistic_model_poly_cv.score(X_test_poly, ytest)*100))
```


    Train set score with Polynomial Features (degree=3) and with Cross Validation: 96.89%
    Test set score with Polynomial Features (degree=3) and with Cross Validation: 96.96%


[Back to TOC](#TOC) <br/>
<a id ='KNN'></a>
#### 4.4 - KNN

<mark> TO DO: Huan </mark>



```python
# the code below in KNN is adapted from HW2 solution

# define k values
k_values = range(1,20)

# build a dictionary KNN models
KNNModels = {k: KNeighborsRegressor(n_neighbors=k) for k in k_values}
train_scores = [KNeighborsRegressor(n_neighbors=k).fit(xtrain, ytrain).score(xtrain, ytrain) for k in k_values]
cv_scores = [cross_val_score(KNeighborsRegressor(n_neighbors=k), xtrain, ytrain, cv=5) for k in k_values]


# fit each KNN model
for k_value in KNNModels: 
    KNNModels[k_value].fit(xtrain, ytrain)
```




```python
# Generate predictions
knn_predicted_train = {k: KNNModels[k].predict(xtrain) for k in KNNModels}
knn_predicted_test = {k: KNNModels[k].predict(xtest) for k in KNNModels}
```




```python
# create a df  of train test rsquare values with corresponding k values
knn_r2_train = {k : r2_score(ytrain, knn_predicted_train[k]) for k in k_values}
knn_r2_test = { k : r2_score(ytest, knn_predicted_test[k]) for k in k_values}

knn_r2_df = pd.DataFrame(data = {"k" : tuple(knn_r2_train.keys()), 
                                    "Train R^2" : tuple(knn_r2_train.values()), 
                                    "Test R^2" : tuple(knn_r2_test.values())})
display(knn_r2_df)
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test R^2</th>
      <th>Train R^2</th>
      <th>k</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.821984</td>
      <td>0.909098</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.870497</td>
      <td>0.905756</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.875264</td>
      <td>0.906722</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.877814</td>
      <td>0.905422</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.878307</td>
      <td>0.900778</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.879051</td>
      <td>0.900513</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.878999</td>
      <td>0.896287</td>
      <td>7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.879293</td>
      <td>0.895179</td>
      <td>8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.876313</td>
      <td>0.892654</td>
      <td>9</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.875180</td>
      <td>0.890528</td>
      <td>10</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.873781</td>
      <td>0.887829</td>
      <td>11</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.872903</td>
      <td>0.885820</td>
      <td>12</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.870846</td>
      <td>0.882970</td>
      <td>13</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.869622</td>
      <td>0.881154</td>
      <td>14</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.869555</td>
      <td>0.879192</td>
      <td>15</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.868041</td>
      <td>0.876589</td>
      <td>16</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.868309</td>
      <td>0.874801</td>
      <td>17</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.867874</td>
      <td>0.872904</td>
      <td>18</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.866196</td>
      <td>0.870529</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
</div>




```python
# plot r2 score versus k
fig, axes = plt.subplots(figsize = (5,5))
axes.plot(knn_r2_df['k'], knn_r2_df['Train R^2'], 's-', label='Train $R^2$ Scores')
axes.plot(knn_r2_df['k'], knn_r2_df['Test R^2'], 's-', label='Test $R^2$ Scores')
axes.set_xlabel('k')
axes.set_ylabel('$R^2$ Scores')
# A generic title of this format (y vs x) is generally appropriate
axes.set_title("$R^2$ Scores vs k")
# Including a legend is very important
axes.legend();
```



![png](Final_Project_TestFile_files/Final_Project_TestFile_143_0.png)


It looks like the model reached best r2 values at k=6.



```python
# the following code was adapted from HW7 solutions
def plot_cv(ax, hyperparameter, cv_scores):
    cv_means = np.mean(cv_scores, axis=1)
    cv_stds = np.std(cv_scores, axis=1)
    handle, = ax.plot(hyperparameter, cv_means, '-*', label="Validation (mean)")
    plt.fill_between(hyperparameter, cv_means - 2.*cv_stds, cv_means + 2.*cv_stds, alpha=.3, color=handle.get_color())
```




```python
# the following code was adapted from HW7 solutions
# find the best model
fig, ax = plt.subplots()
ax.plot(k_values, train_scores, '-+', label="Training")
plot_cv(ax, k_values, cv_scores)
plt.xlabel("n_neighbors")
plt.ylabel("Mean CV accuracy");
plt.legend()

best_k = k_values[np.argmax(np.mean(cv_scores, axis=1))]
print("Best k:", best_k)
```


    Best k: 7



![png](Final_Project_TestFile_files/Final_Project_TestFile_146_1.png)




```python
# save model to the list
best_k = 7
models_list["knn_7"] = KNNModels[best_k].fit(xtrain, ytrain)
```




```python
# evaluate classification accuracy
best_model_KNN_train_score = accuracy_score(ytrain, knn_predicted_train[best_k].round())
best_model_KNN_test_score = accuracy_score(ytest, knn_predicted_test[best_k].round())
print("Training accuracy is {:.4}%".format(best_model_KNN_train_score*100))
print("Test accuracy is {:.4} %".format(best_model_KNN_test_score*100))
```


    Training accuracy is 96.34%
    Test accuracy is 95.92 %


[Back to TOC](#TOC) <br/>
<a id ='KNN'></a>
#### 4.5 - Decision tree

<mark> Nisrine, team please check </mark>



```python
depth_list =list(range(1, 18))

cv_means = []
cv_stds = []
train_scores = []
best_model_mean = 0

for depth in depth_list:
    #Fit a decision tree to the training set
    model_DTC = DecisionTreeClassifier(max_depth=depth).fit(xtrain, ytrain)
    scores = cross_val_score(model_DTC, xtrain, ytrain, cv=5)
    
    #training set performance
    train_scores.append(model_DTC.score(xtrain, ytrain))
    
    #save best model
    if scores.mean() > best_model_mean:
            best_model_mean=scores.mean()
            best_model_DTC=model_DTC
            best_model_std =scores.std()
            
    
    #performance for 5-fold cross validation
    cv_means.append(scores.mean())
    cv_stds.append(scores.std())
    

cv_means = np.array(cv_means)
cv_stds = np.array(cv_stds)
train_scores = np.array(train_scores)
```




```python
plt.subplots(1, 1, figsize=(12,7))
plt.plot(depth_list, cv_means, '*-', label="Mean CV")
plt.fill_between(depth_list, cv_means - 2*cv_stds, cv_means + 2*cv_stds, alpha=0.3)
ylim = plt.ylim()
plt.plot(depth_list, train_scores, '<-', label="Train Accuracy")
plt.legend()
plt.ylabel("Score", fontsize=16)
plt.xlabel("Max Depth", fontsize=16)
plt.title("Scores for Decision Tree for \nDifferent depth value", fontsize=16)
plt.xticks(depth_list);
```



![png](Final_Project_TestFile_files/Final_Project_TestFile_151_0.png)




```python
models_list["random_forest"] = rf_model

best_model_DTC_train_score = accuracy_score(ytrain, best_model_DTC.predict(xtrain))
best_model_DTC_test_score = accuracy_score(ytest, best_model_DTC.predict(xtest))
print("Training accuracy is %.4f"%best_model_DTC_train_score)
print("Test accuracy is %.4f"%best_model_DTC_test_score)
```



    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-357-40e4de29d00d> in <module>()
    ----> 1 models_list["random_forest"] = rf_model
          2 
          3 best_model_DTC_train_score = accuracy_score(ytrain, best_model_DTC.predict(xtrain))
          4 best_model_DTC_test_score = accuracy_score(ytest, best_model_DTC.predict(xtest))
          5 print("Training accuracy is %.4f"%best_model_DTC_train_score)


    NameError: name 'rf_model' is not defined


[Back to TOC](#TOC) <br/>
<a id ='Random-Forest'></a>
#### 4.5 -Random Forest

<mark> Nisrine </mark>



```python
rf = RandomForestClassifier(max_depth=6)
rf_model = rf.fit(xtrain, ytrain)
score = rf_model.score(xtest, ytest)

models_list["random_forest"] = rf_model

print("Random Forest model score is ", score)
```


[Back to TOC](#TOC) <br/>
<a id ='Boosting - AdaBoost Classifier'></a>
#### 4.6 -Boosting - AdaBoost Classifier

<mark> Nisrine, team please check, we can iterate for different depths </mark>



```python
AdaBoost_models = {}
AdaBoost_scores_train = {}
AdaBoost_scores_test = {}
for e in range(1, 5):
    AdaBoost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=e), n_estimators=800, learning_rate=0.05)
    AdaBoost_models[e] = AdaBoost.fit(xtrain, ytrain)
    AdaBoost_scores_train[e] = list(AdaBoost_models[e].staged_score(xtrain, ytrain))
    AdaBoost_scores_test[e] = list(AdaBoost_models[e].staged_score(xtest, ytest))
```




```python
fig, ax = plt.subplots(4,1, figsize=(20,35))
for e in range(0, 4):
    ax[e].plot(AdaBoost_scores_train[e+1], label='Train')
    ax[e].plot(AdaBoost_scores_test[e+1], label='Test')
    ax[e].set_xlabel('Number of Iterations', fontsize=16)
    ax[e].set_ylabel('Accuracy', fontsize=16)
    ax[e].tick_params(labelsize=16)
    ax[e].legend( fontsize=16)
    ax[e].set_title('Depth = %s'%(e+1), fontsize=18)
fig.suptitle('Accuracy by number of Iterations\n for various Depth',y=0.92,fontsize=20);
```


[Back to TOC](#TOC) <br/>
<a id ='SVM'></a>
#### 4.6 -SVM

<mark> TO DO: Fayzan  - if time permits </mark>

[Back to TOC](#TOC) <br/>
<a id ='RNN'></a>
#### 4.7 -RNN

<mark> Are we doing RNN? </mark>

[Back to TOC](#TOC) <br/>
<a id ='KMeans-Clustering'></a>
#### 4.8 - K-Means Clustering

We want to explore unsupervised learning by performing k=2 KMeans clustering with numerical features, and see how the clusters align with our manually verified results.

<mark> TO DO: tune k </mark>



```python
# read the data
# tweets_df = pd.read_json('50_accounts_200_tweets_each_final_std.json')
```




```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, init='random', random_state=0).fit(xtrain.values)
```




```python
# add the classification result
k2 = tweets_df[col_pred_numerical]

k2['k=2'] = kmeans.labels_
```




```python
# create df for easy plot
kmean_0 = k2.loc[k2['k=2']==0]
kmean_1 = k2.loc[k2['k=2']==1]
class_0 = tweets_df.loc[tweets_df['class_verified']==0]
class_1 = tweets_df.loc[tweets_df['class_verified']==1]
```




```python
# see how many were classified as bots
print ('The size of the two clusters from kmeans clustering are {} and {}.'.format(len(kmean_0), len(kmean_1)))
```


    The size of the two clusters from kmeans clustering are 4207 and 2985 and 300.




```python
# quick plot to see if it naturally come into two clusters
plt.scatter(kmean_0['account_age'], kmean_0['tweet_time_std'], c='salmon', s=20, label = 'cluster 0', alpha=0.5)
plt.scatter(kmean_1['account_age'], kmean_1['tweet_time_std'], c='royalblue', s=20, label = 'cluster 1', alpha=0.5)
#plt.scatter(class_0['account_age'], class_0['tweet_time_std'], c='red', s=2, label = 'cluster 0', alpha=0.2)
plt.scatter(class_1['account_age'], class_1['tweet_time_std'], c='yellow', s=2, label = 'bots', alpha=1)
plt.xlabel('account_age')
plt.ylabel('tweet_time_std')
plt.title('KMeans Clustering with K=2');
plt.legend();
```



![png](Final_Project_TestFile_files/Final_Project_TestFile_166_0.png)




```python
# 
verified_df = tweets_df['class_verified', 'id', ''].dropna()
verified_df = 
```


[Back to TOC](#TOC) <br/>
<a id ='KMeans-Clustering'></a>
#### 4.9 - Manually Verify Accounts and Compare Results
<mark> TO DO: Huan <mark>
  
  When comparing botometer scores and manually classified results, we noticed that botometer does not always predict actual bot / non-bot correctly. 



```python
# TO BE UPDATED!!!!
# read the verified dataframe
raw_df_verify = pd.read_csv('boto_verify.csv')
raw_df_verify.dropna()
raw_df_verify.head()
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>screen_name</th>
      <th>class_boto</th>
      <th>boto_univ</th>
      <th>class_verified</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>ICVeo</td>
      <td>1</td>
      <td>0.661264</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>SamGuile2</td>
      <td>0</td>
      <td>0.199883</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>BadassJack101</td>
      <td>0</td>
      <td>0.002308</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100</td>
      <td>FAO_Scotbot</td>
      <td>1</td>
      <td>0.559158</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1000</td>
      <td>Habereglencee</td>
      <td>0</td>
      <td>0.199883</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>





```python
# join features and botometer result to the verified dataframe

```




```python
# discussion and compare results
```


[Back to TOC](#TOC) <br/>
<a id ='KMeans-Clustering'></a>
#### 4.10 - Validate Botometer Results and Model Prediction Results
<mark> TO DO: Huan - need more discussion <mark>
 
We try to use a random forest to explore the subspace between botometer results and the actual result (manually verified classification). We chose to use non-linear model as we expect the relationship between botometer result and actual result to be non-linear.   <br/>
  
We want to train a model with two features plus botometer score as predictors, and the actual classification as the response. In the principle that the botometer is occasionally accurate, and we want to see under what occasions they are accurate / inaccurate, and therefore to capture the residuals between our predictions (which use botometer score as predictors) and the actual results. (* we chose to features as we want to minimize number of features, given our sample size - manually verified bot account - is only 50)

  



```python
# use features and botometer score to predict the validated score

def prepare_vf(df, feature1, feature2):
  y_vf = df[['class_verified']]
  #y_vf = y_vf.dropna()
  x_vf = pd.DataFrame(df[[feature1, feature2, 'class_boto']])
  #x_vf = pd.merge(x_vf, train_tweets_df[[feature1, feature2, 'class_boto', 'id']], left_on='id', right_on='id') # TO DO: debug this line, something is wrong
  return x_vf, y_vf

x_train_vf, y_train_vf = prepare_vf(train_tweets_df, 'account_age', 'tweet_time_mean')
x_test_vf, y_test_vf = prepare_vf(test_tweets_df, 'account_age', 'tweet_time_mean')


rf_vf = RandomForestClassifier(max_depth=6)
rf_vf_model = rf_vf.fit(x_train_vf, y_train_vf)
score = rf_vf_model.score(x_test_vf, y_test_vf)

print("Random Forest model, (class_verified ~ account_age, tweet_time_mean, class_boto), the testscore is ", score)
```


    Random Forest model, (class_verified ~ account_age, tweet_time_mean, class_boto), the testscore is  0.9791833466773419




```python
# discussion: when we can trust the model, when it is unknown
```


[Back to TOC](#TOC) <br/>
<a id ='KMeans-Clustering'></a>
#### 4.11 - Model Comparisons

<mark> Compare all models, compare results to Botometer results </mark>



```python
### Summary Report

models_list
```




```python
# copy and pasted previous code - need update
# graph references

# the following code was adapted from HW5 solutions

# your code here
plt.figure(figsize=(12,8))
sort_index = np.argsort(x_test['D29963_at'].values)

# plotting true binary response
plt.scatter(x_test['D29963_at'].iloc[sort_index], y_test.iloc[sort_index], color='black', label = 'True Response')

# plotting ols output
plt.plot(x_test['D29963_at'].iloc[sort_index], y_test_pred.iloc[sort_index], color='red', alpha=0.3, \
         label = 'Linear Regression Predictions')
# plt.plot(x_test['D29963_at'].iloc[sort_index], y_test_pred.iloc[sort_index]>0.5, color='red', ls='-.', \
#          label = 'Linear Regression Class Predictions ')


# plotting logreg prob output
plt.plot(x_test['D29963_at'].iloc[sort_index], y_test_pred_logreg_prob[sort_index], alpha=0.3,  \
         color='green', label = 'Logistic Regression Predictions Prob')
#plt.plot(x_test['D29963_at'].iloc[sort_index], y_test_pred_logreg[sort_index], color='green', ls='-.' ,label = 'Logistic Regression Predictions')

plt.axhline(0.5, c='c')
plt.legend()
plt.title('True response v/s obtained responses')
plt.xlabel('Gene predictor value')
plt.ylabel('Cancer type response');
```


[Back to TOC](#TOC) <br/>
<a id ='Results-and-Conclusion'></a>
### 5 - Results and Conclusion

<mark> Some Text Here </mark>

General findings (just throwing thought here, to be organized):

- ideally, we would want a large number of manually verified accounts. In real world application, we would use mechanical turk to manually identify at least 1,000 accounts and use that as our data. <br/>
- except for explicit bots / non-bot account, there are accounts that are very difficult to tell bot/non-bot even manually. One potential solution for that is to loo

[Back to TOC](#TOC) <br/>
<a id ='Summary-of-Results'></a>
#### 5.1 -Summary of Results

<mark> Some Text and Code Here </mark>

[Back to TOC](#TOC) <br/>
<a id ='Noteworthy-Findings'></a>
#### 5.2 -Noteworthy Findings

<mark> Some Text and Code Here </mark>

[Back to TOC](#TOC) <br/>
<a id ='Conclusion-and-Future-Work'></a>
#### 5.3 -Conclusion and Future Work

<mark> Some Text and Code Here </mark>

[Back to TOC](#TOC) <br/>
<a id ='Literature-Review-and-Related-Work'></a>
### 6 - Literature Review and Related Work

<mark> Some Text Here </mark>

[Back to TOC](#TOC) <br/>
<a id ='Bots-in-the-Twittersphere'></a>
#### 6.1 -Bots in the Twittersphere

<mark> Some Text and Code Here </mark>

[Back to TOC](#TOC) <br/>
<a id ='How-Twitter-Bots-Help-Fuel-Political-Feuds'></a>
#### 6.2 -How Twitter Bots Help Fuel Political Feuds

<mark> Some Text and Code Here </mark>

[Back to TOC](#TOC) <br/>
<a id ='The-spread-of-low-credibility-content-by-social-bots'></a>
#### 6.3 -The spread of low-credibility content by social bots

<mark> Some Text and Code Here </mark>

[Back to TOC](#TOC) <br/>
<a id ='Twitter-Topic-Modeling-by-Tweet-Aggregation'></a>
#### 6.4 -Twitter Topic Modeling by Tweet Aggregation

<mark> Some Text and Code Here </mark>

[Back to TOC](#TOC) <br/>
<a id ='tweepy-library'></a>
#### 6.5 -The tweepy Python library
http://www.tweepy.org <br/>

<mark> Some Text and Code Here </mark>

[Back to TOC](#TOC) <br/>
<a id ='twitter-developer'></a>
#### 6.6 -Twitter's Developer resources
https://developer.twitter.com <br/>

<mark> Some Text and Code Here </mark>
