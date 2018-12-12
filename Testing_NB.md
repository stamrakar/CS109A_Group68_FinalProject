---
nav_include: 2
title: Test Notebook
notebook: Testing_NB.ipynb
---

#### test notebook 
## Machine Learning Analysis for Twitter Bot Detection
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
    3.1 [Data Cleansing](#Data-Cleansing) <br/>
    3.2 [Data Wrangling](#Data-Wrangling) <br/>
    3.3 [Important Features](#Important-Features) <br/>
    3.4 [Relations in Data](#Relations-in-Data) <br/>
    3.5 [Feature Engineering](#Feature-Engineering) <br/>
    3.6 [Standardization](#Standardization) <br/>   
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


___

[Back to TOC](#TOC) <br/>
<a id ='Introduction'></a>
### 1- Introduction

The main objective of the project is explore twitter dataset using twitter API and try to create a learning algorithim that can differentiate between bot and human Twitter account.

[Back to TOC](#TOC) <br/>
<a id ='Motivation'></a>
#### 1.1 - Motivation <br/>

<mark> Some Text Here </mark>

[Back to TOC](#TOC) <br/>
<a id ='Problem-Statement'></a>
#### 1.2 - Problem Statement <br/>
How to detect Twitter Bots using tweets data from Twitter developer API by using machine learning techniques. Our objective is to determine whether the source of tweets are from accounts that are bot users [1] or non-bot users [0].  (we define bot as: no direct human involvement in generating tweets) <br/>
1. Start by collection data using Twitter API
2. Perform feature engineering and preprocessing techniques to aggregate tweet features to account level features
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


    C:\Users\a\Anaconda3\lib\site-packages\seaborn\apionly.py:6: UserWarning: As seaborn no longer sets a default style on import, the seaborn.apionly module is deprecated. It will be removed in a future version.
      warnings.warn(msg, UserWarning)


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

##### Step 1: Collection of Data : Collect over 6,000 tweets
##### Step 2: Data Labelling: Use Botometer 
##### Step 3: Data Labelling: Chose to Manual Verify 40 accouts (20 bots, 20 actual users)
##### Step 4: Data Collection - Get 200 (max) Most Recent Tweets from Each 6,000+ Tweets Account

[Back to TOC](#TOC) <br/>
<a id ='Data-in-Twitter-API'></a>
#### 2.1 - Data Source: Twitter API

<mark> OPTIONAL - Add some text - General Discussion of Twitter API and Tweepy </mark>

[Back to TOC](#TOC) <br/>
<a id ='Collection-of-Data'></a>
#### 2.2 - Collection of Data : Collect over 6,000 Tweets using Keywords

We first collected some tweets that contains one of the following keywords that are likely to lead to controversial topics:  <br/>
>  1) Immigration <br/>
>  2) Brexit <br/>
>  3) bitcoin <br/>

We used keywords of more controversial topics as those are more likely to have non-obvious bots.

We requested every 2 seconds for 100 tweets each for 120 request and received 1277 tweets.



```python
# http://www.tweepy.org/
import tweepy

# Replace the API_KEY and API_SECRET with your application's key and secret.
auth = tweepy.AppAuthHandler("apikey", "api secret")

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
# collect samples (which we will use botometer to encode)
collect_tweets(7000, 70, 'immigration_brexit_bitcoin_extended.json')
```




```python
# load the file
raw_df = pd.read_json('immigration_brexit_bitcoin_extended.json', lines=True)
```




```python
# take a look at the separate data
display(raw_df.shape)
```




```python
# take a look at the combined data
display(raw_df.columns.values()
display(raw_df.shape)
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

We label each account with botometer score via their id.



```python
#load the data
raw_df = pd.read_json('immigration_brexit_bitcoin_full.json')
raw_df.shape
```




```python
# add account id to dataframe
raw_df['id'] = raw_df['user'].map(lambda d: d['id'])
```




```python
# set up botometer
# the code below was adapted from 
# https://github.com/IUNetSci/botometer-python

import botometer

mashape_key = "MASHAPE KEY"
twitter_app_auth = {
    'consumer_key': 'consumer key',
    'consumer_secret': 'consumer secret',
    'access_token': 'api key',
    'access_token_secret': 'api secret',
  }

bom = botometer.Botometer(wait_on_ratelimit=True,
                          mashape_key=mashape_key,
                          **twitter_app_auth)
```




```python
# retrieve response objects from Botometer
botometer_results = {}
count = 0
for index, user_id in raw_df['id'][.iteritems():
    try:
        botometer_results[index] = bom.check_account(user_id)
        print(count)
        count +=1
    except tweepy.TweepError as err:
        print("Skipping user {} due to error {}".format(user_id, err))
    except NoTimeLineError as err:
        print("Skipping user {} due to error {}".format(user_id, err))
    time.sleep(2)
```




```python
raw_df['botometer_result'].dropna().shape
```





    (6032,)





```python
# convert to series
botometer_series = pd.Series(botometer_results)
```




```python
# add results to a new column
raw_df['botometer_result'] = botometer_series
```




```python
# extract universal score (botometer score)
raw_df['boto_univ'] = raw_df['botometer_result'].map(lambda s: s['cap']['universal'])
raw_df['boto_univ'].describe()
```





    count    6032.000000
    mean        0.070146
    std         0.160049
    min         0.001643
    25%         0.004304
    50%         0.009037
    75%         0.038677
    max         0.967026
    Name: boto_univ, dtype: float64





```python
# encode bot / non-bot via score of 0.2 threshold
# we chose 0.2 threshold instead of 0.5 as we quickly verify the botometer results, and found many of the accounts with less than 0.5 are still bots
threshold = 0.2
raw_df['class_boto'] = np.where(raw_df['boto_univ']>threshold, 1, 0)
```




```python
# examine number of 'bots' as identified by Botometer
sum(raw_df['class_boto'])
```





    593





```python
# save as csv
raw_df.to_csv('immigration_brexit_bitcoin_full_boto.csv')

# save as json
raw_df.to_json('immigration_brexit_bitcoin_full_boto.json')
```


[Back to TOC](#TOC) <br/>
<a id ='Labelling-Manual'></a>
#### 2.4 - Data Labelling: Manual Verification for Each Account (Until Reach 20 Bots 20 Real Users)

We verified accounts by manually search the username to check if they are bots or not using our best judgement. Usually, a verified Twitter accounts are often real users or obvious bots (e.g. Netflix). <br/>

Meanwhile, we only want to look at English tweets. <br/>

The following rules are used for manual Twitter account verification: <br/>
1) Constant retweets of media (especially only retweets no replies)  <br/>
2) Strong concentration on a specific topic <br/>
3) Significantly large number of tweets <br/>
4) Significantly large number of replying - not humanly possible speed <br/>
<mark> (add more verification rules) <mark>
  
During identification......<mark> add text <mark> <br/>
 
We keep manually identifying bots / non-bots account, only record the ones we are certain about. We keep identifying until reached 20 bots and 20 non-bots.



```python
# load the data
raw_df = pd.read_json('immigration_brexit_bitcoin_full_boto.json')
raw_df.shape
```





    (6032, 34)





```python
# to verify each user, we only need "screen_name"
raw_df['screen_name'] = raw_df['user'].map(lambda d: d['screen_name'])
```




```python
# form a simple dataframe with only screen_name and Botometer score for references (so we can manually verify accounts)
# create 'class_verified for verified score'
raw_df_verify = raw_df.loc[:,['screen_name', 'class_verified']]
```




```python
# save as csv (so we can manually verify and input results in excel)
raw_df_verify.to_csv('to_verify.csv')
```




```python
# we manually verified 40 accounts by searching screen_name, view the user's previous tweets, profiles, etc.
# we recorded in the cvs as 1(bot) and 0(non-bot), and only recorded the accounts that we feel certain about
# we kept searching until reach 20 bots and 20 users
verify_df =pd. read_csv('boto_verify.csv')
```




```python
users_list = verify_df.loc[verify_df['class_verified']==0]
bots_list = verify_df.loc[verify_df['class_verified']==1]
```




```python
display(users_list.shape)
display(bots_list.shape)
```


[Back to TOC](#TOC) <br/>
<a id ='Data-Collection-Recent'></a>
#### 2.5 - Data Collection - Get 200 (max) Most Recent Tweets from Verified Bot / User

For each of the 6032 accounts we identified, we requested users' most recent 200 tweets using api.user_timeline via tweepy. <br/>



```python
# read the verified dataframe
raw_df = pd.read_json('immigration_brexit_bitcoin_full_boto.json')
raw_df.shape
```





    (6032, 34)





```python
#names = raw_df['screen_name'].tolist()
names = raw_df[raw_df['botometer_result'].notnull()]['user'].map(lambda u: u['screen_name']).tolist()
```




```python
len(names)
```





    6032





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
    error_count = 0
    
    with open(fName, 'w') as f:
        tweetCount = 0
        for name in names:
            try:
                tweets = api.user_timeline(screen_name=name, count=t_count, tweet_mode='extended')
                for tweet in tweets:
                    f.write(jsonpickle.encode(tweet._json, unpicklable=False) + '\n')
                print("Downloaded {} tweets.".format(len(tweets)))
                tweetCount += len(tweets)
                time.sleep(2)
            except Exception as e:
                # Just exit if any error
                error_count += 1
                print("some error : " + str(e))
                if error_count >= 100:
                    print("too many errors ....break.")
                    break
    print ("Downloaded {0} tweets, Saved to {1}".format(tweetCount, fName))
```




```python
# get max 200 tweets for each user
get_tweets(names=names, fName='tweets.json', t_count=200, verify_df=raw_df) #the fName and corresponding data will be updated later
```




```python
# read the data
tweets_df = pd.read_json('tweets.json', lines=True)
```




```python
tweets_df.columns.values
```





    array(['contributors', 'coordinates', 'created_at', 'display_text_range',
           'entities', 'extended_entities', 'favorite_count', 'favorited',
           'full_text', 'geo', 'id', 'id_str', 'in_reply_to_screen_name',
           'in_reply_to_status_id', 'in_reply_to_status_id_str',
           'in_reply_to_user_id', 'in_reply_to_user_id_str',
           'is_quote_status', 'lang', 'place', 'possibly_sensitive',
           'quoted_status', 'quoted_status_id', 'quoted_status_id_str',
           'quoted_status_permalink', 'retweet_count', 'retweeted',
           'retweeted_status', 'source', 'truncated', 'user',
           'withheld_copyright', 'withheld_in_countries', 'withheld_scope'],
          dtype=object)



[Back to TOC](#TOC) <br/>
<a id ='Description-of-Raw-Data'></a>
#### 2.6 - Description of Raw Data (Tweets)

<mark>(TO BE UPDATED)<mark> <br/>
  
Among the data we collected as json files from the tweepy api.search, the data set contains objects such as ‘user’, which includes comprehensive information of user accounts. Additionally, detailed information about each individual tweet was also collected.

The following describes some of the fields of the raw data collected:

> *followers* : number of user’s followers  <br/>
> *friends* : information about relationship/interaction with other users <br/>
> *following* : users following the specified user/account <br/>
> *retweet count* : number of retweets <br/>
> *verify_credentials* : verifies whether the user credentials are valid <br/>
> *screen_name* : screen name of account user<br/>
> *retweets* : number of retweets of a given tweet<br/>
> *user_description* : description set by user in profile<br/>
> *profile_background_url* : user’s background image for profile <br/>
> *profile_image_url* :  user’s profile image <br/>
> *geo_enabled* : location of tweet if user source has geo-location enabled<br/>

Botometer’s response object returned bot-scores in various different categories. This included categories such as the Complete Automation Probability, which determines how likely the account is a bot. The bot-scores, on a scale, determines if a given account is closer to a bot or a real user. Then. from the json data we gathered through the tweepy api.search, we extracted user account id to retrieve their corresponding Botometer scores. 


[Back to TOC](#TOC) <br/>
<a id ='Exploratory-Data-Analysis'></a>
### 3 - Exploratory Data Analysis
In this section, we want to clean the data, explore patterns, aggregate tweet level features to account level, and standardize our data as needed to prepare for the next step of modelling.

[Back to TOC](#TOC) <br/>
<a id ='Data-Wrangling'></a>
#### 3.1 - Data Wrangling & Cleansing

First, we want to parse features, Include only features with value and drop features with mostly null value.



```python
# read the dataset
tweets_df = pd.read_json('tweets.json', lines=True)
```




```python
# first we want to reduce columns by dropping the features that miss data more than 50% of the time
threshold = len(tweets_df.columns.values)*0.5
tweets_df = tweets_df.dropna(thresh = threshold, axis='columns')
```




```python
# take a look at the shape
tweets_df.shape
```





    (1172951, 31)





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
```




```python
# examine shape
display(tweets_df.shape)
```




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



![png](Testing_NB_files/Testing_NB_58_0.png)




```python
# drop empty columns again (after exploding 'user' and 'entities')
threshold = len(tweets_df.columns.values)*0.5
tweets_df = tweets_df.dropna(thresh = threshold, axis='columns')
```




```python
# take a look at the columns left
display(len(tweets_df.columns.values))
display(tweets_df.columns.values)
```



    65



    array(['coordinates', 'created_at', 'display_text_range', 'entities',
           'extended_entities', 'favorite_count', 'favorited', 'full_text',
           'geo', 'id', 'id_str', 'in_reply_to_screen_name',
           'in_reply_to_status_id', 'in_reply_to_status_id_str',
           'in_reply_to_user_id', 'in_reply_to_user_id_str',
           'is_quote_status', 'lang', 'place', 'possibly_sensitive',
           'quoted_status', 'quoted_status_id', 'quoted_status_id_str',
           'quoted_status_permalink', 'retweet_count', 'retweeted',
           'retweeted_status', 'source', 'truncated', 'user',
           'withheld_in_countries', 'user_created_at', 'user_default_profile',
           'user_default_profile_image', 'user_description', 'user_entities',
           'user_favourites_count', 'user_followers_count',
           'user_friends_count', 'user_geo_enabled',
           'user_has_extended_profile', 'user_id', 'user_id_str',
           'user_is_translation_enabled', 'user_lang', 'user_listed_count',
           'user_location', 'user_name', 'user_profile_background_color',
           'user_profile_background_image_url',
           'user_profile_background_image_url_https',
           'user_profile_background_tile', 'user_profile_banner_url',
           'user_profile_image_url', 'user_profile_image_url_https',
           'user_profile_link_color', 'user_profile_sidebar_border_color',
           'user_profile_sidebar_fill_color', 'user_profile_text_color',
           'user_profile_use_background_image', 'user_screen_name',
           'user_statuses_count', 'user_translator_type', 'user_url',
           'user_verified'], dtype=object)




```python
# we only interested in english tweets
tweets_df_en = tweets_df.loc[tweets_df['lang']=='en']
tweets_df_en.shape
```





    (1042177, 65)





```python
# duplicated / no longer userful columns
col_duplicate = ['entities','user', 'lang', 'user_lang', 'user_id', 'user_id_str', 'id_str']
# we dropped 'lang' as we only use english accounts for our dataset
# 'entities' and 'user' have already been parsed

# columns that we are obviously not interested
col_not_interested = ['user_entities']
# retweeted_status is the tweet object of the retweet - perhaps 
```




```python
# drop duplicated columns and columns that we are not interested
tweets_df_en = tweets_df_en.drop(columns= (col_duplicate + col_not_interested))
```




```python
# take a look at shape
tweets_df_en.shape
```





    (1042177, 57)





```python
# save as json
tweets_df_en.to_json('tweets_clean.json')
```


[Back to TOC](#TOC) <br/>
<a id ='Data-Cleansing'></a>
#### 3.2 - Feature Engineering
Next, we want to aggregate tweet features to the accounts.



```python
# read previous json file
tweets_df = pd.read_json('tweets_clean.json')
```


[Back to TOC](#TOC) <br/>
<a id ='Data-Cleansing'></a>
##### 3.2.1 - Feature Engineering - Tweet Features
feature engineering of tweet features: <br/>
1) text_rt: text of the retweet <br/>
2) text_tweet: text of the tweet (when there is no retweet) <br/>
3) encode tweet features <br/>



```python
# although using tweet_mode='extended', we are still not getting the full text
# therefore, we tried to get full_text from retweeted_status
tweets_df['text_rt'] = tweets_df['retweeted_status'].map(lambda x: x['full_text'] if x and (not isinstance(x, float)) and ('full_text' in x) else None)
tweets_df['text_tweet'] = tweets_df['full_text'].where(tweets_df['text_rt'].map(lambda x: x is None), None)
tweets_df[['text_tweet', 'text_rt']].head(5)
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
      <td>NEW Coinpot Multiplier : How to Win more ( Bes...</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>How To Buy Things With Bitcoin Coinpot Litecoi...</td>
      <td>None</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Filling the Brita all the way to the top count...</td>
      <td>None</td>
    </tr>
    <tr>
      <th>100</th>
      <td>You can collect other bitcoin faucets and incr...</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1000</th>
      <td>None</td>
      <td>Michael Gove on 30 November 2018 and the truth...</td>
    </tr>
  </tbody>
</table>
</div>





```python
# take a look at retweets
tweets_df[['text_tweet', 'text_rt']][tweets_df['text_rt'].map(lambda s: s is not None)].head()
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
      <th>1000</th>
      <td>None</td>
      <td>Michael Gove on 30 November 2018 and the truth...</td>
    </tr>
    <tr>
      <th>10000</th>
      <td>None</td>
      <td>This is me being held by my Dad in 1955. I hav...</td>
    </tr>
    <tr>
      <th>100002</th>
      <td>None</td>
      <td>Nicola Sturgeon uses the threat of a referendu...</td>
    </tr>
    <tr>
      <th>100003</th>
      <td>None</td>
      <td>This horse was spotted walking around a Tesco ...</td>
    </tr>
    <tr>
      <th>100004</th>
      <td>None</td>
      <td>When did worrying about immigration's impact o...</td>
    </tr>
  </tbody>
</table>
</div>





```python
# encode tweet features

# 1 = has extende entities; 0 = don't have extende entities
#tweets_df['extended_entities'] = tweets_df['extended_entities'].map(lambda x: 0 if x==None else 1) 

# 1 = favorited - True; 0 = favorited - False
tweets_df['favorited'] = tweets_df['favorited'].map(lambda x: 0 if x==False else 1) 

# 1 = is_quote_status - True - True; 0 = is_quote_status - False
#tweets_df['is_quote_status'] = tweets_df['is_quote_status'].map(lambda x: 0 if x==False else 1) 

# -1 = None; else - actual value
#tweets_df['possibly_sensitive'] = tweets_df['possibly_sensitive'].map(lambda x: x if x>=0 else -1) 

# 1 = reply to at least one user; 0 = not reply to another user
#tweets_df['in_reply_to_screen_name'] = tweets_df['in_reply_to_screen_name'].map(lambda x: 1 if x else 0)

# 1 = retweeted-true; 0 = retweeted-false
tweets_df['retweeted'] = tweets_df['retweeted'].map(lambda x: 1 if x==True else 0) 

# 1 = tweet includes retweet; 0 = tweet does not include retweet
tweets_df['retweeted_status'] = tweets_df['retweeted_status'].map(lambda x: 0 if x==None else 1)

# 0 = none or information not available
tweets_df['user_listed_count'] = tweets_df['user_listed_count'].map(lambda x: x if x>0 else 0)

# 1 = truncated-true; 0 = truncated-false
#tweets_df['truncated'] = tweets_df['truncated'].map(lambda x: 0 if x==False else 1) 

# replace nan with 0 for the following features (as for these features, missing values usually means 0)
for f in ['user_favourites_count', 'user_followers_count', 'user_friends_count']:
    tweets_df[f] = tweets_df[f].replace(np.nan, 0, regex=True)
```




```python
tweets_df.shape
```





    (1042177, 59)



[Back to TOC](#TOC) <br/>
<a id ='Data-Cleansing'></a>
##### 3.2.2 - Feature Engineering - User Features
feature engineering of user features: <br/>
1) length of user description <br/>
2) tweet frequencies (the mean, std, min, and max time between tweets for each account) <br/>
3) account age (seconds from the account creation time to the latest tweet time) <br/>



```python
# extract
tweets_df['screen_name'] = tweets_df['user_screen_name']
```




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
      <td>0604Arb1320</td>
      <td>1103.088481</td>
      <td>11493.389030</td>
      <td>0.0</td>
      <td>170593.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>07_smith</td>
      <td>98062.614973</td>
      <td>153488.369492</td>
      <td>6.0</td>
      <td>975586.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0AngelHeart</td>
      <td>1354.688172</td>
      <td>5038.485992</td>
      <td>6.0</td>
      <td>34269.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0ttaM</td>
      <td>14095.382653</td>
      <td>29149.325281</td>
      <td>3.0</td>
      <td>134152.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100Climbs</td>
      <td>2862.817204</td>
      <td>6481.823049</td>
      <td>6.0</td>
      <td>40810.0</td>
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
      <td>0604Arb1320</td>
      <td>190794240.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>07_smith</td>
      <td>222929527.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0AngelHeart</td>
      <td>238521195.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0ttaM</td>
      <td>142325785.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100Climbs</td>
      <td>279804439.0</td>
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
      <td>ICVeo</td>
      <td>1033.132275</td>
      <td>1.937518e+03</td>
      <td>0.0</td>
      <td>8375.0</td>
      <td>77</td>
      <td>226104810.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ianw2000uk</td>
      <td>254.486146</td>
      <td>1.616182e+03</td>
      <td>0.0</td>
      <td>26368.0</td>
      <td>106</td>
      <td>307106781.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>hmsglasgow</td>
      <td>107.966921</td>
      <td>1.068803e+03</td>
      <td>0.0</td>
      <td>16601.0</td>
      <td>125</td>
      <td>162646536.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MarkHW1</td>
      <td>607784.043011</td>
      <td>2.526945e+06</td>
      <td>3.0</td>
      <td>28007804.0</td>
      <td>0</td>
      <td>281164185.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>RabHutchison</td>
      <td>1122.005291</td>
      <td>3.159343e+03</td>
      <td>4.0</td>
      <td>21281.0</td>
      <td>119</td>
      <td>261633864.0</td>
    </tr>
  </tbody>
</table>
</div>





```python
# read the dataset with botometer score
boto_df = pd.read_json('immigration_brexit_bitcoin_full_boto.json')
boto_df['screen_name'] = boto_df['user'].map(lambda u: u['screen_name'])
```




```python
# add botometer back
boto_class_df = boto_df[['class_boto','screen_name']].drop_duplicates(subset='screen_name')
tweets_df = pd.merge(tweets_df, boto_class_df, left_on='screen_name', right_on='screen_name')
tweets_df.columns.values
```





    array(['coordinates', 'created_at', 'display_text_range',
           'extended_entities', 'favorite_count', 'favorited', 'full_text',
           'geo', 'id', 'in_reply_to_screen_name', 'in_reply_to_status_id',
           'in_reply_to_status_id_str', 'in_reply_to_user_id',
           'in_reply_to_user_id_str', 'is_quote_status', 'place',
           'possibly_sensitive', 'quoted_status', 'quoted_status_id',
           'quoted_status_id_str', 'quoted_status_permalink', 'retweet_count',
           'retweeted', 'retweeted_status', 'source', 'truncated',
           'withheld_in_countries', 'user_created_at', 'user_default_profile',
           'user_default_profile_image', 'user_description',
           'user_favourites_count', 'user_followers_count',
           'user_friends_count', 'user_geo_enabled',
           'user_has_extended_profile', 'user_is_translation_enabled',
           'user_listed_count', 'user_location', 'user_name',
           'user_profile_background_color',
           'user_profile_background_image_url',
           'user_profile_background_image_url_https',
           'user_profile_background_tile', 'user_profile_banner_url',
           'user_profile_image_url', 'user_profile_image_url_https',
           'user_profile_link_color', 'user_profile_sidebar_border_color',
           'user_profile_sidebar_fill_color', 'user_profile_text_color',
           'user_profile_use_background_image', 'user_screen_name',
           'user_statuses_count', 'user_translator_type', 'user_url',
           'user_verified', 'text_rt', 'text_tweet', 'screen_name',
           'class_boto'], dtype=object)





```python
# merge the account information back to the dataset
tweets_df = pd.merge(tweets_df, users_df, left_on='screen_name', right_on='screen_name')
tweets_df.columns.values
```





    array(['coordinates', 'created_at', 'display_text_range',
           'extended_entities', 'favorite_count', 'favorited', 'full_text',
           'geo', 'id', 'in_reply_to_screen_name', 'in_reply_to_status_id',
           'in_reply_to_status_id_str', 'in_reply_to_user_id',
           'in_reply_to_user_id_str', 'is_quote_status', 'place',
           'possibly_sensitive', 'quoted_status', 'quoted_status_id',
           'quoted_status_id_str', 'quoted_status_permalink', 'retweet_count',
           'retweeted', 'retweeted_status', 'source', 'truncated',
           'withheld_in_countries', 'user_created_at', 'user_default_profile',
           'user_default_profile_image', 'user_description',
           'user_favourites_count', 'user_followers_count',
           'user_friends_count', 'user_geo_enabled',
           'user_has_extended_profile', 'user_is_translation_enabled',
           'user_listed_count', 'user_location', 'user_name',
           'user_profile_background_color',
           'user_profile_background_image_url',
           'user_profile_background_image_url_https',
           'user_profile_background_tile', 'user_profile_banner_url',
           'user_profile_image_url', 'user_profile_image_url_https',
           'user_profile_link_color', 'user_profile_sidebar_border_color',
           'user_profile_sidebar_fill_color', 'user_profile_text_color',
           'user_profile_use_background_image', 'user_screen_name',
           'user_statuses_count', 'user_translator_type', 'user_url',
           'user_verified', 'text_rt', 'text_tweet', 'screen_name',
           'class_boto', 'tweet_time_mean', 'tweet_time_std',
           'tweet_time_min', 'tweet_time_max', 'user_description_len',
           'account_age'], dtype=object)



[Back to TOC](#TOC) <br/>
<a id ='Data-Cleansing'></a>
##### 3.2.3 - Feature Engineering - Finalize and Clean Up Data
drop the columns that are no longer interesting / useful



```python
# delete columns that no longer useful
col_del = ['display_text_range', 'in_reply_to_status_id_str', 'in_reply_to_user_id_str','in_reply_to_status_id', 
           'in_reply_to_user_id', 'is_quote_status', 'quoted_status', 'quoted_status_id', 'quoted_status_id_str',
          'quoted_status_permalink', 'user_url', 'user_translator_type', 'user_default_profile_image',
          'user_default_profile', 'user_geo_enabled', 'user_has_extended_profile', 'user_profile_background_tile',
          'user_profile_image_url', 'user_profile_image_url_https', 'full_text', 'created_at', 
          'user_created_at', 'user_profile_background_image_url', 'user_profile_background_image_url_https',
          'user_profile_banner_url', 'user_profile_link_color', 'user_profile_sidebar_border_color',
           'possibly_sensitive', 'user_profile_sidebar_fill_color', 'user_profile_text_color', 'user_screen_name',
          'user_profile_background_color', 'extended_entities', 'in_reply_to_screen_name', 'truncated', 'user_location',
          'user_name', 'source', 'geo', 'place', 'withheld_in_countries', 'coordinates', 'user_is_translation_enabled', 
           'user_profile_use_background_image']

tweets_df = tweets_df.drop(columns=col_del, axis=1)
```




```python
tweets_df.dtypes
```





    favorite_count             int64
    favorited                  int64
    id                         int64
    retweet_count              int64
    retweeted                  int64
    retweeted_status           int64
    user_description          object
    user_favourites_count    float64
    user_followers_count     float64
    user_friends_count       float64
    user_listed_count        float64
    user_statuses_count        int64
    user_verified            float64
    text_rt                   object
    text_tweet                object
    screen_name               object
    class_boto                 int64
    tweet_time_mean          float64
    tweet_time_std           float64
    tweet_time_min           float64
    tweet_time_max           float64
    user_description_len       int64
    account_age              float64
    dtype: object





```python
# check user_verified
display(tweets_df.shape)
display(tweets_df[tweets_df['user_verified'].isnull()].shape)
```



    (1042177, 23)



    (1030080, 23)




```python
# as it is mostly None, we decided to delete this column
del tweets_df['user_verified']
```




```python
display(tweets_df.columns.values)
display(tweets_df.shape)
```



    array(['favorite_count', 'favorited', 'id', 'retweet_count', 'retweeted',
           'retweeted_status', 'user_description', 'user_favourites_count',
           'user_followers_count', 'user_friends_count', 'user_listed_count',
           'user_statuses_count', 'text_rt', 'text_tweet', 'screen_name',
           'class_boto', 'tweet_time_mean', 'tweet_time_std',
           'tweet_time_min', 'tweet_time_max', 'user_description_len',
           'account_age'], dtype=object)



    (1042177, 22)




```python
tweets_df.describe()
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
      <th>favorited</th>
      <th>id</th>
      <th>retweet_count</th>
      <th>retweeted</th>
      <th>retweeted_status</th>
      <th>user_favourites_count</th>
      <th>user_followers_count</th>
      <th>user_friends_count</th>
      <th>user_listed_count</th>
      <th>user_statuses_count</th>
      <th>class_boto</th>
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
      <td>1.042177e+06</td>
      <td>1042177.0</td>
      <td>1.042177e+06</td>
      <td>1.042177e+06</td>
      <td>1042177.0</td>
      <td>1.042177e+06</td>
      <td>1.042177e+06</td>
      <td>1.042177e+06</td>
      <td>1.042177e+06</td>
      <td>1.042177e+06</td>
      <td>1.042177e+06</td>
      <td>1.042177e+06</td>
      <td>1.042151e+06</td>
      <td>1.042095e+06</td>
      <td>1.042151e+06</td>
      <td>1.042151e+06</td>
      <td>1.042177e+06</td>
      <td>1.042177e+06</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.369078e+00</td>
      <td>0.0</td>
      <td>1.065372e+18</td>
      <td>1.161262e+03</td>
      <td>0.0</td>
      <td>6.618329e-01</td>
      <td>2.911783e+04</td>
      <td>4.698448e+03</td>
      <td>2.436023e+03</td>
      <td>8.248872e+01</td>
      <td>5.079386e+04</td>
      <td>8.472553e-02</td>
      <td>2.421042e+04</td>
      <td>7.999967e+04</td>
      <td>3.607009e+02</td>
      <td>7.625696e+05</td>
      <td>8.635588e+01</td>
      <td>1.653274e+08</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.821426e+01</td>
      <td>0.0</td>
      <td>3.293977e+16</td>
      <td>1.102336e+04</td>
      <td>0.0</td>
      <td>4.730860e-01</td>
      <td>5.792263e+04</td>
      <td>5.551828e+04</td>
      <td>9.868033e+03</td>
      <td>5.149238e+02</td>
      <td>1.228428e+05</td>
      <td>2.784730e-01</td>
      <td>2.718796e+05</td>
      <td>6.574782e+05</td>
      <td>1.442215e+05</td>
      <td>6.166656e+06</td>
      <td>5.849945e+01</td>
      <td>1.003235e+08</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>1.240361e+09</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.654061e+00</td>
      <td>6.853259e-01</td>
      <td>0.000000e+00</td>
      <td>5.200000e+01</td>
      <td>0.000000e+00</td>
      <td>9.692800e+04</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>1.069875e+18</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>1.848000e+03</td>
      <td>2.200000e+02</td>
      <td>2.950000e+02</td>
      <td>2.000000e+00</td>
      <td>4.622000e+03</td>
      <td>0.000000e+00</td>
      <td>2.883926e+02</td>
      <td>1.956763e+03</td>
      <td>0.000000e+00</td>
      <td>2.945100e+04</td>
      <td>3.000000e+01</td>
      <td>7.026666e+07</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>1.071769e+18</td>
      <td>2.100000e+01</td>
      <td>0.0</td>
      <td>1.000000e+00</td>
      <td>9.991000e+03</td>
      <td>7.330000e+02</td>
      <td>8.980000e+02</td>
      <td>1.000000e+01</td>
      <td>1.721700e+04</td>
      <td>0.000000e+00</td>
      <td>1.406402e+03</td>
      <td>5.712884e+03</td>
      <td>0.000000e+00</td>
      <td>5.203200e+04</td>
      <td>9.400000e+01</td>
      <td>1.710798e+08</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>1.072138e+18</td>
      <td>3.770000e+02</td>
      <td>0.0</td>
      <td>1.000000e+00</td>
      <td>3.237500e+04</td>
      <td>2.391000e+03</td>
      <td>2.525000e+03</td>
      <td>4.800000e+01</td>
      <td>5.069800e+04</td>
      <td>0.000000e+00</td>
      <td>6.532113e+03</td>
      <td>1.650136e+04</td>
      <td>5.000000e+00</td>
      <td>1.326000e+05</td>
      <td>1.440000e+02</td>
      <td>2.520437e+08</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.890900e+04</td>
      <td>0.0</td>
      <td>1.072320e+18</td>
      <td>3.560981e+06</td>
      <td>0.0</td>
      <td>1.000000e+00</td>
      <td>8.722570e+05</td>
      <td>2.613669e+06</td>
      <td>5.109480e+05</td>
      <td>2.262400e+04</td>
      <td>9.436490e+06</td>
      <td>1.000000e+00</td>
      <td>1.027626e+08</td>
      <td>6.100153e+07</td>
      <td>1.027626e+08</td>
      <td>2.203665e+08</td>
      <td>1.740000e+02</td>
      <td>3.793244e+08</td>
    </tr>
  </tbody>
</table>
</div>





```python
# create list of columns names for different categories and see if we have missed anything
col_response = ['class_boto']
col_pred_text = list(tweets_df.select_dtypes(['object']).columns.values)
col_id = ['id']
col_pred_numerical = list(tweets_df.select_dtypes(['float64', 'int64']).drop(columns=['class_boto', 'id']).columns.values)
```




```python
# take a look at numerical features
display(col_pred_numerical)
```



    ['favorite_count',
     'favorited',
     'retweet_count',
     'retweeted',
     'retweeted_status',
     'user_favourites_count',
     'user_followers_count',
     'user_friends_count',
     'user_listed_count',
     'user_statuses_count',
     'tweet_time_mean',
     'tweet_time_std',
     'tweet_time_min',
     'tweet_time_max',
     'user_description_len',
     'account_age']




```python
# take a look at text features
display(col_pred_text)
```



    ['user_description', 'text_rt', 'text_tweet', 'screen_name']




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



    (1042177, 20)


    ['favorited', 'retweeted'] are deleted as they only have one values across all the rows.




```python
# before saving the file, we want to delete any rows with NaN values from the new columns
col_w_nan = tweets_df.columns[tweets_df.isna().any()].tolist()
col_w_nan
```





    ['user_description',
     'text_rt',
     'text_tweet',
     'tweet_time_mean',
     'tweet_time_std',
     'tweet_time_min',
     'tweet_time_max']





```python
# while it is okay to have NaN in texts, we want to delete the rows with NaN Values in the tweet_time related columns
tweets_df = tweets_df.dropna(axis=0, subset=['tweet_time_mean', 'tweet_time_std', 'tweet_time_min', 'tweet_time_max'])
display(tweets_df.shape)
display(tweets_df.isna().any())
```



    (1042095, 20)



    favorite_count           False
    id                       False
    retweet_count            False
    retweeted_status         False
    user_description          True
    user_favourites_count    False
    user_followers_count     False
    user_friends_count       False
    user_listed_count        False
    user_statuses_count      False
    text_rt                   True
    text_tweet                True
    screen_name              False
    class_boto               False
    tweet_time_mean          False
    tweet_time_std           False
    tweet_time_min           False
    tweet_time_max           False
    user_description_len     False
    account_age              False
    dtype: bool




```python
# great! let's save as json
users_df.to_json('users.json')
tweets_df.to_json('tweets_clean_final.json')
```


[Back to TOC](#TOC) <br/>
<a id ='Important-Features'></a>
#### 3.3 - Advanced Feature Engineering - NLP Features

After cleaning up the file and did some feature engineering, we tried to create some NLP features that might be interesting to our project.



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
display(tweets_df.shape)
display(tweets_df.columns.values)
```



    (1042095, 28)



    array(['favorite_count', 'id', 'retweet_count', 'retweeted_status',
           'user_description', 'user_favourites_count',
           'user_followers_count', 'user_friends_count', 'user_listed_count',
           'user_statuses_count', 'text_rt', 'text_tweet', 'screen_name',
           'class_boto', 'tweet_time_mean', 'tweet_time_std',
           'tweet_time_min', 'tweet_time_max', 'user_description_len',
           'account_age', 'tweet_len_mean', 'tweet_len_std',
           'tweet_word_mean', 'tweet_word_std', 'retweet_len_mean',
           'retweet_len_std', 'retweet_word_mean', 'retweet_word_std'],
          dtype=object)




```python
# merge text features with uers_df
users_df = pd.merge(users_df, text_df, left_on='screen_name', right_on='screen_name')
```




```python
# clean up users_df a bit and join boto scores
users_df = pd.merge(users_df, tweets_df[['class_boto', 'screen_name']], left_on='screen_name', right_on='screen_name')
users_df = users_df.drop_duplicates(subset='screen_name')
```




```python
# great! let's save as json-again
users_df_final.to_json('users_final.json')
tweets_df.to_json('tweets_clean_final2.json')
```


[Back to TOC](#TOC) <br/>
<a id ='Important-Features'></a>
#### 3.3 - Important Features

Before we conclude our data processing, we want to explore if there are any tweets features that we haven't captured but might be interesting for our analysis. <br/>

We also want to explore the relationship among account-level features we have selected / engineered, and see if any of them are particularly interesting in identifying bots / nonbots.



```python
# read the data
tweets_df = pd.read_json('tweets_clean_final2.json')
```




```python
# separte bots and non-bots tweets for easy plotting
tweets_0 = tweets_df.loc[tweets_df['class_boto']==0]
tweets_1 = tweets_df.loc[tweets_df['class_boto']==1]
```




```python
# read the user dataframe
users_df = pd.read_json('users_final.json')
```




```python
# separte bots and non-bots accounts for easy plotting
users_0 = users_df.loc[users_df['class_boto']==0]
users_1 = users_df.loc[users_df['class_boto']==1]
```




```python
# scatter plot2
def scatterplot (col_b1, col_b2, col_r1, col_r2, col1, col2):
    plt.scatter(col_b1, col_b2, s=5, color='salmon', label='bot', alpha=0.75)
    plt.scatter(col_r1, col_r2, s=5, color='royalblue', label='non-bot', alpha=0.75)
    plt.xlabel(str(col1))
    plt.ylabel(str(col2))
    #plt.xlim(xlimit)
    #plt.ylim(ylimit)
    plt.legend(loc='best', bbox_to_anchor=(0.85, 0., 0.5, 0.5))
    title = str(col1) + ' vs ' + str(col2)
    plt.title(title)
    plt.savefig(str(title)+'.png')
```




```python
# scatter plot2
def scatterplot2 (col_b1, col_b2, col_r1, col_r2, col1, col2):
    plt.scatter(col_b1, col_b2, s=3, color='salmon', label='bot', alpha=0.0025)
    plt.scatter(col_r1, col_r2, s=3, color='royalblue', label='non-bot', alpha=0.0025)
    plt.xlabel(str(col1))
    plt.ylabel(str(col2))
    #plt.xlim(xlimit)
    #plt.ylim(ylimit)
    plt.legend(loc='best', bbox_to_anchor=(0.85, 0., 0.5, 0.5))
    title = str(col1) + ' vs ' + str(col2)
    plt.title(title)
    plt.savefig(str(title)+'.png')
```




```python
# histogram
def hist_plot(col, xlabel, ylabel, title):
    #values = col.values[~np.isnan(col.values)]
    plt.hist(col)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.xlim(xlimit)
    plt.title(title)
    plt.savefig(str(title)+'.png')
    return None
```




```python
# quick plots
plt.figure(figsize=(6,4))
hist_plot(np.log(users_0['tweet_time_min'].values.clip(1, 1000)), 'log(tweets_min_time_interval)','count', 'min time interval among all tweets for each NON-BOT in seconds')
plt.figure(figsize=(6,4))
hist_plot(np.log(users_1['tweet_time_min'].values.clip(1, 1000)), 'log(tweets_minimum_time_interval)','count', 'min time interval among all tweets for each BOT in seconds')
```



![png](Testing_NB_files/Testing_NB_115_0.png)



![png](Testing_NB_files/Testing_NB_115_1.png)


It looks like both botometer-defined non-bot (botometer score < 0.2) and bots (botometer score >=0.2) are heavily screwed towards almost 0 seconds for minimum time interval between tweets of each users. Bots tend to have even more screwed minimum time interval towards 0. <br/>

The botometer-identified bots also have heavily screwed minimim tweet time interval but many of them have significantly larger minimum time interval. We think it is reasonable as the bots might be set up to only tweet at a certain interval.



```python
# quick plots
plt.figure(figsize=(6,4))
hist_plot(np.log(users_0['account_age'].values.clip(0,1000000000)), 'account age (seconds)','count', 'account age of all tweets for for each NON-BOT')
plt.figure(figsize=(6,4))
hist_plot(np.log(users_1['account_age'].values.clip(0,1000000000)), 'account age (seconds)','count', 'account age of all tweets for for each BOT')
```



![png](Testing_NB_files/Testing_NB_117_0.png)



![png](Testing_NB_files/Testing_NB_117_1.png)




```python
# quick plots
plt.figure(figsize=(10,6))
scatterplot(np.log(users_1['account_age'].values.clip(0,1000000000)), np.log(users_1['tweet_time_min'].values.clip(1, 1000)),
          np.log(users_0['account_age'].values.clip(0,1000000000)), np.log(users_0['tweet_time_min'].values.clip(1, 1000)),
           'account_age', 'tweet_time_min')
```



![png](Testing_NB_files/Testing_NB_118_0.png)




```python
# quick plots
plt.figure(figsize=(10,6))
scatterplot2(np.log(tweets_1['user_followers_count'].values.clip(0,1000000000)), tweets_1['retweet_count'].values.clip(0,150),
          np.log(tweets_0['user_followers_count'].values.clip(0,1000000000)), tweets_0['retweet_count'].values.clip(0,150),
           'user_followers_count', 'retweet_count')
```



![png](Testing_NB_files/Testing_NB_119_0.png)


Although the word count of the retweeted post for each tweet has an interesting pattern at around 80 word counts (has most bots), we decided not to include as the rest of the bots are very well blended with non-bots regarding retweet count, and the clusters we've observed above, given the rest of the plot, might be outliers or special events.



```python
# quick plots
plt.figure(figsize=(10,6))
scatterplot2(np.log(tweets_1['user_statuses_count'].values.clip(0,1000000000)), tweets_1['tweet_word_mean'].values.clip(0,250),
          np.log(tweets_0['user_statuses_count'].values.clip(0,1000000000)), tweets_0['tweet_word_mean'].values.clip(0,250),
           'user_statuses_count', 'tweet_word_mean')
```



![png](Testing_NB_files/Testing_NB_121_0.png)


[Back to TOC](#TOC) <br/>
<a id ='Relations-in-Data'></a>
#### 3.4 - Relations in Data
Last step before wrapping up the preprocessing, we want to explore the correlation among the different features.



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



![png](Testing_NB_files/Testing_NB_123_0.png)




```python
from pandas.plotting import scatter_matrix

scatter_matrix(tweets_df[col_pred_numerical], alpha=0.5, figsize=(25,20));
```



![png](Testing_NB_files/Testing_NB_124_0.png)




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
#### 3.5 - Standardization and Discussion

Up to this point, it has become very obvious that the most interesting features in telling bots / non-bots apart are account-level features. Moreover, account-level engineered features are much better at telling bot / non-bots apart. <br/>

Moreover, from the plots in the previous sections, it became obvious that many of our data are not normally distributed. Standardization is necessary to make our model legit. <br/>


As the last step of the EDA and data preprocessing, we consolidated all our account-level data, remove the columns that no longer be useful in our analysis, the standardize the numerical features.



```python
# read the data
users_df = pd.read_json('users_final.json')
#tweets_df = pd.read_json('tweets_clean_final2.json')
```




```python
users_df.columns.values
```





    array(['favorite_count', 'id', 'retweet_count', 'retweeted_status',
           'user_description', 'user_favourites_count',
           'user_followers_count', 'user_friends_count', 'user_listed_count',
           'user_statuses_count', 'text_rt', 'text_tweet', 'screen_name',
           'class_boto', 'tweet_time_mean', 'tweet_time_std',
           'tweet_time_min', 'tweet_time_max', 'user_description_len',
           'account_age', 'tweet_len_mean', 'tweet_len_std',
           'tweet_word_mean', 'tweet_word_std', 'retweet_len_mean',
           'retweet_len_std', 'retweet_word_mean', 'retweet_word_std'],
          dtype=object)





```python
# we want to check how many accounts have left after all the cleansing
users_df.shape
```





    (4226, 28)





```python
# we want to delete the columns that are no longer useful
users_df = users_df.drop(columns=['favorite_count', 'retweet_count', 'retweeted_status'])
```




```python
display(users_df.columns.values)
display(users_df.shape)
```



    array(['id', 'user_description', 'user_favourites_count',
           'user_followers_count', 'user_friends_count', 'user_listed_count',
           'user_statuses_count', 'text_rt', 'text_tweet', 'screen_name',
           'class_boto', 'tweet_time_mean', 'tweet_time_std',
           'tweet_time_min', 'tweet_time_max', 'user_description_len',
           'account_age', 'tweet_len_mean', 'tweet_len_std',
           'tweet_word_mean', 'tweet_word_std', 'retweet_len_mean',
           'retweet_len_std', 'retweet_word_mean', 'retweet_word_std'],
          dtype=object)



    (4226, 25)


We still have 25 columns, which include two reference columns ('id' and 'screen_name'), one predictor column ('class_boto'). 



```python
users_df.dtypes
```





    id                         int64
    user_description          object
    user_favourites_count      int64
    user_followers_count       int64
    user_friends_count         int64
    user_listed_count          int64
    user_statuses_count        int64
    text_rt                   object
    text_tweet                object
    screen_name               object
    class_boto                 int64
    tweet_time_mean          float64
    tweet_time_std           float64
    tweet_time_min           float64
    tweet_time_max           float64
    user_description_len       int64
    account_age                int64
    tweet_len_mean           float64
    tweet_len_std            float64
    tweet_word_mean          float64
    tweet_word_std           float64
    retweet_len_mean         float64
    retweet_len_std          float64
    retweet_word_mean        float64
    retweet_word_std         float64
    dtype: object





```python
# separate numerical columns and text columns again
col_response = ['class_boto']
col_pred_text = list(users_df.select_dtypes(['object']).drop(columns=['screen_name']).columns.values)
col_ref = ['id', 'screen_name']
col_pred_numerical = list(users_df.select_dtypes(['float64', 'int64']).drop(columns=['class_boto', 'id']).columns.values)
```




```python
# save the column lists
c_list_names = ['col_pred_numerical', 'col_ref', 'col_response', 'col_pred_text']
c_list = [col_pred_numerical, col_ref, col_response, col_pred_text]
for c_name, c in zip(c_list_names, c_list):
    with open(c_name+'.txt', 'w') as fp:
        ls_str = ",".join(col_pred_numerical)
        fp.write(ls_str)
```




```python
display(users_df.shape)
display(users_df.isna().any())
```



    (4226, 25)



    id                       False
    user_description          True
    user_favourites_count    False
    user_followers_count     False
    user_friends_count       False
    user_listed_count        False
    user_statuses_count      False
    text_rt                   True
    text_tweet                True
    screen_name              False
    class_boto               False
    tweet_time_mean          False
    tweet_time_std           False
    tweet_time_min           False
    tweet_time_max           False
    user_description_len     False
    account_age              False
    tweet_len_mean            True
    tweet_len_std             True
    tweet_word_mean           True
    tweet_word_std            True
    retweet_len_mean          True
    retweet_len_std           True
    retweet_word_mean         True
    retweet_word_std          True
    dtype: bool




```python
# cleaning up NaN on numerical columns by assigning them 0
users_df[col_pred_numerical] = users_df[col_pred_numerical].fillna(0)
#users_df = users_df.dropna(subset=['tweet_len_mean', 'tweet_len_std'])
```




```python
from sklearn import preprocessing

def standardize(df):
    scaler = preprocessing.StandardScaler()
    df = scaler.fit_transform(df)
    return df
```




```python
# create a new copy with numercial columns standardized
users_df[col_pred_numerical] = standardize(users_df[col_pred_numerical])
```




```python
# check if the copy 
display(users_df.describe())
display(users_df.shape)
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
      <th>id</th>
      <th>user_favourites_count</th>
      <th>user_followers_count</th>
      <th>user_friends_count</th>
      <th>user_listed_count</th>
      <th>user_statuses_count</th>
      <th>class_boto</th>
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
      <td>4.226000e+03</td>
      <td>4.226000e+03</td>
      <td>4.226000e+03</td>
      <td>4.226000e+03</td>
      <td>4.226000e+03</td>
      <td>4.226000e+03</td>
      <td>4226.000000</td>
      <td>4.226000e+03</td>
      <td>4.226000e+03</td>
      <td>4.226000e+03</td>
      <td>4.226000e+03</td>
      <td>4.226000e+03</td>
      <td>4.226000e+03</td>
      <td>4.226000e+03</td>
      <td>4.226000e+03</td>
      <td>4.226000e+03</td>
      <td>4.226000e+03</td>
      <td>4.226000e+03</td>
      <td>4.226000e+03</td>
      <td>4.226000e+03</td>
      <td>4.226000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.072062e+18</td>
      <td>-2.878015e-17</td>
      <td>1.523732e-17</td>
      <td>8.557860e-18</td>
      <td>-1.315599e-16</td>
      <td>-1.088287e-17</td>
      <td>0.082584</td>
      <td>1.884962e-18</td>
      <td>-3.707530e-18</td>
      <td>3.383901e-17</td>
      <td>1.157249e-17</td>
      <td>4.057332e-16</td>
      <td>-1.116003e-16</td>
      <td>-6.029252e-18</td>
      <td>3.007533e-16</td>
      <td>-4.242544e-16</td>
      <td>3.718695e-16</td>
      <td>1.810615e-16</td>
      <td>1.399732e-16</td>
      <td>-1.968242e-16</td>
      <td>-3.429974e-16</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.518030e+15</td>
      <td>1.000118e+00</td>
      <td>1.000118e+00</td>
      <td>1.000118e+00</td>
      <td>1.000118e+00</td>
      <td>1.000118e+00</td>
      <td>0.275285</td>
      <td>1.000118e+00</td>
      <td>1.000118e+00</td>
      <td>1.000118e+00</td>
      <td>1.000118e+00</td>
      <td>1.000118e+00</td>
      <td>1.000118e+00</td>
      <td>1.000118e+00</td>
      <td>1.000118e+00</td>
      <td>1.000118e+00</td>
      <td>1.000118e+00</td>
      <td>1.000118e+00</td>
      <td>1.000118e+00</td>
      <td>1.000118e+00</td>
      <td>1.000118e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>8.705541e+17</td>
      <td>-4.849644e-01</td>
      <td>-9.288413e-02</td>
      <td>-1.882989e-01</td>
      <td>-1.647460e-01</td>
      <td>-2.397743e-01</td>
      <td>0.000000</td>
      <td>-7.548576e-02</td>
      <td>-1.240359e-01</td>
      <td>-4.307066e-02</td>
      <td>-1.539493e-01</td>
      <td>-1.421002e+00</td>
      <td>-1.647139e+00</td>
      <td>-1.923047e+00</td>
      <td>-1.686625e+00</td>
      <td>-3.630790e+00</td>
      <td>-2.858170e+00</td>
      <td>-3.152194e+00</td>
      <td>-2.844186e+00</td>
      <td>-3.748898e+00</td>
      <td>-3.458390e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.072195e+18</td>
      <td>-4.587417e-01</td>
      <td>-8.961513e-02</td>
      <td>-1.677584e-01</td>
      <td>-1.625237e-01</td>
      <td>-2.218025e-01</td>
      <td>0.000000</td>
      <td>-7.474640e-02</td>
      <td>-1.219509e-01</td>
      <td>-4.307066e-02</td>
      <td>-1.501365e-01</td>
      <td>-9.950487e-01</td>
      <td>-9.701341e-01</td>
      <td>-5.238541e-01</td>
      <td>-6.050294e-01</td>
      <td>-7.939639e-03</td>
      <td>-1.203602e-01</td>
      <td>-1.679538e-01</td>
      <td>-3.083459e-02</td>
      <td>1.481374e-01</td>
      <td>8.063067e-02</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.072253e+18</td>
      <td>-3.381680e-01</td>
      <td>-8.131076e-02</td>
      <td>-1.281914e-01</td>
      <td>-1.469681e-01</td>
      <td>-1.717429e-01</td>
      <td>0.000000</td>
      <td>-7.284825e-02</td>
      <td>-1.185913e-01</td>
      <td>-4.297456e-02</td>
      <td>-1.465433e-01</td>
      <td>9.539089e-02</td>
      <td>1.152168e-01</td>
      <td>-8.393633e-02</td>
      <td>1.221680e-01</td>
      <td>1.751376e-01</td>
      <td>1.072919e-01</td>
      <td>2.923449e-01</td>
      <td>1.986888e-01</td>
      <td>2.297437e-01</td>
      <td>1.910382e-01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.072275e+18</td>
      <td>2.087058e-02</td>
      <td>-5.314358e-02</td>
      <td>-1.888093e-02</td>
      <td>-8.030114e-02</td>
      <td>-2.021151e-02</td>
      <td>0.000000</td>
      <td>-6.288424e-02</td>
      <td>-1.066719e-01</td>
      <td>-4.285444e-02</td>
      <td>-1.328250e-01</td>
      <td>9.984112e-01</td>
      <td>8.490765e-01</td>
      <td>4.698467e-01</td>
      <td>5.891938e-01</td>
      <td>3.523146e-01</td>
      <td>3.257232e-01</td>
      <td>5.515590e-01</td>
      <td>3.881942e-01</td>
      <td>3.274733e-01</td>
      <td>2.954122e-01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.072320e+18</td>
      <td>1.793084e+01</td>
      <td>5.192418e+01</td>
      <td>4.162489e+01</td>
      <td>5.010437e+01</td>
      <td>5.402938e+01</td>
      <td>1.000000</td>
      <td>5.308569e+01</td>
      <td>3.565216e+01</td>
      <td>5.440004e+01</td>
      <td>2.301665e+01</td>
      <td>1.543631e+00</td>
      <td>2.090942e+00</td>
      <td>1.103629e+01</td>
      <td>8.016903e+00</td>
      <td>1.045978e+01</td>
      <td>1.460471e+01</td>
      <td>5.003616e+00</td>
      <td>7.567214e+00</td>
      <td>4.688847e+00</td>
      <td>1.645674e+01</td>
    </tr>
  </tbody>
</table>
</div>



    (4226, 25)




```python
# save to json
users_df.to_json('users_final_std.json')
```


[Back to TOC](#TOC) <br/>
<a id ='Models'></a>
### 4 - Models

<mark> Some Text Here </mark>



```python
# read the data
users_df = pd.read_json('users_final_std.json')
```




```python
# Train/Test split 
'''
change as needed, do we want test_size of .25?
'''
train_df, test_df = train_test_split(users_df, test_size=.25, 
                                                       stratify=users_df.class_boto, random_state=99)
```




```python
with open('col_pred_numerical.txt', 'r') as fp:
    col_pred_numerical = fp.read().split(',')
with open('col_response.txt', 'r') as fp:
    col_response = fp.read().split(',')
with open('col_pred_text.txt', 'r') as fp:
    col_pred_text = fp.read().split(',')
with open('col_ref.txt', 'r') as fp:
    col_ref = fp.read().split(',')
```




```python
# write a function to split the data
def split_data(df):
    # num_pred: standardized numerical predictors - what we will be using for most of the models
    # text_pred: text features that are associated with the tweets - only useful for NLP
    # response: response - manually verified classification. 1=bot; 0=non-bot
    # ids: 'id'
    # boto: botometer values
    num_pred, text_pred, response = df[col_pred_numerical], df[col_pred_text], df['class_boto']
    ids, screen_name = df['id'], df['screen_name']
    return num_pred, text_pred, response, ids, screen_name
```




```python
# get the predictors, responses, and other features from train and test set
xtrain, xtrain_text, ytrain, train_id, train_sn = split_data(train_df)
xtest, xtest_text, ytest, test_id, test_sn = split_data(test_df)
```




```python
# save to json
f_list_names = ['train_df', 'test_df', 'xtrain', 'xtrain_text', 'ytrain', 'train_id', 'train_sn', 'xtest', 'xtest_text', 'ytest', 'test_id', 'test_sn']
f_list = [train_df, test_df, xtrain, xtrain_text, ytrain, train_id, train_sn, xtest, xtest_text, ytest, test_id, test_sn]
for f_name, f in zip(f_list_names, f_list):
    f.to_json(f_name + '.json')
```




```python
# create a dictioary to store all our models
models_list = {}
```


[Back to TOC](#TOC) <br/>
<a id ='Baseline-Model'></a>
#### 4.1 - Baseline Model - Simple Linear Regression

<mark> Some Text and Code Here </mark>



```python
# multiple linear regression(no poly)on numerical predictors
X_train = sm.add_constant(xtrain)
X_test = sm.add_constant(xtest)
y_train = ytrain.values.reshape(-1,1)
y_test = ytest.values.reshape(-1,1)
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
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.217</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.212</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   45.83</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 11 Dec 2018</td> <th>  Prob (F-statistic):</th> <td>3.40e-151</td>
</tr>
<tr>
  <th>Time:</th>                 <td>03:31:26</td>     <th>  Log-Likelihood:    </th> <td> -23.162</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  3169</td>      <th>  AIC:               </th> <td>   86.32</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  3149</td>      <th>  BIC:               </th> <td>   207.5</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    19</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
            <td></td>               <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                 <td>    0.0845</td> <td>    0.004</td> <td>   19.328</td> <td> 0.000</td> <td>    0.076</td> <td>    0.093</td>
</tr>
<tr>
  <th>user_favourites_count</th> <td>   -0.0111</td> <td>    0.004</td> <td>   -2.478</td> <td> 0.013</td> <td>   -0.020</td> <td>   -0.002</td>
</tr>
<tr>
  <th>user_followers_count</th>  <td>   -0.0473</td> <td>    0.011</td> <td>   -4.433</td> <td> 0.000</td> <td>   -0.068</td> <td>   -0.026</td>
</tr>
<tr>
  <th>user_friends_count</th>    <td>    0.0281</td> <td>    0.005</td> <td>    5.138</td> <td> 0.000</td> <td>    0.017</td> <td>    0.039</td>
</tr>
<tr>
  <th>user_listed_count</th>     <td>    0.0332</td> <td>    0.009</td> <td>    3.816</td> <td> 0.000</td> <td>    0.016</td> <td>    0.050</td>
</tr>
<tr>
  <th>user_statuses_count</th>   <td>   -0.0052</td> <td>    0.004</td> <td>   -1.218</td> <td> 0.223</td> <td>   -0.014</td> <td>    0.003</td>
</tr>
<tr>
  <th>tweet_time_mean</th>       <td>    0.1159</td> <td>    0.039</td> <td>    2.964</td> <td> 0.003</td> <td>    0.039</td> <td>    0.193</td>
</tr>
<tr>
  <th>tweet_time_std</th>        <td>   -0.0043</td> <td>    0.029</td> <td>   -0.148</td> <td> 0.883</td> <td>   -0.061</td> <td>    0.052</td>
</tr>
<tr>
  <th>tweet_time_min</th>        <td>   -0.0337</td> <td>    0.008</td> <td>   -4.427</td> <td> 0.000</td> <td>   -0.049</td> <td>   -0.019</td>
</tr>
<tr>
  <th>tweet_time_max</th>        <td>   -0.0053</td> <td>    0.015</td> <td>   -0.360</td> <td> 0.719</td> <td>   -0.034</td> <td>    0.023</td>
</tr>
<tr>
  <th>user_description_len</th>  <td>   -0.0011</td> <td>    0.005</td> <td>   -0.234</td> <td> 0.815</td> <td>   -0.010</td> <td>    0.008</td>
</tr>
<tr>
  <th>account_age</th>           <td>   -0.0367</td> <td>    0.004</td> <td>   -8.171</td> <td> 0.000</td> <td>   -0.046</td> <td>   -0.028</td>
</tr>
<tr>
  <th>tweet_len_mean</th>        <td>    0.0171</td> <td>    0.007</td> <td>    2.621</td> <td> 0.009</td> <td>    0.004</td> <td>    0.030</td>
</tr>
<tr>
  <th>tweet_len_std</th>         <td>   -0.0579</td> <td>    0.006</td> <td>   -9.066</td> <td> 0.000</td> <td>   -0.070</td> <td>   -0.045</td>
</tr>
<tr>
  <th>tweet_word_mean</th>       <td>   -0.0577</td> <td>    0.008</td> <td>   -7.453</td> <td> 0.000</td> <td>   -0.073</td> <td>   -0.043</td>
</tr>
<tr>
  <th>tweet_word_std</th>        <td>    0.0065</td> <td>    0.007</td> <td>    0.871</td> <td> 0.384</td> <td>   -0.008</td> <td>    0.021</td>
</tr>
<tr>
  <th>retweet_len_mean</th>      <td>    0.0167</td> <td>    0.008</td> <td>    2.003</td> <td> 0.045</td> <td>    0.000</td> <td>    0.033</td>
</tr>
<tr>
  <th>retweet_len_std</th>       <td>    0.0007</td> <td>    0.007</td> <td>    0.107</td> <td> 0.915</td> <td>   -0.013</td> <td>    0.014</td>
</tr>
<tr>
  <th>retweet_word_mean</th>     <td>   -0.1547</td> <td>    0.016</td> <td>   -9.530</td> <td> 0.000</td> <td>   -0.187</td> <td>   -0.123</td>
</tr>
<tr>
  <th>retweet_word_std</th>      <td>    0.0783</td> <td>    0.015</td> <td>    5.145</td> <td> 0.000</td> <td>    0.048</td> <td>    0.108</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>1418.815</td> <th>  Durbin-Watson:     </th> <td>   1.975</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>6760.888</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 2.163</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 8.699</td>  <th>  Cond. No.          </th> <td>    21.2</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.





```python
y_hat_train = results.predict()
y_hat_test = results.predict(exog=X_test)

# get Train & Test R^2
print('Train R^2 = {}'.format(results.rsquared))
print('Test R^2 = {}'.format(r2_score(test_df['class_boto'], y_hat_test)))
```


    Train R^2 = 0.21662050202335137
    Test R^2 = -0.2992911497645412




```python
# accuracy score
ols_train_acc = accuracy_score(y_train, results.predict(X_train).round())
ols_test_acc = accuracy_score(y_test, results.predict(X_test).round())
print("Training accuracy is {:.4}%".format(ols_train_acc*100))
print("Test accuracy is {:.4} %".format(ols_test_acc*100))
```


    Training accuracy is 91.86%
    Test accuracy is 91.2 %




```python
# save model to the list
models_list["ols"] = results
```




```python
# pickle ols
import pickle

filename = 'ols.sav'
pickle.dump(results, open(filename, 'wb'))
```




```python
#loaded_model = pickle.load(open(filename,'rb'))
```


[Back to TOC](#TOC) <br/>
<a id ='Baseline-Model'></a>
#### 4.2a - Linear Regression with Ridge

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


    Training accuracy is 91.92%
    Test accuracy is 91.86 %




```python
# save model to the list
models_list["ridge"] = fitted_ridge
filename = 'ridge.sav'
pickle.dump(fitted_ridge, open(filename, 'wb'))
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


    Training accuracy is 91.8%
    Test accuracy is 91.77 %




```python
# save model to the list
models_list["lasso"] = fitted_lasso
filename = 'lasso.sav'
pickle.dump(fitted_lasso, open(filename, 'wb'))
```


[Back to TOC](#TOC) <br/>
<a id ='Logistic-Regression'></a>
#### 4.2 - Logistic Regression

<mark> Some Text and Code Here </mark>



```python
X_train = sm.add_constant(xtrain)
X_test = sm.add_constant(xtest)

logistic_model = LogisticRegression().fit(X_train, ytrain)

logistic_model_score = logistic_model.score(X_test, ytest)

print("Train set score: {0:4.4}%".format(logistic_model.score(X_train, ytrain)*100))
print("Test set score: {0:4.4}%".format(logistic_model.score(X_test, ytest)*100))
```


    Train set score: 92.55%
    Test set score: 91.49%




```python
models_list["simple_logistic"] = logistic_model
filename = 'simple_logistic.sav'
pickle.dump(logistic_model, open(filename, 'wb'))
```


[Back to TOC](#TOC) <br/>
<a id ='Logistic-Regression'></a>
#### 4.3a - Logistic Regression with cross validation

<mark> Nisrine, team please check </mark>



```python
logistic_model_cv = LogisticRegressionCV(Cs=[1,10,100,1000,10000], cv=3, penalty='l2', 
                                       solver='newton-cg').fit(X_train,ytrain)

print("Train set score with Cross Validation: {0:4.4}%".format(logistic_model_cv.score(X_train, ytrain)*100))
print("Test set score with Cross Validation: {0:4.4}%".format(logistic_model_cv.score(X_test, ytest)*100))
```


    Train set score with Cross Validation: 92.58%
    Test set score with Cross Validation: 91.96%




```python
models_list["simple_logistic_Cross_Validation"] = logistic_model_cv
filename = 'logistic_model_cv.sav'
pickle.dump(logistic_model_cv, open(filename, 'wb'))
```


[Back to TOC](#TOC) <br/>
<a id ='Logistic-Regression'></a>
#### 4.3b - Logistic Regression with polynomial degree 3

<mark> Nisrine, team please check </mark>



```python
X_train_poly = PolynomialFeatures(degree=3, include_bias=False).fit_transform(X_train)

logistic_model_poly_cv = LogisticRegressionCV(Cs=[1,10,100,1000,10000], cv=3, penalty='l2', 
                                       solver='newton-cg').fit(X_train_poly,ytrain)

X_test_poly = PolynomialFeatures(degree=3, include_bias=False).fit_transform(X_test)
print("Train set score with Polynomial Features (degree=3) and with Cross Validation: {0:4.4}%".
      format(logistic_model_poly_cv.score(X_train_poly, ytrain)*100))
print("Test set score with Polynomial Features (degree=3) and with Cross Validation: {0:4.4}%".
      format(logistic_model_poly_cv.score(X_test_poly, ytest)*100))
```


    Train set score with Polynomial Features (degree=3) and with Cross Validation: 98.26%
    Test set score with Polynomial Features (degree=3) and with Cross Validation: 93.47%




```python
models_list["poly_logistic_cv"] = logistic_model_poly_cv
filename = 'logistic_model_poly_cv.sav'
pickle.dump(logistic_model_poly_cv, open(filename, 'wb'))
```


[Back to TOC](#TOC) <br/>
<a id ='KNN'></a>
#### 4.3 - KNN

<mark> Some Text and Code Here </mark>



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
      <th>k</th>
      <th>Train R^2</th>
      <th>Test R^2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1.000000</td>
      <td>-0.027065</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.679619</td>
      <td>0.145156</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.557569</td>
      <td>0.238747</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.480681</td>
      <td>0.262580</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.441289</td>
      <td>0.264020</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0.424308</td>
      <td>0.263450</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0.407384</td>
      <td>0.269961</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0.383553</td>
      <td>0.262189</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>0.375727</td>
      <td>0.258386</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>0.367518</td>
      <td>0.259762</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>0.362127</td>
      <td>0.259462</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>0.354124</td>
      <td>0.259014</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>0.343383</td>
      <td>0.254270</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>0.340493</td>
      <td>0.257627</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>0.332997</td>
      <td>0.267806</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>0.329544</td>
      <td>0.266690</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>0.325765</td>
      <td>0.268989</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>0.321945</td>
      <td>0.271297</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19</td>
      <td>0.317976</td>
      <td>0.270313</td>
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



![png](Testing_NB_files/Testing_NB_179_0.png)




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


    Best k: 19



![png](Testing_NB_files/Testing_NB_181_1.png)




```python
# evaluate classification accuracy
best_model_KNN_train_score = accuracy_score(ytrain, knn_predicted_train[best_k].round())
best_model_KNN_test_score = accuracy_score(ytest, knn_predicted_test[best_k].round())
print("Training accuracy is {:.4}%".format(best_model_KNN_train_score*100))
print("Test accuracy is {:.4} %".format(best_model_KNN_test_score*100))
```


    Training accuracy is 93.03%
    Test accuracy is 92.53 %




```python
# save model to the list
best_k = 7
best_k_7 = KNNModels[best_k].fit(xtrain, ytrain)

models_list["knn_7"] = best_k_7
filename = 'knn_7.sav'
pickle.dump(best_k_7, open(filename, 'wb'))
```


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



![png](Testing_NB_files/Testing_NB_186_0.png)




```python
best_model_DTC_train_score = accuracy_score(ytrain, best_model_DTC.predict(xtrain))
best_model_DTC_test_score = accuracy_score(ytest, best_model_DTC.predict(xtest))
print("Training accuracy is {:.4}%".format(best_model_DTC_train_score*100))
print("Test accuracy is {:.4}%".format(best_model_DTC_test_score*100))
```


    Training accuracy is 95.61%
    Test accuracy is 93.47%




```python
models_list["decision_tree"] = best_model_DTC
filename = 'decision_tree.sav'
pickle.dump(best_model_DTC, open(filename, 'wb'))
```


[Back to TOC](#TOC) <br/>
<a id ='Random-Forest'></a>
#### 4.4 -Random Forest

<mark> Some Text and Code Here </mark>



```python
rf = RandomForestClassifier(max_depth=6)
rf_model = rf.fit(xtrain, ytrain)
rf_train_acc = rf_model.score(xtrain, ytrain)
rf_test_acc = rf_model.score(xtest, ytest)

print("Random Forest Training accuracy is {:.4}%".format(rf_train_acc*100))
print("Random Forest Test accuracy is {:.4}%".format(rf_test_acc*100))
```


    Random Forest Training accuracy is 95.83%
    Random Forest Test accuracy is 93.57%




```python
models_list["random_forest"] = rf_model
filename = 'random_forest.sav'
pickle.dump(rf_model, open(filename, 'wb'))
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



![png](Testing_NB_files/Testing_NB_194_0.png)




```python
AdaBoost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=800, learning_rate=0.05)
AdaBoost_2 = AdaBoost.fit(xtrain, ytrain)
```




```python
models_list["AdaBoost_2"] = AdaBoost_2
filename = 'AdaBoost_2.sav'
pickle.dump(AdaBoost_2, open(filename, 'wb'))
```


[Back to TOC](#TOC) <br/>
<a id ='SVM'></a>
#### 4.5 -SVM

<mark> Some Text and Code Here </mark>



```python
# Import the Libraries Needed
from sklearn import svm
from sklearn.model_selection import GridSearchCV

# Load the Data

# Fit a SVM Model by Grid Search
# parameters = {'kernel':('linear','rbf','poly','sigmoid'), 'C':[0.01,0.1,1,10,100]}
# svc = svm.SVC(random_state=0)
# svm_model = GridSearchCV(svc, parameters, cv=5)
# svm_model.fit(X_train, ytrain)

# Fit a Model by Eyeballing
svm_model = svm.SVC(kernel='poly',C=1,degree=4, random_state=0)  
svm_model.fit(xtrain, ytrain) 

#models_list = []
#models_list["SVM"] =  svm_model
print("Train set score: {0:4.4}%".format(svm_model.score(xtrain, ytrain)*100))
print("Test set score: {0:4.4}%".format(svm_model.score(xtest, ytest)*100))
```


    Train set score: 94.98%
    Test set score: 93.28%




```python
filename = 'svm.sav'
pickle.dump(svm_model, open(filename, 'wb'))
```


[Back to TOC](#TOC) <br/>
<a id ='RNN'></a>
#### 4.6 -RNN

<mark> Are we doing RNN? </mark>

[Back to TOC](#TOC) <br/>
<a id ='KMeans-Clustering'></a>
#### 4.7 - K-Means Clustering

A quick application to explore natural clustering among users

<mark> Some Text and Code Here </mark>



```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, init='random', random_state=0).fit(users_df[col_pred_numerical].values)
```




```python
# add the classification result
k3 = users_df[col_pred_numerical]

k3['k=3'] = kmeans.labels_
```




```python
# create df for easy plot
kmean_0 = k3.loc[k3['k=3']==0]
kmean_1 = k3.loc[k3['k=3']==1]
kmean_2 = k3.loc[k3['k=3']==2]
#class_0 = tweets_df.loc[tweets_df['class_verified']==0]
#class_1 = tweets_df.loc[tweets_df['class_verified']==1]
```




```python
# see how many were classified as bots
print ('The size of the two clusters from kmeans clustering are {} and {}.'.format(len(kmean_0), len(kmean_1)))
```


    The size of the two clusters from kmeans clustering are 275 and 265.




```python
# quick plot to see if it naturally come into two clusters
plt.scatter(np.log(kmean_0['account_age']), np.log(kmean_0['tweet_time_std']), c='salmon', s=5, label = 'cluster 0', alpha=1)
plt.scatter(np.log(kmean_1['account_age']), np.log(kmean_1['tweet_time_std']), c='royalblue', s=5, label = 'cluster 1', alpha=1)
plt.scatter(np.log(kmean_2['account_age']), np.log(kmean_2['tweet_time_std']), c='gold', s=5, label = 'cluster 2', alpha=0.75)
#plt.scatter(class_0['account_age'], class_0['tweet_time_std'], c='red', s=2, label = 'cluster 0', alpha=0.2)
#plt.scatter(class_1['account_age'], class_1['tweet_time_std'], c='yellow', s=2, label = 'bots', alpha=1)
plt.xlabel('account_age')
plt.ylabel('tweet_time_std')
plt.title('KMeans Clustering with K=3');
plt.legend();
```



![png](Testing_NB_files/Testing_NB_206_0.png)


[Back to TOC](#TOC) <br/>
<a id ='KMeans-Clustering'></a>
#### 4.9 - Manually Verify Accounts and Compare Results
<mark> TO DO: Huan <mark>
  
When comparing botometer scores and manually classified results, we noticed that botometer does not always predict actual bot / non-bot correctly. <br/>

add a table showing botometer result and verified result

[Back to TOC](#TOC) <br/>
<a id ='KMeans-Clustering'></a>
#### 4.10 - Validate Botometer Results and Model Prediction Results
<mark> TO DO: Huan - will work on it later <mark>
 
We try to use a random forest to explore the subspace between botometer results and the actual result (manually verified classification). We chose to use non-linear model as we expect the relationship between botometer result and actual result to be non-linear.   <br/>
  
We want to train a model with two features plus botometer score as predictors, and the actual classification as the response. In the principle that the botometer is occasionally accurate, and we want to see under what occasions they are accurate / inaccurate, and therefore to capture the residuals between our predictions (which use botometer score as predictors) and the actual results. (* we chose to features as we want to minimize number of features, given our sample size - manually verified bot account - is only 50)

  



```python
with open('user_list_20.txt', 'r') as fp:
    user_list_20 = fp.read().split(',')
with open('bots_list_20.txt', 'r') as fp:
    bots_list_20 = fp.read().split(',')
```




```python
# use features and botometer score to predict the validated score

def prepare_vf(df, feature1, feature2):
    y_vf = df[['class_verified']]
    #y_vf = y_vf.dropna()
    x_vf = pd.DataFrame(df[[feature1, feature2, 'class_boto']])
    #x_vf = pd.merge(x_vf, train_tweets_df[[feature1, feature2, 'class_boto', 'id']], left_on='id', right_on='id') # TO DO: debug this line, something is wrong
    return x_vf, y_vf

x_train_vf, y_train_vf = prepare_vf(users_df, 'account_age', 'tweet_time_mean')
x_test_vf, y_test_vf = prepare_vf(users_df, 'account_age', 'tweet_time_mean')


rf_vf = RandomForestClassifier(max_depth=6)
rf_vf_model = rf_vf.fit(x_train_vf, y_train_vf)
score = rf_vf_model.score(x_test_vf, y_test_vf)

print("Random Forest model, (class_verified ~ account_age, tweet_time_mean, class_boto), the testscore is ", score)
```


[Back to TOC](#TOC) <br/>
<a id ='KMeans-Clustering'></a>
#### 4.1 - Model Comparisons

<mark> Compare all models, compare results to Botometer results </mark>



```python
### Summarize results (acc, test score, etc.)

models_list
```





    {'ols': <statsmodels.regression.linear_model.RegressionResultsWrapper at 0x2349f588ef0>,
     'ridge': RidgeCV(alphas=array([1.e-02, 5.e-02, 1.e-01, 5.e-01, 1.e+00, 5.e+00, 1.e+01, 5.e+01,
            1.e+02]),
         cv=5, fit_intercept=True, gcv_mode=None, normalize=False, scoring=None,
         store_cv_values=False),
     'lasso': LassoCV(alphas=array([1.e-02, 5.e-02, 1.e-01, 5.e-01, 1.e+00, 5.e+00, 1.e+01, 5.e+01,
            1.e+02]),
         copy_X=True, cv=5, eps=0.001, fit_intercept=True, max_iter=100000,
         n_alphas=100, n_jobs=1, normalize=False, positive=False,
         precompute='auto', random_state=None, selection='cyclic', tol=0.0001,
         verbose=False),
     'simple_logistic': LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
               intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
               penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
               verbose=0, warm_start=False),
     'simple_logistic_Cross_Validation': LogisticRegressionCV(Cs=[1, 10, 100, 1000, 10000], class_weight=None, cv=3,
                dual=False, fit_intercept=True, intercept_scaling=1.0,
                max_iter=100, multi_class='ovr', n_jobs=1, penalty='l2',
                random_state=None, refit=True, scoring=None, solver='newton-cg',
                tol=0.0001, verbose=0),
     'poly_logistic_cv': LogisticRegressionCV(Cs=[1, 10, 100, 1000, 10000], class_weight=None, cv=3,
                dual=False, fit_intercept=True, intercept_scaling=1.0,
                max_iter=100, multi_class='ovr', n_jobs=1, penalty='l2',
                random_state=None, refit=True, scoring=None, solver='newton-cg',
                tol=0.0001, verbose=0),
     'knn_7': KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=7, p=2,
               weights='uniform'),
     'random_forest': RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                 max_depth=6, max_features='auto', max_leaf_nodes=None,
                 min_impurity_decrease=0.0, min_impurity_split=None,
                 min_samples_leaf=1, min_samples_split=2,
                 min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                 oob_score=False, random_state=None, verbose=0,
                 warm_start=False),
     'decision_tree': DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5,
                 max_features=None, max_leaf_nodes=None,
                 min_impurity_decrease=0.0, min_impurity_split=None,
                 min_samples_leaf=1, min_samples_split=2,
                 min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                 splitter='best'),
     'AdaBoost_2': AdaBoostClassifier(algorithm='SAMME.R',
               base_estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=2,
                 max_features=None, max_leaf_nodes=None,
                 min_impurity_decrease=0.0, min_impurity_split=None,
                 min_samples_leaf=1, min_samples_split=2,
                 min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                 splitter='best'),
               learning_rate=0.05, n_estimators=800, random_state=None)}





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
