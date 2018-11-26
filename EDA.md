---
title: Overview:
notebook: Preliminary_EDA.ipynb
nav_include: 1
---

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
...

### Preliminary EDA

```python
# Libraries Needed
import json as json
import pandas as pd
```

```python
#open json file
data = []
# gdrive/My Drive/Twitter/tweets_sample.json
with open("gdrive/My Drive/Twitter/tweets_sample.json") as td:
    for line in td:
        data.append(json.loads(line))   
        
```
