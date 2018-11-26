---
title: Overview:
notebook: Preliminary_EDA.ipynb
nav_include: 2
---

## Introduction and EDA

  
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
