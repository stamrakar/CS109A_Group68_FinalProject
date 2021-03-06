---
title: Results and Conclusions
notebook: Result_and_Conclusion.ipynb
nav_include: 3
---


<a id ='TOC'></a>

## Contents
{:.no_toc}
*  
{: toc}



___



```python
# Import Libraries, Global Options and Styles
import requests
from IPython.core.display import HTML
styles = requests.get(
    "https://raw.githubusercontent.com/Harvard-IACS/2018-CS109A/master/content/styles/cs109.css").text
HTML(styles)
%matplotlib inline

#import libraries
import pandas as pd


```


[Back to TOC](#TOC) <br/>
<a id ='Results-and-Conclusion'></a>
### 5 - Results and Conclusion

We did not have a lot of data to train our models, good labelling mechanics to label our data, and most of the users were not bots. However, we were able to train models that performs better than random guessing using the base rate for bots vs non-bots in our samples.

Our best model reached an accuracy of <mark>94%<mark>. <br/>

[Back to TOC](#TOC) <br/>
<a id ='Summary-of-Results'></a>
### 5.1 Summary of Results

#### Model Comparison and Analysis

We chose to use linear regression as our base model although this is a classification problem, we thought we could try with a threshold of 0.5. Our test score was 91.39% which is fair for a base model as first step. To improve the model we have run several models on user account data, and all the models were performing between 91% and 94% on the test data with Adaboost having the higher accuracy but these were done on a small sample datasets as we had manually to verify if accounts are Bots or non-Bots.

Alternatively, We wanted to train a model with one feature plus botometer score as predictors, and the actual classification as the response. While the model above improved accuracy from 72.73% to 83.33%, the model is very arbitary especially given that our sample size (44) is very small. However, this is an approach that could potentially be further devloped to improve prediction accuracy, especially to train a model with larger training with imperfect labels, and improve it with a smaller training set with better labels.

Finally, The team wanted to explore how we can read the text tweets to predict whether the tweets are coming from bot or human. This model was done to explore new techniques and discuss how we can use NLP on tweets data to identify bots and non-bots users.



```python
acc = pd.read_json('acc.json')
display(acc)

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
      <th>bl</th>
      <th>ols</th>
      <th>ridge</th>
      <th>lasso</th>
      <th>lm</th>
      <th>lm_cv3</th>
      <th>lm_poly3</th>
      <th>knn_17</th>
      <th>dtc</th>
      <th>rf</th>
      <th>adaboost</th>
      <th>svm_poly_c1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.917324</td>
      <td>0.918586</td>
      <td>0.919217</td>
      <td>0.917955</td>
      <td>0.925529</td>
      <td>0.925844</td>
      <td>0.982644</td>
      <td>0.930577</td>
      <td>0.966867</td>
      <td>0.957084</td>
      <td>0.956453</td>
      <td>0.949826</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.917692</td>
      <td>0.913907</td>
      <td>0.919584</td>
      <td>0.917692</td>
      <td>0.914853</td>
      <td>0.919584</td>
      <td>0.934721</td>
      <td>0.927152</td>
      <td>0.925260</td>
      <td>0.938505</td>
      <td>0.944182</td>
      <td>0.932829</td>
    </tr>
  </tbody>
</table>
</div>


[Back to TOC](#TOC) <br/>
<a id ='Noteworthy-Findings'></a>
#### 5.2 - Noteworthy Findings

**Botometer Label Accuracy**<br/>
We noticed that botometer scores were not always accurate. We were able to improve the botometer score prediction for actual bot / non-bot detection using a simple extended model. As we only have a small number of manually verified samples, the results we got was not perfect. However, there is an improvement could be achieved using this technique with a larger manually verified user dataset. <br/><br/> A generalization of this technique / approch is that it allow us to train a model using a large dataset with imperfect labels, use those predictions to train a model on a smaller dataset with better labels. This ensembled model could achieve an improvement on prediction than using the large dataset or the small datset alone. <br/><br/> We were able to get some promising initial results from an unsupervised KMeans model, which we could investigate further to see if we could avoid the need for using botometer labels. Similar to botometer score, KMeans clustering could also be used to a smaller dataset with manually verified labels to create an ensemble model.

**Class and Imbalance** <br/>
Among all the users, vastly majority of them were labeled as real users by botometer, which casued class imbalance in our data and potentially could result in very high accuracy (even if the model may not be that good). We tried to resolve this issue by stratify our data by botometer results, so similar proportion of bots were presented in trian and test set. <br/> <br/>
One thing we could have done, however, is use sampling to reach 50/50 balance. 

**Weights** <br/>
Another technique we could have done is to change loss functions to weight errors on bots higher. Similar to fraud detection in practive, we would want to make sure we do not miss any fraud (bots, in our case) as we can always verify fraud/non-fraud (bots/non-bots) with actual legit users (non-bot actual users), and we will get feedback. However, it would be very difficult to do the other way around.

[Back to TOC](#TOC) <br/>
<a id ='Noteworthy-Findings'></a>
#### 5.3 - Challenges

We have learned a lot during the project, especially on how to get data and performing feature engineering, which took up most of our time and much longer than we anticipated. This is mostly due to the following challenges that we have encountered during the process:

**Memory** <br/>
the 6000 * 200 tweets ended up to be a file of almost 7 GB. While we have access to computer with 48 Gb memory, it is still fail to load data some time. Not to mention that it becomes very challneging to run on regular PCs. This could potentially be resolved by only reading the json features that we are interested in. In that case, only part of the files will be read and it is easier for computers to handle. The memory issue is also the result of panda dataframe inefficiency and bad coding habbit (e.g. keep copying files without deleting them). 

**Downloading Data with Error** <br/>
One common thing we have encoutered quite often during the project is not except errors (tweepy errors, user_timeline errors, etc.), especially when running api. This often leads to a break with only one error and made the data collecting process longer than we expected.

**API Rate Limits** <br/>
Collecting many tweets / botometer scores have been time consuming due to API rate limits (both from twitter and botometer). However, we also found that some API pricing could be quite affordable. Regarding the time a paid API will save on a project, we would think of this option next time.

**Data Cleaning**<br/>
Data cleaning has been very challenging for this project - especially given the number of features embeded in each tweet, and the large number of missing data, errors, etc. Although sometimes we tried to first test on a small dataset, new errors would often occur when we tried to load a larger set of data.

**Lack of Labelled Data**<br/>
As we were not provided with labelled data for this project, we need to find labels by ourselves (using botometer, manual verification, etc.) in order to train and/or evaluate our success. Moreover, while self-claimed bots accounts are easy to identify, often times the bots with malicious intentions would try to pretend to be a normal user, which is franky quite difficult to tell sometimes even by going through all the tweets history and reading user profiles.

**Open Ended Challenge**<br/>
Unlike other assignments in the course, which we were provided with identified problems where approaches are clear and straightforward, for this challenge we were given an open ended challenge. Identifying the problem and design the approach have been very interesting but also challenging.

**Feature Engineering**<br/>
Feature engineering generated most of the predictors in the dataset we used to train models. We tried to aggregate tweet level data to account level to provide more insights for each user (e.g. more uniformed tweeting time might imply a bot). However, similar to idenfitying the problem, what features to look for, how to extract them, how to execute our plan, have been challenging and time consuming.

**DEBUGGING!**<br/>
From code not running, to graphs do not make senses, debugging has always be one of the most challenging part of this proejct. One thing we have found helpful is to debug systematically by breaking down the chunk of code one execution at a time.

[Back to TOC](#TOC) <br/>
<a id ='Conclusion-and-Future-Work'></a>
#### 5.3 -Conclusion and Future Work



We think we need to use both account user data and tweets text data with NLP to best identify bots and non-bots users in the future. We need to identy a bigger dataset manually as bots and non-bots or alternavely use clustering to identify objects in the same group that are more similiar then the other group which is in this case bot and non-bot.
