---
title: Models
notebook: Models.ipynb
nav_include: 2
---

<a id ='TOC'></a>
### Table of Contents <br/>
4. [Models](#Models) <br/>
    4.1 [Baseline Model - Simple Linear Regression](#Baseline-Model) <br/>
    4.2 [Linear Regression with Ridge](#Linear-Regression-with-Ridge) <br/>
    4.3 [Lasso](#Lasso) <br/>
    4.4 [Lasso and Ridge Coefficients Comparison](#Lasso-and-Ridge-Coefficients-Comparison) <br/>
    4.5 [Logistic Regression](#Logistic-Regression) <br/>
    4.6 [Logistic Regression with cross validation](#Logistic-Regression-with-cross-validation) <br/>
    4.7 [Logistic Regression with polynomial degree 3](#Logistic-Regression-with-polynomial-degree-3) <br/>
    4.8 [KNN](#KNN) <br/>
    4.9 [Decision tree](#Decision-tree) <br/>
    4.10 [Random Forest](#Random-Forest) <br/>
    4.11 [Boosting - AdaBoost Classifier](#Boosting-AdaBoost-Classifier) <br/>
    4.12 [SVM](#SVM) <br/>
    4.13 [K-Means Clustering](#KMeans-Clustering) <br/>
    4.14 [Validate Botometer Results](#Validate-Botometer-Results) <br/>
    4.15 [Sentence Embeddings + Clutering + Neural Networks](#Sentence-Embeddings-Clutering-Neural-Networks)<br/>
___


```python
#@title 
# Import Libraries, Global Options and Styles
import requests
from IPython.core.display import HTML
styles = requests.get(
    "https://raw.githubusercontent.com/Harvard-IACS/2018-CS109A/master/content/styles/cs109.css").text
HTML(styles)
%matplotlib inline

#import libraries
import warnings
warnings.filterwarnings('ignore')

import tweepy
import random
random.seed(112358)

%matplotlib inline
import numpy as np
import scipy as sp
import json as json
import pandas as pd
import jsonpickle
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from pandas.plotting import scatter_matrix
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

import scipy.sparse as ss
import os
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from keras.utils import np_utils

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
<a id ='Models'></a>
### 4 - Models

We splited train / test dataset by 0.25 and stratify by class_boto to ensure equal presentation of bots account in both datasets. The baseline accuracy of training dataset was 91.73%, the baseline accuracy for test set was 91.77%. Both of which are quite high. <br/>
<mark> By testing several models, we were able to achieve an accuracy up to 94.4%. </mark> 



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
# create a dictionary to store all our models
models_list = {}
acc ={}
```




```python
# take a quick look at the accuracy if we just choose to classifying everything as users
baseline_train_acc = float(1-sum(ytrain)/len(ytrain))
baseline_test_acc = float(1-sum(ytest)/len(ytest))
print('the baseline accuracy for training set is {:.2f}%, for test set is {:.2f}%.'.format(baseline_train_acc*100, 
                                                                                           baseline_test_acc*100))
```


    the baseline accuracy for training set is 91.73%, for test set is 91.77%.




```python
# save baseline acc to model list
acc['bl'] = (baseline_train_acc, baseline_test_acc)
```


[Back to TOC](#TOC) <br/>
<a id ='Baseline-Model'></a>
#### 4.1 - Baseline Model - Simple Linear Regression

Although this is a classification problem that people normally won't use linear regression, we thought we could try with a threshold of 0.5 and use it as a baseline model.  <br/>

Our Test score is around 91.39% on the test data which is not bad for a Base Model at the first glance; as our possibilies are either Bot or No-Bot. However, it is actually lower than our baseline accuracy on test set, which was 91.77%. Therefore, OLS, even we tried to use threshold, it is not performing, we need to improve the model.



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
  <th>Date:</th>             <td>Wed, 12 Dec 2018</td> <th>  Prob (F-statistic):</th> <td>3.40e-151</td>
</tr>
<tr>
  <th>Time:</th>                 <td>21:45:50</td>     <th>  Log-Likelihood:    </th> <td> -23.162</td> 
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


    Train R^2 = 0.21662050202218985
    Test R^2 = -0.2992911496733639




```python
# accuracy score
ols_train_acc = accuracy_score(y_train, results.predict(X_train).round().clip(0, 1))
ols_test_acc = accuracy_score(y_test, results.predict(X_test).round().clip(0, 1))
print("Training accuracy is {:.4}%".format(ols_train_acc*100))
print("Test accuracy is {:.4} %".format(ols_test_acc*100))
```


    Training accuracy is 91.86%
    Test accuracy is 91.39 %




```python
# save model to the list
models_list["ols"] = results
acc['ols'] = (ols_train_acc, ols_test_acc)
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
<a id ='Linear-Regression-with-Ridge'></a>
#### 4.2 - Linear Regression with Ridge

Although in the simple linear model, the test score is comparable to training score and there was no sign of overfitting, we still want to try Ridge to see if we could reduce any potential overfitting. <br/>

With ridge selection, we received a test accuracy of 91.96%, which is slightly improved from 91.39% (OLS), which implies that the OLS model does not have overfitting. However, it is still about the same / lower than baseline accuracy.



```python
alphas = np.array([.01, .05, .1, .5, 1, 5, 10, 50, 100])
fitted_ridge = RidgeCV(alphas=alphas, cv=5).fit(X_train, y_train)
```




```python
# accuracy score
ridge_train_acc = accuracy_score(y_train, fitted_ridge.predict(X_train).round().clip(0, 1))
ridge_test_acc = accuracy_score(y_test, fitted_ridge.predict(X_test).round().clip(0, 1))
print("Training accuracy is {:.4}%".format(ridge_train_acc*100))
print("Test accuracy is {:.4} %".format(ridge_test_acc*100))
```


    Training accuracy is 91.92%
    Test accuracy is 91.96 %




```python
# save model to the list
models_list["ridge"] = fitted_ridge
filename = 'ridge.sav'
pickle.dump(fitted_ridge, open(filename, 'wb'))
acc['ridge'] = (ridge_train_acc, ridge_test_acc)
```


[Back to TOC](#TOC) <br/>
<a id ='Lasso'></a>
#### 4.3 - Lasso
We also want to try feature reductions with Lasso and see if the model will perform better by dropping less important features. The lasso model received an accuracy of 91.77%, which again improves from 91.14% but not very significant, and it is just slightly higher than baseline accuracy. However, Lasso may not have significant improvement on test accuracy but lead to differnet coefficients. We want to examine that.



```python
fitted_lasso = LassoCV(alphas=alphas, max_iter=100000, cv=5).fit(X_train, y_train)
```


    /anaconda3/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:1094: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)




```python
# accuracy score
lasso_train_acc = accuracy_score(y_train, fitted_lasso.predict(X_train).round().clip(0, 1))
lasso_test_acc = accuracy_score(y_test, fitted_lasso.predict(X_test).round().clip(0, 1))
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
acc['lasso']=(lasso_train_acc, lasso_test_acc)
```


[Back to TOC](#TOC) <br/>
<a id ='Lasso-and-Ridge-Coefficients-Comparison'></a>
#### 4.4 -  Lasso and Ridge Coefficients Comparison
We want to see how lasso and ridge results in different coefficients. As expected, Lasso greatly reduced the number of non-zero coefficients.



```python
for feature, coef in zip(xtrain.columns.values.tolist(), fitted_ridge.coef_[0].tolist()):
    print("{}: {}".format(feature, coef))
```


    user_favourites_count: 0.0
    user_followers_count: -0.010984392775297717
    user_friends_count: -0.0373860331993809
    user_listed_count: 0.024158234955190823
    user_statuses_count: 0.026646084813644076
    tweet_time_mean: -0.0044977208562929465
    tweet_time_std: 0.044232900606536285
    tweet_time_min: 0.029995359711894685
    tweet_time_max: -0.02480300277881136
    user_description_len: -0.012475212935169604
    account_age: -0.0015423611100371431
    tweet_len_mean: -0.03616251833633441
    tweet_len_std: 0.01451572510127823
    tweet_word_mean: -0.054079504222467004
    tweet_word_std: -0.0514121927327697
    retweet_len_mean: 0.0025870175288910087
    retweet_len_std: 0.007751629182111389
    retweet_word_mean: -0.003022968737127
    retweet_word_std: -0.09823618767855992




```python
for feature, coef in zip(xtrain.columns.values.tolist(), fitted_lasso.coef_.tolist()):
    print("{}: {}".format(feature, coef))
```


    user_favourites_count: 0.0
    user_followers_count: -0.0030684119532310974
    user_friends_count: -0.0
    user_listed_count: 0.005861006334253806
    user_statuses_count: 0.0
    tweet_time_mean: -0.0
    tweet_time_std: 0.0
    tweet_time_min: 0.019705055827586075
    tweet_time_max: -0.0
    user_description_len: 0.0
    account_age: -0.0
    tweet_len_mean: -0.02878718176736991
    tweet_len_std: 0.0
    tweet_word_mean: -0.04152323503512772
    tweet_word_std: -0.03718634181572111
    retweet_len_mean: -0.0
    retweet_len_std: -0.0
    retweet_word_mean: -0.0
    retweet_word_std: -0.06062048667775922


[Back to TOC](#TOC) <br/>
<a id ='Logistic-Regression'></a>
#### 4.5 - Logistic Regression

The logistic regression presented a small improvement on the accuracy from the base model, we need to try additional techniques to improve the accuracy.



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
acc['lm'] = (logistic_model.score(X_train, ytrain), logistic_model_score)
```


[Back to TOC](#TOC) <br/>
<a id ='Logistic-Regression-with-cross-validation'></a>
#### 4.6 - Logistic Regression with cross validation

Logistic regression with Cross Validation has improved the accuracy and reached 91.96% on Test data which is an improvement from the Logistic regression, we will continue to see if we can improve further using other techniques.



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
acc['lm_cv3'] = (logistic_model_cv.score(X_train, ytrain), logistic_model_cv.score(X_test, ytest))
```


[Back to TOC](#TOC) <br/>
<a id ='Logistic-Regression-with-polynomial-degree-3'></a>
#### 4.7 - Logistic Regression with polynomial degree 3

Test score accuracy has increased with Polynomial degree of predictors for Logistic Regression on the test data and reached 93.47%. 



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
acc['lm_poly3'] = (logistic_model_poly_cv.score(X_train_poly, ytrain), logistic_model_poly_cv.score(X_test_poly, ytest))
```


[Back to TOC](#TOC) <br/>
<a id ='KNN'></a>
#### 4.8 - KNN

We have tested the k-Nearest Neighbors algorithm as well and we used cross validation to evaluate the best k with the highest accuracy score. We have stored the best k in the variable best_k which has a value equal of 17. The test score is higher than the base model but lower than Logistic Regression with polynomial degree 3.



```python
# the code below in KNN is adapted from HW2 solution

# define k values
k_values = range(1,20)

# build a dictionary KNN models
KNNModels = {k: KNeighborsClassifier(n_neighbors=k) for k in k_values}
train_scores = [KNeighborsClassifier(n_neighbors=k).fit(xtrain, ytrain).score(xtrain, ytrain) for k in k_values]
cv_scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k), xtrain, ytrain, cv=5) for k in k_values]


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
fig, ax = plt.subplots(figsize=(12,7))
ax.plot(k_values, train_scores, '-+', label="Training")
plot_cv(ax, k_values, cv_scores)
plt.xlabel("n_neighbors")
plt.ylabel("Mean CV accuracy");
plt.legend()

best_k = k_values[np.argmax(np.mean(cv_scores, axis=1))]
print("Best k:", best_k)
```


    Best k: 17



![png](Models_files/Models_45_1.png)




```python
# evaluate classification accuracy
best_model_KNN_train_score = accuracy_score(ytrain, knn_predicted_train[best_k].round())
best_model_KNN_test_score = accuracy_score(ytest, knn_predicted_test[best_k].round())
print("Training accuracy is {:.4}%".format(best_model_KNN_train_score*100))
print("Test accuracy is {:.4} %".format(best_model_KNN_test_score*100))
```


    Training accuracy is 93.06%
    Test accuracy is 92.72 %




```python
# save model to the list
best_k = 17
best_k_17 = KNNModels[best_k].fit(xtrain, ytrain)

models_list["knn_17"] = best_k_17
filename = 'knn_17.sav'
pickle.dump(best_k_17, open(filename, 'wb'))
acc['knn_17'] = (best_model_KNN_train_score, best_model_KNN_test_score)
```


[Back to TOC](#TOC) <br/>
<a id ='Decision-tree'></a>
#### 4.9 - Decision tree

The decision tree is performing similiar to the logistic regression with polynomial 3. 



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



![png](Models_files/Models_50_0.png)




```python
best_model_DTC_train_score = accuracy_score(ytrain, best_model_DTC.predict(xtrain))
best_model_DTC_test_score = accuracy_score(ytest, best_model_DTC.predict(xtest))
print("Training accuracy is {:.4}%".format(best_model_DTC_train_score*100))
print("Test accuracy is {:.4}%".format(best_model_DTC_test_score*100))
```


    Training accuracy is 96.69%
    Test accuracy is 92.53%




```python
models_list["decision_tree"] = best_model_DTC
filename = 'decision_tree.sav'
pickle.dump(best_model_DTC, open(filename, 'wb'))
acc['dtc'] = (best_model_DTC_train_score, best_model_DTC_test_score )
```


[Back to TOC](#TOC) <br/>
<a id ='Random-Forest'></a>
#### 4.10 -Random Forest

The Random Forest is giving us the highest accuracy from all the models tested so far on the test data. but we may be able to increase this value with Boosting or Bagging.



```python
rf = RandomForestClassifier(max_depth=6)
rf_model = rf.fit(xtrain, ytrain)
rf_train_acc = rf_model.score(xtrain, ytrain)
rf_test_acc = rf_model.score(xtest, ytest)

print("Random Forest Training accuracy is {:.4}%".format(rf_train_acc*100))
print("Random Forest Test accuracy is {:.4}%".format(rf_test_acc*100))
```


    Random Forest Training accuracy is 95.71%
    Random Forest Test accuracy is 93.85%




```python
models_list["random_forest"] = rf_model
filename = 'random_forest.sav'
pickle.dump(rf_model, open(filename, 'wb'))
acc['rf'] = (rf_train_acc, rf_test_acc)
```


[Back to TOC](#TOC) <br/>
<a id ='Boosting-AdaBoost-Classifier'></a>
#### 4.11 -Boosting - AdaBoost Classifier

For the model with depth = 1, the accuracy for train and test datasets are close to each other. However, for the models with depth = 2, 3 and 4, there are a big difference in the accuracy for test and train data. I would choose depth =2 and iterations = 180. This model is performing the best so far.



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



![png](Models_files/Models_58_0.png)




```python
AdaBoost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=800, learning_rate=0.05)
AdaBoost_2 = AdaBoost.fit(xtrain, ytrain)
```




```python
models_list["AdaBoost_2"] = AdaBoost_2
filename = 'AdaBoost_2.sav'
pickle.dump(AdaBoost_2, open(filename, 'wb'))
acc['adaboost'] = (AdaBoost_scores_train[2][179], AdaBoost_scores_test[2][179])
```


[Back to TOC](#TOC) <br/>
<a id ='SVM'></a>
#### 4.12 -SVM

We tried SVM and reached a test accuracy of 93.28%. As it is an expensive model, we ended up using eyeballing to fit a model so we can try the SVM method. However, ideally, we would like to perform a grid search to find te best kernal and c value.



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
models_list["SVM"] =  svm_model
print("Train set score: {0:4.4}%".format(svm_model.score(xtrain, ytrain)*100))
print("Test set score: {0:4.4}%".format(svm_model.score(xtest, ytest)*100))
```


    Train set score: 94.98%
    Test set score: 93.28%




```python
filename = 'svm.sav'
pickle.dump(svm_model, open(filename, 'wb'))
acc['svm_poly_c1'] = (svm_model.score(xtrain, ytrain), svm_model.score(xtest, ytest))
```




```python
# we have finished all our models. we want to save the accuracy score and models to json
with open('models.pickle', 'wb') as handle:
    pickle.dump(models_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
acc = pd.DataFrame(acc)
acc.to_json('acc.json')
```


[Back to TOC](#TOC) <br/>
<a id ='KMeans-Clustering'></a>
#### 4.13 - K-Means Clustering

We want to explore if unsupervised k-means clustering align with bot / non-bot classification.



```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, init='random', random_state=0).fit(users_df[col_pred_numerical].values)
```




```python
# add the classification result
k2 = users_df[col_pred_numerical]

k2['k=2'] = kmeans.labels_
```




```python
# create df for easy plot
kmean_0 = k2.loc[k2['k=2']==0]
kmean_1 = k2.loc[k2['k=2']==1]
class_0 = users_df.loc[users_df['class_boto']==0]
class_1 = users_df.loc[users_df['class_boto']==1]
```




```python
# see how many were classified as bots
print ('The size of the two clusters from kmeans clustering are {} and {}.'.format(len(kmean_0), len(kmean_1)))
```


    The size of the two clusters from kmeans clustering are 277 and 3949.


Given the size of cluster 0, it looks like cluster 0 might be a bot cluster.<br/>

We picked two arbitary features to visualize the two clusters from unsupervised KMeans (k=2), and how they align with botometer classification. Visually they align well, and we want to see how many bots are in cluster 0 and non-bots in cluster 1.



```python
# quick plot to see if it naturally come into two clusters
plt.figure(figsize=(10,6))
plt.scatter(np.log(class_0['account_age']), np.log(class_0['tweet_time_std']), c='salmon', s=70, label = 'class_boto: non-bots', alpha=0.2)
plt.scatter(np.log(class_1['account_age']), np.log(class_1['tweet_time_std']), c='royalblue', s=70, label = 'class_bots', alpha=0.2)
plt.scatter(np.log(kmean_0['account_age']), np.log(kmean_0['tweet_time_std']), c='royalblue', s=7, label = 'cluster 0: possible bot', alpha=1)
plt.scatter(np.log(kmean_1['account_age']), np.log(kmean_1['tweet_time_std']), c='salmon', s=7, label = 'cluster 1: possible non-bot', alpha=1)
plt.xlabel('account_age')
plt.ylabel('tweet_time_std')
plt.title('KMeans Clustering with K=2')
plt.legend(loc='best', bbox_to_anchor=(1, 0., 0.5, 0.5));
```



![png](Models_files/Models_71_0.png)




```python
# proportion of cluster 0 users which are bots (precision)
precision_bot_0 = k2[(users_df['class_boto']==1) & (k2['k=2']==0)].shape[0] / kmean_0.shape[0]
print ('proportion of cluster 0 users which are bots (precision) is {:.2f}%'.format(precision_bot_0*100))
```


    proportion of cluster 0 users which are bots (precision) is 36.46%




```python
# proportion of bots which are in cluster 0 (recall)
recall_bot_0 = k2[(users_df['class_boto']==1) & (k2['k=2']==0)].shape[0] / class_1.shape[0]
print ('proportion of bots which are in cluster 0 (recall) is {:.2f}%'.format(recall_bot_0*100))
```


    proportion of bots which are in cluster 0 (recall) is 28.94%




```python
# proportion of cluster 1 users which are bots (precision)
precision_bot_1 = k2[(users_df['class_boto']==1) & (k2['k=2']==1)].shape[0] / kmean_1.shape[0]
print ('proportion of cluster 1 users which are bots (precision) is {:.2f}%'.format(precision_bot_1*100))
```


    proportion of cluster 1 users which are bots (precision) is 6.28%




```python
# proportion of bots which are in cluster 1 (recall)
recall_bot_1 = k2[(users_df['class_boto']==1) & (k2['k=2']==1)].shape[0] / class_1.shape[0]
print ('proportion of bots which are in cluster 0 (recall) is {:.2f}%'.format(recall_bot_1*100))
```


    proportion of bots which are in cluster 0 (recall) is 71.06%


However, when we look at precision and recall for cluster 0 being bots and cluster 1 being bots, we observed that clusters are not as well aligned with botometer classification as the graph is showing above. <br/>

It looks like cluster 0 would a better choice as bot cluster as it has a better precision. Therefore KMeans looks like a promising approach in identifying bots and non-bots with unsupervised learning. KMeans clustering could also be used in supervised learning model as a predictor.



```python
filename = 'kmeans.sav'
pickle.dump(kmeans, open(filename, 'wb'))
```


[Back to TOC](#TOC) <br/>
<a id ='Validate-Botometer-Results'></a>
#### 4.14 - Validate Botometer Results
  
When comparing botometer scores and manually classified results, we noticed that botometer does not always predict actual bot / non-bot correctly. Therefore we want to compare our verified users with Botometer classifications, and see if we can capture the subspace between botometer results and the manually verified results. <br/>

We try to use a random forest to explore the subspace between botometer results and the actual result (manually verified classification). We chose to use non-linear model as we expect the relationship between botometer result and actual result to be non-linear.  <br/>
  
We want to train a model with one feature plus botometer score as predictors, and the actual classification as the response. In the principle that the botometer is occasionally accurate, and we want to see under what occasions they are accurate / inaccurate, and therefore to capture the residuals between our predictions (which use botometer score as predictors) and the actual results. (* we chose to features as we want to minimize number of features, given our sample size - manually verified bot account - is only 44)

While the model above improved accuracy from 72.73% to 83.33%, the model is very arbitary especially given that our sample size (44) is very small. However, this is an approach that could potentially be further devloped to improve prediction accuracy, especially to train a model with larger training with imperfect labels, and improve it with a smaller training set with better labels.



```python
# load verified bots and nonbots
verify_df = pd.read_csv('boto_verify.csv')[['screen_name', 'class_verified']]
verify_df = verify_df[~verify_df['class_verified'].isnull()]
```




```python
# build a dataframe that has screen_name, class_bot, class_verified, feature 1
# we picked one arbitary features we think will be important
# and see if we can improve botometer's prediction on verified account accuracy using decision tree 
feature_1 = 'tweet_time_mean'
verify_df = pd.merge(verify_df, users_df[['class_boto', 'screen_name', feature_1]])
```




```python
# take a look at data
verify_df.drop(columns=['screen_name']).head(5)
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
      <th>class_verified</th>
      <th>class_boto</th>
      <th>tweet_time_mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1</td>
      <td>-0.074599</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1</td>
      <td>-0.075464</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0</td>
      <td>-0.074661</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0</td>
      <td>-0.075440</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0</td>
      <td>-0.075345</td>
    </tr>
  </tbody>
</table>
</div>





```python
# first we want to examine the accuracy of class_boto when cross checking with manually verified classifications
boto_vf_acc = sum(verify_df['class_boto']==verify_df['class_verified'])/len(verify_df['class_boto'])
print ('The accuracy of Botometer in predicting manually verified classification is {:.2f}%.'.format(boto_vf_acc*100))
```


    The accuracy of Botometer in predicting manually verified classification is 71.43%.




```python
# use features and botometer score to predict the validated score

x_train_vf, x_test_vf, y_train_vf, y_test_vf = train_test_split(verify_df[['class_boto', feature_1]], 
                                                                verify_df['class_verified'], test_size=0.4, random_state=50)

dtc_vf = DecisionTreeClassifier(max_depth=2).fit(x_train_vf, y_train_vf)
score = dtc_vf.score(x_test_vf, y_test_vf)

print("The accuracy of decision tree model (depth=3) in predicting manually verified classification is {:.2f}%.".format(score*100))
```


    The accuracy of decision tree model (depth=3) in predicting manually verified classification is 42.86%.


[Back to TOC](#TOC) <br/>
<a id ='Sentence-Embeddings-Clutering-Neural-Networks'></a>
#### 4.15 - Classification of tweets using Sentence Embeddings + Clutering + LDA + Neural Networks

Additionally, we want to explore some models on classifying tweets. 

The team wanted to explore for this project how we can read the text tweets to predict whether the tweets are coming from bot or human. First, we found out that the text tweets require data cleansing (by navigating through the tweets). So we took a sample data and performed manual data cleansing by replacing stopwords, special characters, emoji expressions, numbers and we saved the new clean data under cleaned_tweets.txt file. Then we decided to find how the data can be clustered and grouped together, so we have converted textual tweets data into numerical vectors using tensor flow encoder for the conversion and we have used text clustering using K-means (Mini Batch Kmeans). Then we labeled data into two categories (Group A and Group B as Bot and Human), at this stage we didn't manually labeled the data to check which tweets are coming from Bots or human (as this will require checking the records manually), so we just assigned the data to be labeled into two categories randomly as there are only two options a bot or non-bot. Then we build the Classification model using Neural Network on all sample data(Used Keras lib on top of tensor flow) The next step is to test the model on new datasets and checking the tweets content, this model was done to explore new techniques and discuss how we can do NLP on tweets data.

word embedding details https://www.tensorflow.org/tutorials/representation/word2vec https://www.tensorflow.org/guide/embedding https://www.fer.unizg.hr/_download/repository/TAR-07-WENN.pdf

Clustering https://scikit-learn.org/stable/modules/clustering.html#mini-batch-kmeans https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html https://algorithmicthoughts.wordpress.com/2013/07/26/machine-learning-mini-batch-k-means/ https://scikit-learn.org/stable/auto_examples/cluster/plot_mini_batch_kmeans.html

Classification http://www.zhanjunlang.com/resources/tutorial/Deep%20Learning%20with%20Keras.pdf https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/ https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/

# Sentence Embeddings for Clustering



```python
# converting textual data into numerical vectors for clustering; we have used tensor flow encoder for the conversion
def build_index(embedding_fun, batch_size, sentences):
    ann = []
    batch_sentences = []
    batch_indexes = []
    last_indexed = 0
    num_batches = 0
    with tf.Session() as sess: #starting tensor session 
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        with open('cleaned_tweets.txt', 'r') as fr:
            for sindex, sentence in enumerate(fr):
                batch_sentences.append(sentence)
                batch_indexes.append(sindex)

                if len(batch_sentences) == batch_size:
                    context_embed = sess.run(embedding_fun, feed_dict={sentences: batch_sentences})
                    for index in batch_indexes:
                        ann.append(context_embed[index - last_indexed])
                        batch_sentences = []
                        batch_indexes = []
                    last_indexed += batch_size
                    num_batches += 1
            if batch_sentences:
                context_embed = sess.run(embedding_fun, feed_dict={sentences: batch_sentences})
                for index in batch_indexes:
                    ann.append(context_embed[index - last_indexed])
    return ann
```




```python
batch_size = 128
embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
sentences = tf.placeholder(dtype=tf.string, shape=[None])
embedding_fun = embed(sentences)
ann = build_index(embedding_fun, batch_size, sentences)
```


    INFO:tensorflow:Using /var/folders/cd/js4b46vx0rq_2zt5bnm1fblw0000gn/T/tfhub_modules to cache modules.
    INFO:tensorflow:Saver not created because there are no variables in the graph to restore


# Text Clustering using Kmeans



```python
#We used Kmeans for clustering the data because data is not labeled 
from sklearn.cluster import MiniBatchKMeans
no_clus = 2
```




```python
km = MiniBatchKMeans(n_clusters=no_clus, random_state=0, batch_size=1000)
km = km.fit(ann)
```




```python
label_ = km.predict(ann)
```


# Labels Choosen after Cluster Analysis



```python
#we can give other labels to tweets after analysing the data but right now our motive is to identify bot & non-bots tweets.
label = ["human","bot"]
```


# Data Preperation for Training Neural Network



```python
#preparing the model fior neural netwwork
ds = pd.DataFrame()
for j in range(0,no_clus):
    temp = pd.DataFrame()
    temp = pd.DataFrame(np.array(ann)[np.where(label_ == j)[0]])
    temp['label'] = (label[j])
    ds = pd.concat((ds,temp), ignore_index = True)
ds.head() 
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
      <th>26</th>
      <th>27</th>
      <th>28</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
      <th>32</th>
      <th>33</th>
      <th>34</th>
      <th>35</th>
      <th>36</th>
      <th>37</th>
      <th>38</th>
      <th>39</th>
      <th>40</th>
      <th>41</th>
      <th>42</th>
      <th>43</th>
      <th>44</th>
      <th>45</th>
      <th>46</th>
      <th>47</th>
      <th>48</th>
      <th>49</th>
      <th>...</th>
      <th>463</th>
      <th>464</th>
      <th>465</th>
      <th>466</th>
      <th>467</th>
      <th>468</th>
      <th>469</th>
      <th>470</th>
      <th>471</th>
      <th>472</th>
      <th>473</th>
      <th>474</th>
      <th>475</th>
      <th>476</th>
      <th>477</th>
      <th>478</th>
      <th>479</th>
      <th>480</th>
      <th>481</th>
      <th>482</th>
      <th>483</th>
      <th>484</th>
      <th>485</th>
      <th>486</th>
      <th>487</th>
      <th>488</th>
      <th>489</th>
      <th>490</th>
      <th>491</th>
      <th>492</th>
      <th>493</th>
      <th>494</th>
      <th>495</th>
      <th>496</th>
      <th>497</th>
      <th>498</th>
      <th>499</th>
      <th>500</th>
      <th>501</th>
      <th>502</th>
      <th>503</th>
      <th>504</th>
      <th>505</th>
      <th>506</th>
      <th>507</th>
      <th>508</th>
      <th>509</th>
      <th>510</th>
      <th>511</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.014465</td>
      <td>-0.041160</td>
      <td>-0.080382</td>
      <td>0.049232</td>
      <td>-0.072470</td>
      <td>0.035281</td>
      <td>-0.007432</td>
      <td>-0.023164</td>
      <td>-0.016499</td>
      <td>0.059375</td>
      <td>0.011128</td>
      <td>-0.072195</td>
      <td>0.001333</td>
      <td>0.081120</td>
      <td>-0.043538</td>
      <td>0.025982</td>
      <td>-0.008040</td>
      <td>0.008672</td>
      <td>0.046465</td>
      <td>-0.041111</td>
      <td>0.019693</td>
      <td>-0.054316</td>
      <td>-0.037432</td>
      <td>-0.049775</td>
      <td>0.043199</td>
      <td>-0.082781</td>
      <td>0.022100</td>
      <td>-0.055881</td>
      <td>0.000243</td>
      <td>-0.039490</td>
      <td>0.046346</td>
      <td>0.045277</td>
      <td>0.063884</td>
      <td>-0.049139</td>
      <td>0.046108</td>
      <td>0.046417</td>
      <td>0.068380</td>
      <td>0.010075</td>
      <td>-0.024406</td>
      <td>0.054736</td>
      <td>0.036027</td>
      <td>-0.079516</td>
      <td>-0.016540</td>
      <td>-0.013671</td>
      <td>0.029925</td>
      <td>0.019939</td>
      <td>0.012983</td>
      <td>0.008551</td>
      <td>-0.080187</td>
      <td>-0.033431</td>
      <td>...</td>
      <td>0.042374</td>
      <td>0.012550</td>
      <td>-0.021070</td>
      <td>-0.028898</td>
      <td>0.041150</td>
      <td>-0.040165</td>
      <td>-0.015725</td>
      <td>-0.064446</td>
      <td>-0.043480</td>
      <td>-0.038069</td>
      <td>0.054859</td>
      <td>0.071981</td>
      <td>0.059431</td>
      <td>-0.059622</td>
      <td>0.057350</td>
      <td>-0.028784</td>
      <td>-0.012592</td>
      <td>0.047343</td>
      <td>-0.042691</td>
      <td>-0.018448</td>
      <td>-0.047661</td>
      <td>0.018976</td>
      <td>-0.020382</td>
      <td>-0.007089</td>
      <td>0.055725</td>
      <td>-0.066460</td>
      <td>0.044143</td>
      <td>-0.032896</td>
      <td>-0.035257</td>
      <td>0.045124</td>
      <td>-0.077788</td>
      <td>0.009261</td>
      <td>0.051502</td>
      <td>-0.002606</td>
      <td>-0.037444</td>
      <td>0.028699</td>
      <td>0.008687</td>
      <td>0.048924</td>
      <td>-0.060097</td>
      <td>0.011616</td>
      <td>-0.043432</td>
      <td>-0.057813</td>
      <td>0.023498</td>
      <td>0.029007</td>
      <td>-0.057199</td>
      <td>0.033862</td>
      <td>0.034509</td>
      <td>-0.051691</td>
      <td>-0.068487</td>
      <td>human</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.004159</td>
      <td>0.037407</td>
      <td>0.010676</td>
      <td>0.051571</td>
      <td>-0.082445</td>
      <td>0.042067</td>
      <td>0.085635</td>
      <td>-0.068073</td>
      <td>0.008043</td>
      <td>-0.057601</td>
      <td>-0.010396</td>
      <td>0.061897</td>
      <td>0.026388</td>
      <td>-0.039207</td>
      <td>-0.081150</td>
      <td>0.062177</td>
      <td>-0.025542</td>
      <td>-0.005056</td>
      <td>-0.055322</td>
      <td>0.058100</td>
      <td>0.018608</td>
      <td>-0.029878</td>
      <td>-0.078493</td>
      <td>0.080863</td>
      <td>-0.065654</td>
      <td>0.068974</td>
      <td>0.002821</td>
      <td>-0.040240</td>
      <td>0.057693</td>
      <td>-0.048370</td>
      <td>0.068685</td>
      <td>-0.000077</td>
      <td>-0.000213</td>
      <td>0.063530</td>
      <td>-0.031372</td>
      <td>0.023782</td>
      <td>-0.004415</td>
      <td>-0.041821</td>
      <td>0.007741</td>
      <td>0.023710</td>
      <td>0.070109</td>
      <td>0.032692</td>
      <td>0.002921</td>
      <td>0.077430</td>
      <td>-0.029552</td>
      <td>0.048876</td>
      <td>0.016704</td>
      <td>0.049636</td>
      <td>-0.067951</td>
      <td>-0.050462</td>
      <td>...</td>
      <td>0.040322</td>
      <td>0.027861</td>
      <td>-0.008080</td>
      <td>0.063199</td>
      <td>0.044135</td>
      <td>-0.052681</td>
      <td>0.019565</td>
      <td>0.064226</td>
      <td>-0.048641</td>
      <td>0.044495</td>
      <td>-0.069249</td>
      <td>0.004205</td>
      <td>0.056537</td>
      <td>-0.052674</td>
      <td>0.081130</td>
      <td>0.020684</td>
      <td>0.035940</td>
      <td>-0.068895</td>
      <td>-0.037510</td>
      <td>-0.047443</td>
      <td>-0.072809</td>
      <td>-0.046582</td>
      <td>0.025906</td>
      <td>0.014971</td>
      <td>0.067606</td>
      <td>-0.063587</td>
      <td>-0.019536</td>
      <td>-0.028843</td>
      <td>0.056103</td>
      <td>0.070734</td>
      <td>-0.072800</td>
      <td>0.061065</td>
      <td>0.006131</td>
      <td>0.008708</td>
      <td>-0.019813</td>
      <td>-0.031101</td>
      <td>0.063472</td>
      <td>0.022468</td>
      <td>-0.003953</td>
      <td>0.034237</td>
      <td>-0.014825</td>
      <td>0.083977</td>
      <td>-0.041056</td>
      <td>0.074752</td>
      <td>-0.062685</td>
      <td>0.030684</td>
      <td>-0.066145</td>
      <td>-0.046920</td>
      <td>-0.046959</td>
      <td>human</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.018760</td>
      <td>0.017700</td>
      <td>-0.001465</td>
      <td>0.026475</td>
      <td>-0.065444</td>
      <td>0.055917</td>
      <td>0.083909</td>
      <td>-0.037637</td>
      <td>-0.067772</td>
      <td>0.003231</td>
      <td>0.060457</td>
      <td>0.075828</td>
      <td>-0.035041</td>
      <td>0.030025</td>
      <td>-0.081898</td>
      <td>0.017239</td>
      <td>-0.062100</td>
      <td>0.069808</td>
      <td>0.033860</td>
      <td>-0.009992</td>
      <td>0.061071</td>
      <td>-0.034434</td>
      <td>-0.062524</td>
      <td>0.046930</td>
      <td>-0.012984</td>
      <td>-0.020368</td>
      <td>-0.077449</td>
      <td>-0.030762</td>
      <td>0.012479</td>
      <td>0.034478</td>
      <td>0.067442</td>
      <td>-0.028413</td>
      <td>-0.029516</td>
      <td>0.003395</td>
      <td>-0.059197</td>
      <td>0.062458</td>
      <td>0.073726</td>
      <td>-0.021924</td>
      <td>-0.060282</td>
      <td>-0.040060</td>
      <td>0.015813</td>
      <td>0.018026</td>
      <td>0.025038</td>
      <td>0.071120</td>
      <td>-0.024542</td>
      <td>0.010045</td>
      <td>-0.029734</td>
      <td>0.045184</td>
      <td>-0.013265</td>
      <td>-0.070352</td>
      <td>...</td>
      <td>-0.028412</td>
      <td>-0.017237</td>
      <td>0.011519</td>
      <td>0.030776</td>
      <td>0.001742</td>
      <td>-0.027445</td>
      <td>0.077707</td>
      <td>0.045478</td>
      <td>-0.061658</td>
      <td>-0.033477</td>
      <td>0.001959</td>
      <td>0.028779</td>
      <td>0.007566</td>
      <td>-0.056936</td>
      <td>0.026699</td>
      <td>-0.032550</td>
      <td>-0.059910</td>
      <td>-0.080014</td>
      <td>0.006620</td>
      <td>0.053456</td>
      <td>-0.069313</td>
      <td>-0.069715</td>
      <td>0.013956</td>
      <td>-0.007759</td>
      <td>0.058094</td>
      <td>-0.011070</td>
      <td>0.051757</td>
      <td>0.006790</td>
      <td>0.052903</td>
      <td>0.005722</td>
      <td>-0.009652</td>
      <td>0.057129</td>
      <td>0.016612</td>
      <td>-0.002932</td>
      <td>-0.079768</td>
      <td>0.062508</td>
      <td>0.038955</td>
      <td>0.049365</td>
      <td>-0.052527</td>
      <td>-0.017487</td>
      <td>-0.031634</td>
      <td>0.030714</td>
      <td>-0.059589</td>
      <td>0.066754</td>
      <td>-0.033562</td>
      <td>0.026823</td>
      <td>0.054268</td>
      <td>-0.023460</td>
      <td>-0.019951</td>
      <td>human</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.061679</td>
      <td>0.001478</td>
      <td>0.031168</td>
      <td>0.022693</td>
      <td>-0.015502</td>
      <td>0.021912</td>
      <td>-0.052999</td>
      <td>0.000949</td>
      <td>-0.068597</td>
      <td>0.030209</td>
      <td>0.008987</td>
      <td>-0.029433</td>
      <td>0.011550</td>
      <td>-0.020340</td>
      <td>-0.031608</td>
      <td>0.073531</td>
      <td>-0.025333</td>
      <td>-0.004687</td>
      <td>-0.054933</td>
      <td>-0.007605</td>
      <td>-0.055760</td>
      <td>-0.024632</td>
      <td>-0.059326</td>
      <td>0.062436</td>
      <td>0.032248</td>
      <td>0.062266</td>
      <td>-0.002961</td>
      <td>-0.088176</td>
      <td>0.031939</td>
      <td>-0.036887</td>
      <td>0.005734</td>
      <td>-0.023288</td>
      <td>0.081136</td>
      <td>-0.004106</td>
      <td>-0.033513</td>
      <td>-0.009084</td>
      <td>0.030620</td>
      <td>0.043523</td>
      <td>0.053875</td>
      <td>-0.028582</td>
      <td>0.023576</td>
      <td>0.021544</td>
      <td>0.060563</td>
      <td>0.007828</td>
      <td>0.023314</td>
      <td>-0.045387</td>
      <td>-0.053455</td>
      <td>0.035316</td>
      <td>0.054424</td>
      <td>0.004902</td>
      <td>...</td>
      <td>0.014024</td>
      <td>0.060987</td>
      <td>0.059613</td>
      <td>0.057116</td>
      <td>-0.012657</td>
      <td>0.040048</td>
      <td>0.064219</td>
      <td>0.037523</td>
      <td>0.028764</td>
      <td>-0.028689</td>
      <td>0.088758</td>
      <td>-0.001394</td>
      <td>0.075421</td>
      <td>-0.087056</td>
      <td>0.040511</td>
      <td>0.039444</td>
      <td>-0.060023</td>
      <td>-0.029292</td>
      <td>-0.051675</td>
      <td>0.021379</td>
      <td>-0.079976</td>
      <td>0.030407</td>
      <td>-0.033485</td>
      <td>0.000740</td>
      <td>-0.028985</td>
      <td>-0.073933</td>
      <td>-0.022529</td>
      <td>0.036499</td>
      <td>0.049380</td>
      <td>0.071529</td>
      <td>0.007678</td>
      <td>0.055372</td>
      <td>0.020072</td>
      <td>0.001890</td>
      <td>0.043390</td>
      <td>0.015330</td>
      <td>-0.011553</td>
      <td>0.066848</td>
      <td>0.003082</td>
      <td>-0.007621</td>
      <td>-0.024629</td>
      <td>0.038945</td>
      <td>-0.084256</td>
      <td>-0.020659</td>
      <td>-0.046709</td>
      <td>0.087554</td>
      <td>0.070257</td>
      <td>-0.096228</td>
      <td>0.002832</td>
      <td>human</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.046772</td>
      <td>0.025765</td>
      <td>0.018090</td>
      <td>-0.011639</td>
      <td>-0.080991</td>
      <td>-0.020765</td>
      <td>0.075177</td>
      <td>-0.056908</td>
      <td>0.006270</td>
      <td>-0.006358</td>
      <td>0.063727</td>
      <td>0.061065</td>
      <td>-0.039870</td>
      <td>0.057956</td>
      <td>0.046617</td>
      <td>0.034687</td>
      <td>-0.024982</td>
      <td>-0.011706</td>
      <td>0.037853</td>
      <td>0.041801</td>
      <td>0.043793</td>
      <td>-0.026953</td>
      <td>-0.047420</td>
      <td>0.009063</td>
      <td>-0.067187</td>
      <td>0.021363</td>
      <td>-0.057976</td>
      <td>-0.018172</td>
      <td>0.052888</td>
      <td>-0.005593</td>
      <td>0.073312</td>
      <td>0.066153</td>
      <td>0.057724</td>
      <td>-0.018811</td>
      <td>0.027091</td>
      <td>-0.039104</td>
      <td>-0.037468</td>
      <td>-0.049329</td>
      <td>0.046349</td>
      <td>-0.044057</td>
      <td>0.016623</td>
      <td>0.005787</td>
      <td>0.011761</td>
      <td>0.036444</td>
      <td>-0.048685</td>
      <td>-0.005705</td>
      <td>0.039632</td>
      <td>0.002725</td>
      <td>-0.009126</td>
      <td>0.029212</td>
      <td>...</td>
      <td>0.062085</td>
      <td>0.034350</td>
      <td>0.026678</td>
      <td>0.069446</td>
      <td>-0.046912</td>
      <td>-0.040991</td>
      <td>-0.004489</td>
      <td>0.060533</td>
      <td>-0.067664</td>
      <td>-0.064948</td>
      <td>0.079440</td>
      <td>0.050857</td>
      <td>-0.013103</td>
      <td>-0.070723</td>
      <td>0.065417</td>
      <td>-0.059853</td>
      <td>-0.023149</td>
      <td>0.061218</td>
      <td>-0.066777</td>
      <td>0.029265</td>
      <td>-0.078627</td>
      <td>-0.026768</td>
      <td>0.059218</td>
      <td>0.033352</td>
      <td>-0.010272</td>
      <td>-0.056246</td>
      <td>-0.026515</td>
      <td>-0.054604</td>
      <td>0.047835</td>
      <td>0.002972</td>
      <td>0.072556</td>
      <td>0.026290</td>
      <td>0.051406</td>
      <td>0.065368</td>
      <td>-0.066754</td>
      <td>-0.068543</td>
      <td>0.072629</td>
      <td>0.051849</td>
      <td>-0.035418</td>
      <td>-0.037926</td>
      <td>-0.032357</td>
      <td>0.079559</td>
      <td>-0.042057</td>
      <td>-0.025956</td>
      <td>-0.064477</td>
      <td>0.039704</td>
      <td>0.052853</td>
      <td>-0.071350</td>
      <td>0.026886</td>
      <td>human</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 513 columns</p>
</div>





```python
label_c = len(ds.label.unique())
```




```python
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(ds.label)
encoded_Y = encoder.transform(ds.label)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
X = ds.drop('label',axis=1)

encoder.classes_
```





    array(['bot', 'human'], dtype=object)





```python
import pickle
def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
save_object(encoder,"encoder.pkl")
```


## NN-Architecture for Multi-Class classification  and Training 



```python
model = Sequential()
model.add(Dense(50, activation='relu', input_dim=512))
model.add(Dense(25, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(label_c, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X,dummy_y, epochs=15, batch_size=64,validation_split=0.15,verbose=2,shuffle=True)
```


    Train on 5241 samples, validate on 926 samples
    Epoch 1/15
     - 4s - loss: 0.3245 - acc: 0.9454 - val_loss: 0.0410 - val_acc: 0.9838
    Epoch 2/15
     - 0s - loss: 0.0356 - acc: 0.9901 - val_loss: 0.0433 - val_acc: 0.9773
    Epoch 3/15
     - 0s - loss: 0.0237 - acc: 0.9920 - val_loss: 0.0218 - val_acc: 0.9924
    Epoch 4/15
     - 0s - loss: 0.0200 - acc: 0.9927 - val_loss: 0.0251 - val_acc: 0.9903
    Epoch 5/15
     - 0s - loss: 0.0141 - acc: 0.9960 - val_loss: 0.0128 - val_acc: 0.9946
    Epoch 6/15
     - 0s - loss: 0.0119 - acc: 0.9966 - val_loss: 0.0274 - val_acc: 0.9881
    Epoch 7/15
     - 0s - loss: 0.0107 - acc: 0.9968 - val_loss: 0.0146 - val_acc: 0.9946
    Epoch 8/15
     - 0s - loss: 0.0078 - acc: 0.9983 - val_loss: 0.0190 - val_acc: 0.9935
    Epoch 9/15
     - 0s - loss: 0.0069 - acc: 0.9983 - val_loss: 0.0251 - val_acc: 0.9914
    Epoch 10/15
     - 0s - loss: 0.0046 - acc: 0.9992 - val_loss: 0.0246 - val_acc: 0.9935
    Epoch 11/15
     - 0s - loss: 0.0047 - acc: 0.9990 - val_loss: 0.0325 - val_acc: 0.9892
    Epoch 12/15
     - 0s - loss: 0.0040 - acc: 0.9996 - val_loss: 0.0232 - val_acc: 0.9946
    Epoch 13/15
     - 0s - loss: 0.0032 - acc: 0.9992 - val_loss: 0.0226 - val_acc: 0.9935
    Epoch 14/15
     - 0s - loss: 0.0024 - acc: 0.9998 - val_loss: 0.0141 - val_acc: 0.9957
    Epoch 15/15
     - 0s - loss: 0.0015 - acc: 1.0000 - val_loss: 0.0210 - val_acc: 0.9957





    <tensorflow.python.keras.callbacks.History at 0x1c3f5f0908>



### Saving Model



```python
model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
```

