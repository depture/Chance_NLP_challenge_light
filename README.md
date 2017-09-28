# Coding Challenge - Data Scientist - NLP

This repository was created to host the challenge for Data Scientist - NLP roles at
Chance.

## Evaluation

This is a non-exaustive list of the aspects that we will consider:

* Code organization
* Problem solving
* Code readability
* Chosen solutions
* Version control practices


## Problem - MBTI prediction

On this task you will have to create a model to predict MBTI personality types
from posts. If you don't know what MBTI is, see [MBTI basics]( 
http://www.myersbriggs.org/my-mbti-personality-type/mbti-basics/home.htm?bhcp=1)

### Details

For this task you can use any language of your choice, but we recommend the use
of some of the following: python, jupyter, scikit-learn. You will have
to :
* Create a local git repository
* Download the dataset for this task [here](
    https://www.kaggle.com/datasnaek/mbti-type)
* Create an NLP machine learning algorithm that determines a person personality type based on a set of posts. You can use the model of your preference.
* Train your model, expose it through an API (function), and describe
    how to access it.

## Usage

1. Start a git repository with ```git init```
1. Do your magic on your local machine, trying to commit often
1. Add at the end of this README a small descriptions about your choices.
1. Run ```git bundle create YOURNAME.bundle HEAD ```
1. Send the generated file by email to tech-interview@chance.co

## Description 

This is a simple model which trains relatively quickly. The cross validation output an f1-score of ~64. It is possible to search for hyper paramaters with bayesian optimisation to improve accuracy. Also, a pre-trained word embeddings like glove in a neural model may bring more insight. A lot of the links are dead links but it would be interesting to scrap the different urls to extract more information. 

An ipython notebook describe step by step the choice i took to implement the algorithm.
Ii is implemented in Python 3.6.

The main steps are :

1. Treat data
* Load data using pandas
* Split each row by type | comments 
* Cleaning each comments by :
    * removing all urls replaced by dummy word 'link'
    * removing everything except letters
    * removing larger spaces and lowering
    * removing stopwords and lemmatizing
 * Labelizing each mbti personality

2. Learning phase
* Run tf-idf and count vectorization of words vectorization (the final model trains only with count vector)
* Try pca, fast\_ica (no improvements, if more data investigate feature reduction to improve speed)
* Try Multinomial Naive Bayes (F1-score ~ 58) [5 fold stratified cross validation]
* Try XGboost (F1-score ~ 64) [5 fold stratified cross validation]
* Save vectorization and models parameters for later usage

3. Application
* The function MBTI\_XGB loads precalculated vectorizer and boosting model to train new data

TO DO:
- Scrape urls and extract topic of pages
- Balance the dataset with resampling techniques
- Try to ouput the probabilities for a letter instead of a personality (multioutput) and merge the 4 highest
- Hyper Parameter Optimization
- Convolutional Network with glove embedding
- Search multiple classifiers (not correlated) and stack them and build pipeline with Lasso.

The notebook contains a multi-output classifier based on 4 adaboost models. As the letters are only paired by two, the problem is to predict 4 binary variables. The cross validation gives the following table.

Attribute| I-E  | N-S  | F-T  | J-P  |   
|:-:|---|---|---|---|
F1-Score|0.59 | 0.44 | 0.8 |  0.83| 

This gives us insight about what are the axes that the algorithm may fail at identifying (S Sensation vs N Intuition).

Finally there are two neural networks, (1D convnet and LSTM) but I did not have the time to tune them.


