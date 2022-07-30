# Logistic Regression for Sentiment Analysis
# Implementing logistic regression for sentiment analysis on tweets.
# Given a tweet, my algorithm will decide if it has a positive sentiment or a negative one.  
# Demonstrates the following skills:
# * Being able to extract features for logistic regression given some text
# * Implementing logistic regression from scratch
# * Applying logistic regression on a natural language processing task

# Importing functions and data

import nltk
from os import getcwd

nltk.download('twitter_samples')
nltk.download('stopwords')

# Importing helper functions provided in utils.py file:
# * process_tweet: cleans the text, tokenizes it into separate words, removes stopwords, and converts words to stems
# * build_freqs: this counts how often a word in the 'corpus' (the entire set of tweets) was associated with a positive label '1' or a negative label '0',
#   then builds the 'freqs' dictionary, where each key is the (word, label) tuple, and the value is the count of its frequency within the corpus of tweets

filePath = f"{getcwd()}/../tmp2/"
nltk.data.path.append(filePath)


import numpy as np
import pandas as pd
from nltk.corpus import twitter_samples 

from utils import process_tweet, build_freqs


# Preparing the data

# select the set of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')


# Train test split: 20% will be in the test set, and 80% in the training set

# split the data into two pieces, one for training and one for testing (validation set) 
test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg 
test_x = test_pos + test_neg

# combine positive and negative labels
train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

# create frequency dictionary
freqs = build_freqs(train_x, train_y)

# sigmoid function that maps the input 'z' to a value that ranges between 0 and 1
def sigmoid(z): 
    '''
    Input:
        z: is the input (can be a scalar or an array)
    Output:
        h: the sigmoid of z
    '''
    
    # calculating the sigmoid of z
    h = (1/(1 + np.exp(-z)))
    
    return h


# Updating the weights

# gradient descent function used to update weights using cost function
def gradientDescent(x, y, theta, alpha, num_iters):
    '''
    Input:
        x: matrix of features which is (m,n+1)
        y: corresponding labels of the input matrix x, dimensions (m,1)
        theta: weight vector of dimension (n+1,1)
        alpha: learning rate
        num_iters: number of iterations you want to train your model for
    Output:
        J: the final cost
        theta: your final weight vector
    Hint: you might want to print the cost to make sure that it is going down.
    '''
    # getting the number of rows in matrix x
    m = x.shape[0]
    
    for i in range(0, num_iters):
        
        # getting the dot product of x and theta
        z = np.dot(x, theta)
        
        # getting the sigmoid of z
        h = sigmoid(z)
        
        # calculating the cost function
        y_t = np.transpose(y)
        one_minus_y_t = np.transpose(1-y)
        first_dot_prod = np.dot(y_t, np.log(h))
        second_dot_prod = np.dot(one_minus_y_t, np.log(1-h))
        J = -1/m * (first_dot_prod + second_dot_prod)

        # updating the weights theta
        x_t = np.transpose(x)
        theta = theta - (alpha/m * np.dot(x_t, (h-y)))
        
    J = float(J)
    return J, theta


# Extracting the features

# extracting two features given a list of tweets
#     * the first feature is the number of positive words in a tweet
#     * the second feature is the number of negative words in a tweet
def extract_features(tweet, freqs, process_tweet=process_tweet):
    '''
    Input: 
        tweet: a list of words for one tweet
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output: 
        x: a feature vector of dimension (1,3)
    '''
    # process_tweet tokenizes, stems, and removes stopwords
    word_l = process_tweet(tweet)
    
    # 3 elements in the form of a 1 x 3 vector
    x = np.zeros((1, 3)) 
    
    # bias term is set to 1
    x[0,0] = 1 
    
    # looping through each word in the list of words
    for word in word_l:
        
        # incrementing the word count for the positive label 1
        if (word, 1.0) in freqs:
            x[0,1] += freqs[(word, 1.0)]
        
        # incrementing the word count for the negative label 0
        if (word, 0.0) in freqs:
            x[0,2] += freqs[(word, 0.0)]
            
    assert(x.shape == (1, 3))
    return x


# Training the model

# collecting the features 'x' and stack them into a matrix 'X'
X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :]= extract_features(train_x[i], freqs)

# training labels corresponding to X
Y = train_y

# applying gradient descent
J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)

# predicts whether a tweet is positive or negative.
def predict_tweet(tweet, freqs, theta):
    '''
    Input: 
        tweet: a string
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        theta: (3,1) vector of weights
    Output: 
        y_pred: the probability of a tweet being positive or negative
    '''
    
    # extracting the features of the tweet and store it into x
    x = extract_features(tweet, freqs)
    
    # making the prediction using x and theta
    y_pred = sigmoid(np.dot(x, theta))
    
    return y_pred



# Checking performance using the test set
# * given the test data and the weights of the trained model, calculating its accuracy
# * using 'predict_tweet' function to make predictions on each tweet in the test set
# * if the prediction is > 0.5, set the model's classification 'y_hat' to 1, otherwise set the model's classification 'y_hat' to 0
# * a prediction is accurate when the y_hat equals the test_y.  summed up all the instances when they are equal and divided by m.


# testing logistic regression model for sentiment analysis
def test_logistic_regression(test_x, test_y, freqs, theta, predict_tweet=predict_tweet):
    """
    Input: 
        test_x: a list of tweets
        test_y: (m, 1) vector with the corresponding labels for the list of tweets
        freqs: a dictionary with the frequency of each pair (or tuple)
        theta: weight vector of dimension (3, 1)
    Output: 
        accuracy: (# of tweets classified correctly) / (total # of tweets)
    """
    
    # total number of tweets
    m = test_y.shape[0]
    
    # total number of tweets classified correctly
    num_correct = 0
    
    # the list for storing predictions
    y_hat = []
    
    
    for tweet in test_x:
        # getting the label prediction for the tweet
        y_pred = predict_tweet(tweet, freqs, theta)
        
        if y_pred > 0.5:
            # appending 1.0 to the list
            y_hat.append(1.0)
        else:
            # appending 0 to the list
            y_hat.append(0.0)
            

    # with the above implementation, y_hat is a list, but test_y is (m,1) array
    # converting both to one-dimensional arrays in order to compare them using the '==' operator
    test_y_list = test_y.tolist()
    for i in range(m):
        if y_hat[i] == test_y_list[i][0]:
            num_correct += 1
            
    a = np.array([[num_correct / m]])
    accuracy =  np.float64(a)
    
    return accuracy

test_acc = test_logistic_regression(test_x, test_y, freqs, theta)
print("After training, the accuracy of the logistic regression model on the test set is ", test_acc)

# Predict with tweet input by user

user_tweet = input("Please input any sentence in order to predict its sentiment: ")
y_hat = predict_tweet(user_tweet, freqs, theta)
print("The calculated sentiment score for your sentence is", y_hat[0][0])
if y_hat > 0.5:
    print("Hence it is a positive sentiment")
else: 
    print("Hence it is a negative sentiment")





