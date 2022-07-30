# Naive Bayes Model for Sentiment Analysis
# * Training a naive bayes model on a sentiment analysis task
# * Testing the model
# * Computing ratio of positive words to negative words
# * Predicting user input tweet

# Importing packages

from utils import process_tweet, lookup
import pdb
from nltk.corpus import stopwords, twitter_samples
import numpy as np
import pandas as pd
import nltk
import string
from nltk.tokenize import TweetTokenizer
from os import getcwd

nltk.download('twitter_samples')
nltk.download('stopwords')
filePath = f"{getcwd()}/../tmp2/"
nltk.data.path.append(filePath)

# get the sets of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# split the data into two pieces, one for training and one for testing (validation set)
test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

# storing labels of each example for training
train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))


# Processing the data

# takes a list of tweets as input, cleans all of them, and returns a dictionary
# * the key in the dictionary is a tuple containing the stemmed word and its class label, e.g. ("happi",1).
# * the value the number of times this word appears in the given collection of tweets (an integer).
def count_tweets(result, tweets, ys):
    '''
    Input:
        result: a dictionary that will be used to map each pair to its frequency
        tweets: a list of tweets
        ys: a list corresponding to the sentiment of each tweet (either 0 or 1)
    Output:
        result: a dictionary mapping each pair to its frequency
    '''
    for y, tweet in zip(ys, tweets):
        for word in process_tweet(tweet):
            # defining the key, which is the word and label tuple
            pair = (word, y)
            
            # if the key exists in the dictionary, increment the count
            if pair in result:
                result[pair] += 1

            # else, if the key is new, add it to the dictionary and set the count to 1
            else:
                result[pair] = 1

    return result

# Training the model using Naive Bayes

# creating `freqs` dictionary
# * the key is the tuple (word, label)
# * The value is the number of times it has appeared.
freqs = count_tweets({}, train_x, train_y)

# implementing naive baiyes classifier
def train_naive_bayes(freqs, train_x, train_y):
    '''
    Input:
        freqs: dictionary from (word, label) to how often the word appears
        train_x: a list of tweets
        train_y: a list of labels correponding to the tweets (0,1)
    Output:
        logprior: the log prior
        loglikelihood: the log likelihood of Naive bayes equation
    '''
    loglikelihood = {}
    logprior = 0

    # calculating V, the number of unique words in the vocabulary
    vocab = set(pair[0] for pair in freqs.keys())
    V = len(vocab)  
    
    # calculating N_pos, N_neg, V_pos, V_neg
    N_pos = N_neg = 0
    for pair in freqs.keys():
        # if the label is positive (greater than zero)
        if pair[1] > 0:

            # incrementing the number of positive words by the count for this (word, label) pair
            N_pos += freqs[pair]

        # else, the label is negative
        else:

            # incrementing the number of negative words by the count for this (word,label) pair
            N_neg += freqs[pair]
    
    # calculating D, the number of documents
    D = len(train_y)

    # calculating D_pos, the number of positive documents
    D_pos = 0
    # calculating D_neg, the number of negative documents
    D_neg = 0
    for label in train_y:
        if label == 1:
            D_pos += 1
        else:
            D_neg += 1

    # calculating logprior
    logprior = np.log(D_pos) - np.log(D_neg)
    
    # for each word in the vocabulary...
    for word in vocab:
        
        # get the positive and negative frequency of the word
        freq_pos = lookup(freqs, word, 1.0)
        freq_neg = lookup(freqs, word, 0.0)
        
        # calculate the probability that each word is positive, and negative
        p_w_pos = (freq_pos + 1) / (N_pos + V)
        p_w_neg = (freq_neg + 1) / (N_neg + V)

        # calculate the log likelihood of the word
        loglikelihood[word] = np.log(p_w_pos) - np.log(p_w_neg)

    return logprior, loglikelihood


logprior, loglikelihood = train_naive_bayes(freqs, train_x, train_y)

# Testing the model

# using the naive bayes model to predict sentiment of a tweet
# * the function returns the probability that the tweet belongs to the positive or negative class.
# * for each tweet, it sums up loglikelihoods of each word in the tweet.
# * also adds the logprior to this sum to get the predicted sentiment of that tweet.
def naive_bayes_predict(tweet, logprior, loglikelihood):
    '''
    Input:
        tweet: a string
        logprior: a number
        loglikelihood: a dictionary of words mapping to numbers
    Output:
        p: the sum of all the logliklihoods of each word in the tweet (if found in the dictionary) + logprior (a number)

    '''
    # processing the tweet to get a list of words
    word_l = process_tweet(tweet)

    # initializing probability to zero
    p = 0

    # adding the logprior
    p += logprior

    for word in word_l:

        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            # add the log likelihood of that word to the probability
            p += loglikelihood[word]

    return p


# check the accuracy of naive bayels model predictions.
def test_naive_bayes(test_x, test_y, logprior, loglikelihood, naive_bayes_predict=naive_bayes_predict):
    """
    Input:
        test_x: A list of tweets
        test_y: the corresponding labels for the list of tweets
        logprior: the logprior
        loglikelihood: a dictionary with the loglikelihoods for each word
    Output:
        accuracy: (# of tweets classified correctly)/(total # of tweets)
    """
    accuracy = 0 

    y_hats = []
    for tweet in test_x:
        # if the prediction is > 0
        if naive_bayes_predict(tweet, logprior, loglikelihood) > 0:
            # the predicted class is 1
            y_hat_i = 1.0
        else:
            # otherwise the predicted class is 0
            y_hat_i = 0.0

        # appending the predicted class to the list y_hats
        y_hats.append(y_hat_i)

    # error is the average of the absolute values of the differences between y_hats and test_y
    total_diff = 0
    for i in range(len(y_hats)):
        difference = abs(y_hats[i] - test_y[i])
        total_diff += difference
    error = total_diff / len(y_hats)

    # accuracy is 1 minus the error
    accuracy = 1 - error

    return accuracy

print("Naive Bayes accuracy = %0.4f" %
      (test_naive_bayes(test_x, test_y, logprior, loglikelihood)))

"""user_tweet = input("Please input any tweet to see if it has positive or negative sentiment. ")
score = naive_bayes_predict(user_tweet, logprior, loglikelihood)
print("The sentiment score of your tweet is ", score)
if score > 0:
    print("The sentiment is positive.")
elif score < 0:
    print("The sentiment is negative.")
else:
    print("Could not predict sentiment.")"""

# Filtering words by ratio of positive to negative counts

# gets ratio of positive divided by negative counts
# * given the freqs dictionary of words and a particular word, we use `lookup(freqs,word,1)` to get the positive count of the word
# * similarly, we use the `lookup` function to get the negative count of that word.
def get_ratio(freqs, word):
    '''
    Input:
        freqs: dictionary containing the words

    Output: a dictionary with keys 'positive', 'negative', and 'ratio'.
        Example: {'positive': 10, 'negative': 20, 'ratio': 0.5}
    '''
    pos_neg_ratio = {'positive': 0, 'negative': 0, 'ratio': 0.0}
    
    # using lookup() to find positive counts for the word (denoted by the integer 1)
    pos_neg_ratio['positive'] = lookup(freqs, word, 1.0)
    
    # using lookup() to find negative counts for the word (denoted by integer 0)
    pos_neg_ratio['negative'] = lookup(freqs, word, 0.0)
    
    # calculating the ratio of positive to negative counts for the word
    pos_neg_ratio['ratio'] = (lookup(freqs, word, 1.0) + 1) / (lookup(freqs, word, 0.0) + 1)
    
    return pos_neg_ratio


# getting words if they exceed ratio threshold 
# * if we set the label to 1, then we look for all words whose ratio is equal to or greater than the threshold
# * if we set the label to 0, then we look for all words whose ratio is equal to or lesser than the threshold
# * return new dictionary where the key is the word, and the value is the dictionary `pos_neg_ratio` that is returned by the `get_ratio` function.
def get_words_by_threshold(freqs, label, threshold, get_ratio=get_ratio):
    '''
    Input:
        freqs: dictionary of words
        label: 1 for positive, 0 for negative
        threshold: ratio that will be used as the cutoff for including a word in the returned dictionary
    Output:
        word_list: dictionary containing the word and information on its positive count, negative count, and ratio of positive to negative counts.
        example of a key value pair:
        {'happi':
            {'positive': 10, 'negative': 20, 'ratio': 0.5}
        }
    '''
    word_list = {}

    for key in freqs.keys():
        word, _ = key

        # getting the positive/negative ratio for a word
        pos_neg_ratio = get_ratio(freqs, word)
        
        # if the label is 1 and the ratio is greater than or equal to the threshold...
        if label == 1 and pos_neg_ratio['ratio'] >= threshold:
        
            # add the pos_neg_ratio to the dictionary
            word_list[word] = pos_neg_ratio

        # if the label is 0 and the pos_neg_ratio is less than or equal to the threshold...
        elif label == 0 and pos_neg_ratio['ratio'] <= threshold:
        
            # add the pos_neg_ratio to the dictionary
            word_list[word] = pos_neg_ratio

        # otherwise, do not include this word in the list (do nothing)

    return word_list


user_threshold = float(input("Please input a threshold for words to extract from training data of Naive Bayes classifier. "))
user_label = int(input("Please input 1 for words with positive sentiment or 0 for words with negative sentiment. "))
word_l = get_words_by_threshold(freqs, user_label, user_threshold)
print("Your list of words is: ")
words = []
for key in word_l:
    words.append(key)
print(words)