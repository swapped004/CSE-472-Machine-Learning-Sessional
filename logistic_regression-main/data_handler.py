import pandas as pd
import numpy as np
import random


def load_dataset():
    """
    function for reading data from csv
    and processing to return a 2D feature matrix and a vector of class
    :return:
    """
    # todo: implement

    # read data from csv
    df = pd.read_csv('data_banknote_authentication.csv')

    # get feature matrix and class vector
    X = df.iloc[:, df.columns != 'isoriginal'].values
    y = df.iloc[:, df.columns == 'isoriginal'].values
    return X,y


def split_dataset(X, y, test_size, shuffle):
    """
    function for spliting dataset into train and test
    :param X:
    :param y:
    :param float test_size: the proportion of the dataset to include in the test split
    :param bool shuffle: whether to shuffle the data before splitting
    :return:
    """
    
    #split data into train and test without using sklearn

    #concatenate X and y
    data = np.concatenate((X, y), axis=1)

    if shuffle:
        #shuffle data
        random.seed(10)
        np.random.shuffle(data)

    #get number of rows in test set
    test_rows = int(test_size * data.shape[0])

    #split data into train and test
    train = data[test_rows:, :]
    test = data[:test_rows, :]

    #get X and y for train and test
    X_train = train[:, :-1]
    y_train = train[:, -1]

    X_test = test[:, :-1]
    y_test = test[:, -1]

    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)

    return X_train, y_train, X_test, y_test


def bagging_sampler(X, y):
    """
    Randomly sample with replacement
    Size of sample will be same as input data
    :param X:
    :param y:
    :return:
    """
    # todo: implement
    #concatenate X and y
    data = np.concatenate((X, y), axis=1)

    #sample with replacement
    sample = np.random.choice(data.shape[0], data.shape[0], replace=True)

    #get X and y for sample
    X_sample = data[sample, :-1]
    y_sample = data[sample, -1]
    y_sample = y_sample.reshape(y_sample.shape[0], 1)
    
    assert X_sample.shape == X.shape
    assert y_sample.shape == y.shape
    return X_sample, y_sample
