from data_handler import bagging_sampler
import numpy as np
import random


class BaggingClassifier:
    def __init__(self, base_estimator, n_estimator):
        """
        :param base_estimator:
        :param n_estimator:
        :return:
        """
        # todo: implement
        self.base_estimator = base_estimator
        self.n_estimator = n_estimator

    # def bagging_sampler(self, X, y, sample_size):
    #     """
    #     Randomly sample with replacement
    #     Size of sample will be same as input data
    #     :param X:
    #     :param y:
    #     :return:
    #     """
    #     #concatenate X and y
    #     data = np.concatenate((X, y), axis=1)

    #     #shuffle data
    #     random.seed(10)
    #     np.random.shuffle(data)

    #     #get number of rows in sample
    #     sample_rows = int(sample_size * data.shape[0])

    #     #split data into sample
    #     sample = data[:sample_rows, :]

    #     #get X and y for sample
    #     X_sample = sample[:, :-1]
    #     y_sample = sample[:, -1]

    #     y_sample = y_sample.reshape(y_sample.shape[0], 1)

    #     return X_sample, y_sample

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        # todo: implement

        #initialize list of classifiers
        self.classifiers = []

        #create n_estimator classifiers
        for i in range(self.n_estimator):
            #create new classifier
            classifier = self.base_estimator

            #train classifier on bagged data
            X, y = bagging_sampler(X, y)

            #fit classifier to data
            classifier.fit(X, y)

            #add classifier to list
            self.classifiers.append(classifier)

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        apply majority voting
        :param X:
        :return:
        """

        # todo: implement

        #initialize list of predictions
        predictions = []

        #predict for each classifier
        for classifier in self.classifiers:
            #predict
            prediction = classifier.predict(X)

            #add prediction to list
            predictions.append(prediction)

        
        #convert predictions to numpy array
        predictions = np.array(predictions)

        #get majority vote
        prediction = np.mean(predictions, axis=0)

        #convert to binary
        prediction = np.where(prediction >= 0.5, 1, 0)

        return prediction
