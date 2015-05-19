from operator import itemgetter
import random

import matplotlib.pyplot as plt

from nltk import classify

from sklearn import cross_validation
from sklearn.metrics import classification_report, f1_score

import dataset


class Classifier:
    def __init__(self, classifier, feature_set):
        size = int(len(feature_set) * 0.8)
        random.shuffle(feature_set)
        self.classifier = classifier
        self.feature_set = feature_set
        self.train_set = feature_set[:size]
        self.test_set = feature_set[size:]

    def predict(self, features):
        prediction = self.classifier.classify(features)
        print prediction
        return prediction

    def train(self, cross_val=False, train_set=None):
        if not train_set:
            train_set = self.train_set
        if cross_val:
            self.classifier = self.cross_validate()['classifier']
        else:
            self.classifier = self.classifier.train(train_set)

    def test(self, test_set=None):
        if not test_set:
            test_set = self.test_set
        accuracy = classify.accuracy(self.classifier, test_set)
        print 'Accuracy: ' + str(accuracy)

        features = []
        labels = []
        for (feat, label) in test_set:
            features.append(feat)
            labels.append(label)
        predictions = self.classifier.classify_many(features)

        print classification_report(predictions, labels)
        fscore = f1_score(predictions, labels)
        return accuracy, fscore

    def cross_validate(self, n_folds=2):
        best_accuracy = 0.0
        best_train_accuracy = 0.0
        k_fold = cross_validation.KFold(len(self.train_set), n_folds=n_folds)
        for train_indices, test_indices in k_fold:
            train = itemgetter(*train_indices)(self.train_set)
            cv = itemgetter(*test_indices)(self.train_set)

            classifier = self.classifier.train(train)
            if len(test_indices) == 1:
                cv = (cv,)
            accuracy = classify.accuracy(classifier, cv)
            if accuracy > best_accuracy:
                best_classifier = classifier
                best_accuracy = accuracy
        return {'classifier': best_classifier, 'accuracy': accuracy}

    def plot_curves(self, start_size, inc_size):
        data_size = start_size
        data_sizes = []
        accuracies = []
        fscores = []
        while data_size <= len(self.feature_set):
            curr_data_set = self.feature_set[:data_size]
            curr_size = int(len(curr_data_set) * 0.8)
            train_set = curr_data_set[:curr_size]
            test_set = curr_data_set[curr_size:]
            
            self.train(False, train_set)
            accuracy, fscore = self.test(test_set)

            accuracies.append(accuracy)
            fscores.append(fscore)
            data_sizes.append(data_size)
            data_size += inc_size

        plt.plot(data_sizes, accuracies, label='Accuracy', linewidth=3)
        plt.plot(data_sizes, fscores, label='F1-score', linewidth=3)
        plt.title('Learning Curves')
        plt.xlabel('Dataset Size')
        plt.legend(loc='lower right')
        plt.show()

    def plot_performance(self, num_of_authors=10):
        accuracies = []
        fscores = []
        for authors_ctr in range(3, num_of_authors+1):
            self.feature_set = dataset.get_data(
                number_of_sentences=100,
                number_of_authors=authors_ctr,
                n_gram=4
            )
            random.shuffle(self.feature_set)
            curr_size = int(len(self.feature_set) * 0.8)
            self.train_set = self.feature_set[:curr_size]
            self.test_set = self.feature_set[curr_size:]

            self.train()
            accuracy, fscore = self.test()
            accuracies.append(accuracy)
            fscores.append(fscore)

        plt.plot(range(3, num_of_authors + 1), accuracies, label='Accuracy', linewidth=2)
        plt.plot(range(3, num_of_authors + 1), fscores, label='F1-score', linewidth=2)
        plt.xlabel('number of authors')
        plt.legend(loc='lower right')
        plt.show()

    def plot_performnace_ngram(self, limit=6):
        accuracies = []
        fscores = []
        for ngram in range(1, limit + 1):
            self.feature_set = dataset.get_data(
                number_of_sentences=100,
                n_gram=ngram
            )
            random.shuffle(self.feature_set)
            curr_size = int(len(self.feature_set) * 0.8)
            self.train_set = self.feature_set[:curr_size]
            self.test_set = self.feature_set[curr_size:]
    
            self.train()
            accuracy, fscore = self.test()
            accuracies.append(accuracy)
            fscores.append(fscore)

        plt.plot(range(1, limit + 1), accuracies, label='Accuracy', linewidth=2)
        plt.plot(range(1, limit + 1), fscores, label='F1-score', linewidth=2)
        plt.xlabel('n-gram')
        plt.legend(loc='lower right')
        plt.show()
