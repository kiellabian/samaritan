from operator import itemgetter
import random

from nltk import classify

from sklearn import cross_validation
from sklearn.metrics import classification_report

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

    def test(self):
        accuracy = classify.accuracy(self.classifier, self.test_set)
        print 'Accuracy: ' + str(accuracy)

        features = []
        labels = []
        for (feat, label) in self.feature_set:
            features.append(feat)
            labels.append(label)
        predictions = self.classifier.classify_many(features)
        print classification_report(predictions, labels)

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