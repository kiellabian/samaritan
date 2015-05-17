from nltk import classify

from sklearn.metrics import classification_report

class Classifier:
    def __init__(self, classifier, feature_set):
        self.classifier = classifier
        self.feature_set = feature_set
        size = int(len(feature_set) * 0.8)
        self.train_set = feature_set[:size]
        self.test_set = feature_set[size:]

    def predict(self, features):
        prediction = self.classifier.classify(features)
        print prediction
        return prediction

    def train(self):
        self.classifier = self.classifier.train(self.train_set)

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