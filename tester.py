import dataset
from pos_tagger import tag_text

from classifier import Classifier

from sklearn.svm import LinearSVC, SVC, NuSVC

from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer


feature_set = dataset.get_data(number_of_sentences=200)

c = SklearnClassifier(Pipeline([('tfidf', TfidfTransformer()),
                      ('clf', LinearSVC(C=2))]))

classifier = Classifier(c, feature_set)
classifier.train()
classifier.test()

t = 'a'
while t != '':
    t = raw_input('>')
    if t:
        tags = tag_text(t)
        features = dataset.__convern_to_count_dictionary(tags)
        classifier.predict(features)
