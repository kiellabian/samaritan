import dataset
from pos_tagger import tag_text

import matplotlib.pyplot as plt

from classifier import Classifier

from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer

n_gram = 4

feature_set = dataset.get_data(number_of_sentences=100, n_gram=n_gram)

# c = SklearnClassifier(Pipeline([('clf', MultinomialNB())]))
# c = SklearnClassifier(Pipeline([('clf', LinearSVC(C=2))]))
# c = SklearnClassifier(Pipeline([('idf', TfidfTransformer()), ('clf', LinearSVC(C=2))]))

# classifier = Classifier(c, feature_set)
# classifier.plot_performance()
# classifier.plot_performnace_ngram(limit=6)
# classifier.train()
# classifier.test()

i = 0.2
accuracies = []
fscores = []
cs = []
while i <= 5:
    c = SklearnClassifier(Pipeline([('clf', LinearSVC(C=i))]))
    classifier = Classifier(c, feature_set)
    classifier.train()
    accuracy, fscore = classifier.test()
    accuracies.append(accuracy)
    fscores.append(fscore)
    cs.append(i)
    i += 0.2
    print i

plt.plot(cs, accuracies, label='Accuracy', linewidth=2)
plt.plot(cs, fscores, label='F1-score', linewidth=2)
plt.xlabel('C')
plt.legend(loc='lower right')
plt.show()

t = 'a'
while t != '':
    t = raw_input('>')
    if t:
        tags = tag_text(t)
        features = dataset.__convern_to_count_dictionary(tags, n_gram=n)
        classifier.predict(features)
