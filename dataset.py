import re

from os import walk

from nltk.util import ngrams


def __get_file_paths(path='dataset'):
    paths = []
    for (dirpath, dirnames, filenames) in walk(path):
        for name in filenames:
            paths.append([dirpath + '/' + name, name])
    return paths


def __convern_to_count_dictionary(sentences, n_gram=4):
    features = {}
    for sentence in sentences:
        for i in range(1, n_gram + 1):
            for feat in ngrams(sentence, i):
                feat = ' '.join(feat)
                if feat in features:
                    features[feat] += 1
                else:
                    features[feat] = 1
    return features


def get_data(number_of_sentences=50, n_gram=4):
    paths = __get_file_paths('dataset')
    dataset = []
    for path in paths:
        if path[1].find('.pos') == -1:
            continue
        with open(path[0]) as f:
            index = path[1].index('__')
            cls = path[1][:index]
            sentences = []
            line = f.readline()
            count = 0
            while line:
                line = re.split(' +', line)
                sentences.append(line)
                count += 1
                if count == number_of_sentences:
                    features = __convern_to_count_dictionary(sentences, n_gram)
                    dataset.append((features, cls))
                    sentences = []
                    count = 0
                line = f.readline()
            features = __convern_to_count_dictionary(sentences, n_gram)
            dataset.append((features, cls))
    return dataset
    
