# -*_ coding: utf-8 -*-
from nltk import pos_tag, word_tokenize, data
import re


def _tag_sentence(sentence):
    """ pos tags each word in a sentence and return the tags. """
    print sentence
    words = word_tokenize(sentence)
    p = pos_tag(words)
    tags = []
    for x in p:
        if len(x[1]) > 1:
            tags.append(x[1])
    return tags


def tag_text(text, output_path=None):
    """ pos tags the sentences in a string """
    if output_path:
        f = open(output_path, 'a')
    result = []
    tokenizer = data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(text)
    for sentence in sentences:
        tags = _tag_sentence(sentence)
        result.append(tags)
        tags = ' '.join(tags)
        print tags
        if output_path:
            f.write(tags + '\n')
    if output_path:
        f.close()
    return result


def _read(filepath):
    """ read file and return a single string of the content of the file."""
    final_text = ''
    with open(filepath) as f:
        for line in f:
            line = line.strip().lower()
            if len(line) > 0:
                final_text += ' ' + line.decode('utf-8')
    final_text = re.sub('".*?"', '', final_text)
    return final_text


def tag(filepath):
    print filepath
    text = _read(filepath)
    output_path = re.sub('\..*', '.pos', filepath)
    tag_text(text, output_path=output_path)

# print tag('dataset/joseph_conrad__under_western_eyes.txt')
