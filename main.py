import argparse
import csv
import itertools
import json
import pathlib
import pickle
from argparse import ArgumentParser
from typing import Iterable

import jieba
from lazypredict.Supervised import LazyClassifier
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Perceptron

import lazyresult


def __train_data_iter():
    with open('./dataset/train.json') as f:
        for item in json.load(f):
            sentiment = item['response']
            yield (0 if sentiment == 'positive' else 1 if sentiment == 'negative' else 2,
                   item['query'])


def __test_data_iter():
    with open('./dataset/test.csv', 'rt') as fd:
        for row in csv.reader(fd):
            sentiment = row[-1]
            yield (0 if sentiment == '积极' else 1 if sentiment == '消极' else 2,
                   row[-2])


def read_dataset(data_iter: Iterable[tuple[int, str]], to_predict: str = None):
    if to_predict is not None:
        data_iter = itertools.chain(data_iter, [(-1, to_predict)])
    data_iter = itertools.tee(data_iter)

    vectorizer = CountVectorizer()

    return (csr_matrix(vectorizer.fit_transform(' '.join(jieba.cut(row[1], cut_all=False)) for row in data_iter[0])),
            list(row[0] for row in data_iter[1]))


def be_lazy(no_cache: bool = False):
    model_path = "./models.pkl"
    if not no_cache and pathlib.Path(model_path).exists():
        print(lazyresult.read_pickles(model_path).to_string())
        return

    (x_train, y_train), (x_test, y_test) = read_dataset(__train_data_iter()), read_dataset(__test_data_iter())
    clf = LazyClassifier(predictions=True)
    models, _ = clf.fit(x_train.toarray(), x_test.toarray(), y_train, y_test)
    print(models.to_string())
    with open(model_path, "wb") as f:
        pickle.dump(models, f)


def predict(query: str):
    clf = Perceptron()
    data_set = read_dataset(to_predict=query)
    train_set_x, train_set_y = data_set[0][0:-2], data_set[1][0:-2]
    clf.fit(train_set_x, train_set_y)

    return clf.predict(data_set[0][-1])[0]


def user_interact():
    while True:
        sms = input("sms: ")
        if not sms:
            break
        if predict(sms) == 0:
            print("spam")
        else:
            print("ham")


def batch(fin, fout):
    reader = csv.reader(fin)
    writer = csv.writer(fout)
    for text in reader:
        if predict(text[0]) == 1:
            writer.writerow([text[0], "ham"])
        else:
            writer.writerow([text[0], "spam"])


if __name__ == '__main__':
    opt_parser = ArgumentParser()
    opt_parser.add_argument("-l", "--lazy", action='store_true')
    opt_parser.add_argument("-n", "--no-cache", action='store_true')
    opt_parser.add_argument("-i", "--input", type=argparse.FileType('r'))
    opt_parser.add_argument("output", type=argparse.FileType('w'), nargs='?')
    opts = opt_parser.parse_args()
    lazy = opts.lazy

    if lazy:
        be_lazy(opts.no_cache)
    elif opts.input is not None:
        if opts.output is not None:
            batch(opts.input, opts.output)
    else:
        user_interact()
