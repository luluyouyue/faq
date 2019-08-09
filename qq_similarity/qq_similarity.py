#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =========================================================================
#
# Copyright (c) 2017 <> All Rights Reserved
#
#
# File: /Users/hain/ai/Synonyms/synonyms/__init__.py
# Author: Hai Liang Wang
# Date: 2017-09-27
#
# =========================================================================

"""
Chinese Synonyms for Natural Language Processing and Understanding.
"""
from __future__ import print_function
from __future__ import division

__copyright__ = "Copyright (c) 2017 . All Rights Reserved"
__author__ = "Hu Ying Xi<>, Hai Liang Wang<hailiang.hl.wang@gmail.com>"
__date__ = "2017-09-27"
__version__ = "3.3.10"

import os
import sys
import numpy as np
import math
import time


curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curdir)

PLT = 2

if sys.version_info[0] < 3:
    default_stdout = sys.stdout
    default_stderr = sys.stderr
    reload(sys)
    sys.stdout = default_stdout
    sys.stderr = default_stderr
    sys.setdefaultencoding("utf-8")
    # raise "Must be using Python 3"
else:
    PLT = 3

# Get Environment variables
ENVIRON = os.environ.copy()


from .word2vec import KeyedVectors
from .utils import any2utf8
from .utils import any2unicode
from .utils import sigmoid
from .utils import cosine
from .utils import is_digit

'''
globals
'''
_vocab = dict()
_size = 0
_vectors = None
_stopwords = set()
_cache_nearby = dict()

'''
lambda fns
'''
# combine similarity scores
_similarity_smooth = lambda x, y, z, u: (x * y) + z - u
_flat_sum_array = lambda x: np.sum(x, axis=0)  # 分子


'''
word embedding
'''
# vectors
_f_model = os.path.join(curdir, 'dict', 'words.vector')
if "SYNONYMS_WORD2VEC_BIN_MODEL_ZH_CN" in ENVIRON:
    _f_model = ENVIRON["SYNONYMS_WORD2VEC_BIN_MODEL_ZH_CN"]


def _load_w2v(model_file=_f_model, binary=True):
    '''
    load word2vec model
    '''
    if not os.path.exists(model_file):
        print("os.path : ", os.path)
        raise Exception("Model file [%s] does not exist." % model_file)
    return KeyedVectors.load_word2vec_format(
        model_file, binary=binary, unicode_errors='ignore')


print(">> Synonyms on loading vectors [%s] ..." % _f_model)
_vectors = _load_w2v(model_file=_f_model)


def _get_wv_test(sentence, ignore=False):
    '''
    get word2vec data by sentence
    sentence is segmented string.
    '''
    global _vectors
    vectors = []
    for y in sentence:
        y_ = any2unicode(y).strip()
        if y_ not in _stopwords:
            syns = nearby(y_)[0]
        # print("sentence %s word: %s" %(sentence, y_))
        # print("sentence %s word nearby: %s" %(sentence, " ".join(syns)))
        c = []
        try:
            c.append(_vectors.word_vec(y_))
        except KeyError as error:
            if ignore:
                continue
            else:
                # c.append(np.zeros((100,), dtype=float))
                random_state = np.random.RandomState(seed=(hash(y_) % (2 ** 32 - 1)))
                c.append(random_state.uniform(low=-10.0, high=10.0, size=(100,)))

        for n in syns:
            if n is None:
                continue
            try:
                v = _vectors.word_vec(any2unicode(n))
            except KeyError as error:
                # v = np.zeros((100,), dtype=float)
                random_state = np.random.RandomState(seed=(hash(n) % (2 ** 32 - 1)))
                v = random_state.uniform(low=10.0, high=10.0, size=(100,))
            c.append(v)
        r = np.average(c, axis=0)
        vectors.append(r)
    return vectors


def _get_sentence_vector(words, ignore=False):
    """
    get sentence embedding by sum words embedding
    """
    c = []
    for word in words:
        try:
            c.append(_vectors.word_vec(word))
        except:
            if ignore:
                continue
            else:
                # print('no word vector, words = ', word)
                random_state = np.random.RandomState(seed=(hash(word) % (2 ** 32 - 1)))
                c.append(random_state.uniform(low=-10.0, high=10.0, size=(100,)))
                # print('c[-1] = ', c[-1])
    return np.mean(c, axis=0)


def word2vec_sentence_similarity(s1, s2, ignore=False):
    return cosine(_get_sentence_vector(s1), _get_sentence_vector(s2))


def bm25():
    pass


def _get_wv(sentence, synonym, ignore=False):
    '''
    get word2vec data by sentence
    sentence is segmented string.
    '''
    global _vectors
    vectors = []
    for y in sentence:
        y_ = any2unicode(y).strip()
        # if y_ not in _stopwords:
        # syns = nearby(y_)[0]
        syns = synonym.get(y_, [])
        # print("sentence %s word: %s" %(sentence, y_))
        # print("sentence %s word nearby: %s" %(sentence, " ".join(syns)))
        c = []
        try:
            c.append(_vectors.word_vec(y_))
        except KeyError as error:
            if ignore:
                continue
            else:
                # c.append(np.zeros((100,), dtype=float))
                random_state = np.random.RandomState(seed=(hash(y_) % (2 ** 32 - 1)))
                c.append(random_state.uniform(low=-10.0, high=10.0, size=(100,)))

        for n in syns:
            if n is None: continue
            try:
                v = _vectors.word_vec(any2unicode(n))
            except KeyError as error:
                # v = np.zeros((100,), dtype=float)
                random_state = np.random.RandomState(seed=(hash(n) % (2 ** 32 - 1)))
                v = random_state.uniform(low=10.0, high=10.0, size=(100,))
            c.append(v)
        r = np.average(c, axis=0)
        vectors.append(r)
    return vectors


'''
Distance
'''


# Levenshtein Distance
def _levenshtein_distance(sentence1, sentence2):
    '''
    Return the Levenshtein distance between two strings.
    Based on:
        http://rosettacode.org/wiki/Levenshtein_distance#Python
    '''
    first = any2utf8(sentence1).decode('utf-8', 'ignore')
    second = any2utf8(sentence2).decode('utf-8', 'ignore')
    sentence1_len, sentence2_len = len(first), len(second)
    maxlen = max(sentence1_len, sentence2_len)
    if sentence1_len > sentence2_len:
        first, second = second, first

    distances = range(len(first) + 1)
    for index2, char2 in enumerate(second):
        new_distances = [index2 + 1]
        for index1, char1 in enumerate(first):
            if char1 == char2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(1 + min((distances[index1],
                                              distances[index1 + 1],
                                              new_distances[-1])))
        distances = new_distances
    levenshtein = distances[-1]
    d = float((maxlen - levenshtein) / maxlen)
    # smoothing
    s = (sigmoid(d * 6) - 0.5) * 2
    # print("smoothing[%s| %s]: %s -> %s" % (sentence1, sentence2, d, s))
    return s


# def sv(sentence, ignore=False):
#     '''
#     获得一个分词后句子的向量，向量以BoW方式组成
#     sentence: 句子是分词后通过空格联合起来
#     ignore: 是否忽略OOV，False时，随机生成一个向量
#     '''
#     return _get_wv(sentence, ignore=ignore)


# def v(word):
#     '''
#     获得一个词语的向量，OOV时抛出 KeyError 异常
#     '''
#     y_ = any2unicode(word).strip()
#     return _vectors.word_vec(y_)

def _nearby_levenshtein_distance_test(s1, s2):
    '''
    使用空间距离近的词汇优化编辑距离计算
    '''
    s1_len, s2_len = len(s1), len(s2)
    maxlen = s1_len
    if s1_len == s2_len:
        first, second = sorted([s1, s2])
    elif s1_len < s2_len:
        first = s1
        second = s2
        maxlen = s2_len
    else:
        first = s2
        second = s1

    ft = set()  # all related words with first sentence
    for x in first:
        ft.add(x)
        n, _ = nearby(x)
        for o in n[:10]:
            # print('0:', o)
            ft.add(o)

    scores = []
    for x in second:
        choices = [_levenshtein_distance(x, y) for y in ft]
        if len(choices) > 0:
            scores.append(max(choices))

    s = np.sum(scores) / maxlen if len(scores) > 0 else 0
    return s


def _nearby_levenshtein_distance(s1, s2, synonym):
    '''
    使用空间距离近的词汇优化编辑距离计算
    '''
    s1_len, s2_len = len(s1), len(s2)
    maxlen = s1_len
    if s1_len == s2_len:
        first, second = sorted([s1, s2])
    elif s1_len < s2_len:
        first = s1
        second = s2
        maxlen = s2_len
    else:
        first = s2
        second = s1

    ft = set()  # all related words with first sentence
    for x in first:
        ft.add(x)
        # n, _ = nearby(x)
        n = synonym.get(x, [])
        for o in n[:10]:
            # print('0:', o)
            ft.add(o)

    scores = []
    # print('ft = ', ft, 'sencond = ', second)
    for x in second:
        choices = [_levenshtein_distance(x, y) for y in ft]
        if len(choices) > 0: scores.append(max(choices))

    s = np.sum(scores) / maxlen if len(scores) > 0 else 0
    return s


def _similarity_distance_test(s1, s2, ignore):
    '''
    compute similarity with distance measurement
    '''
    g = 0.0
    try:
        g_ = cosine(_flat_sum_array(_get_wv_test(s1, ignore)), _flat_sum_array(_get_wv_test(s2, ignore)))
        if is_digit(g_):
            g = g_
    except:
        pass
    u = _nearby_levenshtein_distance_test(s1, s2)
    if u >= 0.99:
        r = 1.0
    # _similarity_smooth = lambda x, y, z, u: (x * y) + z - u
    elif u > 0.9:
        r = _similarity_smooth(g, 0.05, u, 0.05)
    elif u > 0.8:
        r = _similarity_smooth(g, 0.1, u, 0.2)
    elif u > 0.4:
        r = _similarity_smooth(g, 0.2, u, 0.15)
    elif u > 0.2:
        r = _similarity_smooth(g, 0.3, u, 0.1)
    else:
        r = _similarity_smooth(g, 0.4, u, 0)

    if r < 0: r = abs(r)
    r = min(r, 1.0)
    # fout.write('sent1 filter stopwords = ' + ' '.join(s1) + '\t' + 'sent2 filter stopwords = ' + ' '.join(s2) + '\n')
    # fout.write('cosine = ' + str(g) + '\t' + 'levenshtein_distance = ' + str(u) + '\t' + str(r) + '\n\n')
    return float("%.3f" % r), s1, s2, u, g


def _similarity_distance(s1, s2, synonym, ignore):
    '''
    compute similarity with distance measurement
    '''
    g = 0.0
    try:
        g_ = cosine(_flat_sum_array(_get_wv(s1, synonym, ignore)), _flat_sum_array(_get_wv(s2, synonym, ignore)))
        if is_digit(g_): g = g_
    except:
        pass
    u = _nearby_levenshtein_distance(s1, s2, synonym)
    # print('word2vec similarity = ', g, 'levenshtein = ', u)
    if u >= 0.99:
        r = 1.0
    elif u > 0.9:
        r = _similarity_smooth(g, 0.05, u, 0.05)
    elif u > 0.8:
        r = _similarity_smooth(g, 0.1, u, 0.2)
    elif u > 0.4:
        r = _similarity_smooth(g, 0.2, u, 0.15)
    elif u > 0.2:
        r = _similarity_smooth(g, 0.3, u, 0.1)
    else:
        r = _similarity_smooth(g, 0.4, u, 0)

    if r < 0: r = abs(r)
    r = min(r, 1.0)
    # fout.write('sent1 filter stopwords = ' + ' '.join(s1) + '\t' + 'sent2 filter stopwords = ' + ' '.join(s2) + '\n')
    # fout.write('cosine = ' + str(g) + '\t' + 'levenshtein_distance = ' + str(u) + '\t' + str(r) + '\n\n')
    return float("%.3f" % r)


'''
Public Methods
'''

def nearby(word):
    '''
    Nearby word
    '''
    start = time.clock()
    w = any2unicode(word)
    # read from cache
    if w in _cache_nearby: return _cache_nearby[w]

    words, scores = [], []
    try:
        for x in _vectors.neighbours(w):
            words.append(x[0])
            scores.append(x[1])
    except:
        pass  # ignore key error, OOV
    # put into cache
    _cache_nearby[w] = (words, scores)
    end = time.clock()
    # print("find nearby used = %f second" % (end-start))
    return words, scores


def compare_test(s1, s2, ignore=False, stopwords=False):
    """
    compare similarity
    s1 : sentence1
    s2 : sentence2
    ignore: True: ignore OOV words
            False: get vector randomly for OOV words
    """
    if s1 == s2:
        return 1.0, s1, s2, 1, 1

    s1_words = []
    s2_words = []

    # fout.write('sent1 = ' + ''.join(s1) + '\tsent2 = ' + ''.join(s2) + '\t' + '\n')
    # check stopwords
    if not stopwords:
        # start = time.clock()
        global _stopwords
        for x in s1:
            if not x in _stopwords:
                s1_words.append(x)
        for x in s2:
            if not x in _stopwords:
                s2_words.append(x)
        # s1_words = s1
        # s2_words = s2
        # end = time.clock()
        # print('filter stopwords used %f second' % (end-start))
    else:
        s1_words = s1
        s2_words = s2

    assert len(s1) > 0 and len(s2) > 0, "The length of s1 and s2 should > 0."
    return _similarity_distance_test(s1_words, s2_words, ignore)


def compare_by_sentence_vector(s1, s2, ignore=False):
    return word2vec_sentence_similarity(s1, s2, ignore)


def compare(s1, s2, synonym, ignore=False, stopwords=False):
    """
    compare similarity
    s1 : sentence1
    s2 : sentence2
    ignore: True: ignore OOV words
            False: get vector randomly for OOV words
    """
    s1_words = s1
    s2_words = s2
    # print('s1_words = ', '#'.join(s1_words), 's2_words = ', '#'.join(s2_words))
    if s1_words == s2_words:
        return 1.0
    assert len(s1) > 0 and len(s2) > 0, "The length of s1 and s2 should > 0."
    return _similarity_distance(s1_words, s2_words, synonym, ignore)


def display(word):
    print("'%s'近义词：" % word)
    o = nearby(word)
    assert len(o) == 2, "should contain 2 list"
    if len(o[0]) == 0:
        print(" out of vocabulary")
    for k, v in enumerate(o[0]):
        print("  %d. %s:%s" % (k + 1, v, o[1][k]))


class BM25(object):
    """
    参考文献: https://www.jianshu.com/p/1e498888f505
    """
    def __init__(self, filted_faqs):
        """
        :param answers:
        """
        answers = []
        uni_answer_ids = set()
        for key, words, _, _ in filted_faqs:
            uni_answer_ids.add(key)
            answer = [key, words]
            answers.append(answer)

        self.D = len(uni_answer_ids)
        self.avgdl = sum([len(answer) + 0.0 for answer in answers]) / self.D  # 文档平均长度
        self.answers = answers
        self.f = []  # 列表的每一个元素是一个dict，dict存储着一个文档中每个词的出现次数
        self.answer_words_join2id = {}  # 存储每个答案的id
        self.word2answer_ids = {}  # 存储每个单词出现的答案id
        self.df = {}  # 存储每个词及出现了该词的文档数量
        self.idf = {}  # 存储每个词的idf值
        self.k1 = 1.5
        self.b = 0.75
        self.init()

    def init(self):
        num = 0
        for answer_id, answer_words in self.answers:
            answer_words_join = '#'.join(answer_words)
            if answer_words_join in self.answer_words_join2id:
                continue
            self.answer_words_join2id[answer_words_join] = num
            num += 1
            tmp = {}
            for word in answer_words:
                # print('type word in answer = ', type(word))
                tmp[word] = tmp.get(word, 0) + 1  # 存储每个文档中每个词的出现次数
            self.f.append(tmp)
            for k in tmp.keys():
                ids = self.word2answer_ids.setdefault(k, set())
                if answer_id not in ids:
                    self.df[k] = self.df.get(k, 0) + 1
                    ids.add(answer_id)
        for k, v in self.df.items():
            self.idf[k] = math.log(self.D - v + 0.5) - math.log(v + 0.5)

    def sim(self, query, answer):
        score = 0
        index = self.answer_words_join2id.get('#'.join(answer), 0)
        for word in query:
            if word not in answer:
                continue
            d = len(answer)
            score += (self.idf[word] * self.f[index][word] * (self.k1 + 1)
                      / (self.f[index][word] + self.k1 * (1 - self.b + self.b * d
                                                          / self.avgdl)))
        return score


def main():
    display("人脸")
    display("NOT_EXIST")
    sent1 = ['验证码', '收不到']
    sent2 = ['收不到', '验证码']
    score = compare(sent1, sent2)
    print('score :', score)


if __name__ == '__main__':
    main()
