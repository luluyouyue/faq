# -*- coding: utf-8 -*-
import jieba
from config import config

import os
import sys
from utils import code
from frame import logger

cur_path = os.path.dirname(os.path.abspath(__file__))
# stopwords
_fin_stopwords_path = os.path.join(cur_path, 'stopwords.txt')
_fin_self_define_dict = os.path.join(cur_path, 'self_define_word_dict')
_stopwords = set()
_self_define_words = set()


def _load_define_dict(file_path):
    global _self_define_words
    if sys.version_info[0] < 3:
        words = open(file_path, 'r')
    else:
        words = open(file_path, 'r', encoding='utf-8')
    words = words.readlines()
    for w in words:
        if w.startswith("#"):
            continue
        _self_define_words.add(code.any2unicode(w).strip())
    logger.info("self_define_words = %d", len(_self_define_words))


def _load_stopwords(file_path):
    '''
    load stop words
    '''
    global _stopwords
    if sys.version_info[0] < 3:
        words = open(file_path, 'r')
    else:
        words = open(file_path, 'r', encoding='utf-8')
    stopwords = words.readlines()
    for w in stopwords:
        _stopwords.add(code.any2unicode(w).strip())
    _stopwords.add(" ")


print(">> Synonyms on loading stopwords [%s] ..." % _fin_stopwords_path)
_load_stopwords(_fin_stopwords_path)
# _load_define_dict(_fin_self_define_dict)
config.set_value('stopwords', _stopwords)

_load_define_dict(_fin_self_define_dict)


class WordSeger(object):
    def __init__(self):
        self.stopwords = _stopwords

    def __del__(self):
        pass

    @staticmethod
    def combine_self_define_word(words):
        """
        前项最大匹配算法：
        利用自定义字典修正分词错误，如虚拟 手机号 -> 虚拟手机号, toutiao . user . info -> toutiao.user.info
        """
        if len(words) < 1 or words is None:
            return words

        new_words = []
        start = 0
        max_match = -1
        end = 1
        combine_word = words[start]
        while start < len(words) - 1:
            if end < len(words):
                combine_word = combine_word + words[end]
            else:
                if max_match != -1:
                    new_words.append(''.join(words[start:max_match + 1]))
                    start = max_match + 1
                else:
                    new_words.append(words[start])
                    start += 1

                if start >= (len(words) - 1):
                    if start == len(words) - 1:
                        new_words.append(words[start])
                    break
                end = start + 1
                max_match = -1
                combine_word = words[start] + words[end]

            if combine_word not in _self_define_words:
                end += 1
            else:
                max_match = end
                end += 1
        return new_words

    @staticmethod
    def remove_stopwords(words):
        clean_words = []
        stopwords = config.get_value("stopwords")
        for word in words:
            if word not in stopwords:
                # print 'filter words', word
                clean_words.append(word)
        # print '#'.join(clean_words)
        # for word in clean_words:
        #     print 'word type = ', type(word)
        return clean_words

    @staticmethod
    def get_wordseg(sentence, mode='basic'):
        if mode == "basic":
            words = jieba.cut(sentence, cut_all=False)
        elif mode == "all":
            words = jieba.cut(sentence, cut_all=True)
        elif mode == "search":
            words = jieba.cut_for_search(sentence)
        else:
            raise Exception("wrong mode for wordseg(only support basic、all、search mode)")
        # print 'words before = ', '#'.join(words)
        words = WordSeger.combine_self_define_word(words)
        # print 'words after combine = ', '#'.join(words)
        return WordSeger.remove_stopwords(words)


if __name__ == "__main__":
    sent1 = "请问如何解绑账号呢？"
    wordseger = WordSeger()
    print '#'.join(wordseger.get_wordseg(sent1))

