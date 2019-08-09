# -*- coding: utf-8 -*-
import os
from langconv import *
from config import config
from utils import code
from collections import OrderedDict


cur_path = os.path.dirname(os.path.abspath(__file__))
dict_path = os.path.join(cur_path, 'synonym', 'dict')

websize = re.compile(r'https?://[a-z0-9\43 -\176]*')
long_num = re.compile(r'[0-9]{5,}')
synonym_dict = OrderedDict()


def init_synonym():
    with open(dict_path, 'r') as fin:
        for line in fin.readlines():
            line = code.any2unicode(line)
            synonyms = line.strip().split('\t')
            if len(synonyms) < 2:
                continue
            normal_word = synonyms[0]
            synonyms = sorted(synonyms[1:], key=lambda x: len(x), reverse=True)
            for word in synonyms:
                synonym_dict[word] = normal_word


config.set_value("synonyms_dict", synonym_dict)


# query 改写
def rewrite(query):
    query = query.strip()
    query = query.lower()

    # 去除无用字符串，如网址、数字
    query = rm_useless_string(query)

    # 繁体转简体
    query = tradition2simple_chinese(query).encode('utf-8')

    # 同义词替换
    query = synonym_replace(query)
    return query


# 繁体中文转简体中文
def tradition2simple_chinese(query):
    return Converter('zh-hans').convert(query.decode('utf-8'))


def rm_useless_string(query):
    """
    去除无用的字符
    """
    # 去除query中网址
    query = code.any2unicode(query)
    query = re.sub(websize, '', query)
    query = re.sub(long_num, '', query)
    return query


def synonym_replace(query):
    query = code.any2unicode(query)
    for word, normal_word in synonym_dict.items():
        query = query.replace(word, normal_word)
    return query


init_synonym()

if __name__ == '__main__':
    pass
