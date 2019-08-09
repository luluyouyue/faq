# -*- coding: utf-8 -*-
import preprocess
from seg.seg import WordSeger
from qq_similarity.qq_similarity import word2vec_sentence_similarity
import os
from preprocess import rewrite
from preprocess import rm_useless_string
from utils import code
from seg.seg import WordSeger


cur_path = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    query1 = "aid 1347， 用户登录提示‘访问太频繁，请稍后再试’"
    query2 = "手机号登录返回“访问太频繁了”怎么处理？"
    query3 = "session 获取 uid 监控？"

    query1 = rewrite(query1)
    query2 = rewrite(query2)
    query3 = rewrite(query3)
    wordseger = WordSeger()
    s1 = wordseger.get_wordseg(query1)
    s2 = wordseger.get_wordseg(query2)
    s3 = wordseger.get_wordseg(query3)

    similarity1 = word2vec_sentence_similarity(s1, s2)
    similarity2 = word2vec_sentence_similarity(s1, s3)
    print 'query1 = ', query1
    print 'query2 = ', query2
    print 'query3 = ', query3

    print 'rewrite s1 = ', '#'.join(s1)
    print 'rewrite s2 = ', '#'.join(s2)
    print 'rewrite s3 = ', '#'.join(s3)
    print 'similarity = ', similarity1

    print 'similarity = ', similarity2
