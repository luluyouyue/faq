# -*- coding: utf-8 -*-
# from qq_similarity.qq_similarity import BM25
import os
from utils import code

cur_dir = os.path.dirname(os.path.abspath(__file__))
keywords_path = os.path.join(cur_dir, 'keywords')
_keywords = set()


# def _load_keywords(keywords_path):
#     global _keywords
#     with open(keywords_path, 'r') as fin:
#         for line in fin.readlines():
#             line = code.any2unicode(line.strip())
#             _keywords.add(line.lower())
#
#
# print(">> Synonyms on loading stopwords [%s] ..." % keywords_path)
# _load_keywords(keywords_path)


def recall(faq_list, query):
    filted_faq = []
    for faq in faq_list:
        _, words, _, _ = faq
        if len(set(words) & set(query)) != 0:
            filted_faq.append(faq)
    return filted_faq


# def recall_by_bm25(faq_list, query):
#     filted_faq = []
#     bm25er = BM25(faq_list)
#     candidate = []
#     for faq in faq_list:
#         _, words, _, _ = faq
#         score = bm25er.sim(query, words)
#         candidate.append([score, faq])
#     candidate.sort(key=lambda x: x[0], reverse=True)
#
#     count = 0
#     for score, faq in candidate:
#         if count > 19:
#             break
#         filted_faq.append(faq)
#         count += 1
#     return filted_faq


# def find_keywords_from_query(query):
#     keywords = set()
#     for keyword in _keywords:
#         if keyword in query:
#             keywords.add(keyword)
#     return keywords
#
#
# def recall_by_keywords(faq_list, query):
#     """
#     舍弃这种做法：忘记配关键词的话，会带来错误结果。如:
#     标准问句: 收不到验证码。
#     query: 短信收不到验证码怎么办。
#     这种将出不了收不到验证码标准问句
#     """
#     filted_faq = []
#     keywords = find_keywords_from_query(query)
#     if len(keywords) == 0:
#         return recall(faq_list, query)
#
#     print 'recall_by_keywords'
#     for faq in faq_list:
#         key, words, _, _ = faq
#         if len(set(words) & set(keywords)) != 0:
#             print 'key = ', key, 'filted faq = ', '#'.join(words)
#             filted_faq.append(faq)
#     return filted_faq


if __name__ == '__main__':
    _load_keywords(keywords_path)
