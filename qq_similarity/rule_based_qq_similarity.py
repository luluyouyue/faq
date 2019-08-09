# -*- coding: utf-8 =*-
# 提供规则的方法强干预语义相似度的计算

import os
from passport_lark_bot_nlp_pyrpc.thrift_gen.toutiao.passport.lark_bot_nlp.ttypes import Answer
from seg.seg import WordSeger
from frame import logger
from .utils import any2unicode
from config import config

_rules = []
cur_path = os.path.dirname(os.path.abspath(__file__))
_rule_path = rule_path = os.path.join(cur_path, 'dict', 'rule')


def key_words_to_synonyms(keywords):
    """
    :return: 返回关键词替换后的同义词
    """
    synonyms_dict = config.get_value("synonyms_dict", None)
    replace_keywords = set()
    if synonyms_dict is None or len(synonyms_dict) == 0:
        logger.info("load synonyms dict error!")
        return set(keywords)
    else:
        logger.info("in rule based qq similarity, len(synonyms_dict) = %s", len(synonyms_dict))
    for word in keywords:
        replace_keywords.add(synonyms_dict.get(word, word))
    return replace_keywords


def load_rules(rule_path):
    conf = config.get_value("conf")
    try:
        algorithm_type = conf["algorithm"]["type"]
    except:
        algorithm_type = "mix"
    with open(rule_path, 'r') as fout:
        rule_id = 0
        for line in fout.readlines():
            line = any2unicode(line)
            items = line.strip().split('\t')
            if len(items) != 3:
                raise Exception("导入规则错误！, rule id = %d" % (rule_id))
            keywords = key_words_to_synonyms(map(lambda x: x.strip(), items[0].split('#')))
            answer_ids = map(lambda x: int(x.strip()), items[1].strip().split('|'))
            score = float(items[2].strip())
            if algorithm_type == 'word2vec':
                score = 1.0
            rule = [str(rule_id), keywords, answer_ids, score]
            _rules.append(rule)
            rule_id += 1


load_rules(_rule_path)
logger.info(">>>>load rule sucess, len rule = %d", len(_rules))


def find_answer_by_rule(query, filted_faq):
    """
    :param query:
    :param filted_faq:
    :return: 匹配一条规则后返回答案
    """
    def is_keywords_in_query(query, key_words):
        if len(key_words) != 0:
            flag = True
            query_words_join = ''.join(query)
            for keyword in key_words:
                if keyword not in query_words_join:
                    flag = False
                    break
            return flag
        else:
            return False

    logger.info('filted faq length in find answer by rule = %d', len(filted_faq))
    hit_count = 0
    answers = []
    for rule_id, key_words, answer_ids, score in _rules:
        if is_keywords_in_query(query, key_words) and (hit_count < 3):
            for answer_id in answer_ids:
                answer_string = find_answer_string_in_filted_faqs(answer_id, filted_faq)
                if answer_string is None:
                    continue
                answer = [answer_id, (score, answer_string, rule_id, '#'.join(key_words),)]
                answers.append(answer)
                hit_count += 1
    return answers


def find_answer_string_in_filted_faqs(answer_id, filted_faq):
    """
    :param answer_id: 答案id
    :param filted_faq: 过滤的faq
    :return: faq的字符串
    """
    for key, _, ori_query, _ in filted_faq:
        # logger.info("filted key = %d, answer = %s", key, ori_query)
        if key == answer_id:
            return ori_query
    return None


def old_find_answer_by_rule(query, filted_faq):
    """
    :param query:
    :param filted_faq:
    :return: 匹配一条规则后返回答案
    """
    logger.info('filted faq length in find answer by rule = %d', len(filted_faq))
    query_words = set(query)
    for rule_id, key_words, answer_id, score in _rules:
        if len(query_words & key_words) != len(key_words):
            continue
        for key, words, ori_query, faq_type in filted_faq:
            print 'query = ', '#'.join(query_words), len('#'.join(query_words))
            # print 'filted faq = ', '#'.join(words)
            print 'key words = ', '#'.join(key_words), len('#'.join(key_words))
            # for word in list(key_words)[0]:
            #     print 'key word type = ', type(word)
            # for word in list(query_words)[0]:
            #     print 'query word type = ', type(word)
            # for word in list(query_words)[0]:
            #     print 'filterd word type = ', type(word)

            answer_words = set(words)
            print 'filted faq = ', '#'.join(answer_words), len('#'.join(answer_words))
            if key == 1004:
                print 'find key'
            print 'answer_words & key_words = ', '#'.join(answer_words & key_words)
            if len(answer_words & key_words) == len(key_words):
                logger.info('match rule, rule id = %d, rule = %s', rule_id, '#'.join(key_words))
                logger.info('match query = %s, origin query = %s', '#'.join(answer_words), ori_query)
                answer = Answer(key, score, ori_query, )
                return answer
    return None


if __name__ == '__main__':
    query = "你好，为什么收不到验证码"
    clean_words = WordSeger.get_wordseg(query)

    filted_faq = [(1, ['收不到', '验证码'], '我为什么收不到验证码', 'push')]
    print find_answer_by_rule(clean_words, filted_faq)

